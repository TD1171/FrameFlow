#!/usr/bin/env python3
"""Independent verification of every pipeline stage against a NumPy oracle.

This exists because a build that compiles and runs proves nothing about whether
the kernels compute the right thing. A kernel writing all zeros, or swapping the
blue and red channels, passes a smoke test perfectly.

The oracle is written from the mathematical definition in float64 NumPy, NOT by
calling OpenCV. cv2.cvtColor dispatches to SIMD paths with fixed-point
coefficients that vary by build and CPU, so it is not a stable bit-exact
reference. NumPy from first principles is.

Border handling is np.pad(mode="reflect"), which is exactly BORDER_REFLECT_101:
it mirrors about the edge pixel without repeating it (abc -> cba|abc|cba).
NumPy's "symmetric" mode is the one that repeats, and is NOT what we want.

Usage:
    python scripts/verify_stages.py --dir results --ksize 5 --sigma 1.4
Exits non-zero if any stage exceeds tolerance.
"""

import argparse
import os
import sys

import cv2
import numpy as np


def gaussian_1d(ksize, sigma):
    """Mirrors cv::getGaussianKernel and src/kernels.cu gaussian_kernel_1d."""
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    c = (ksize - 1) * 0.5
    x = np.arange(ksize, dtype=np.float64) - c
    w = np.exp(-0.5 * (x * x) / (sigma * sigma))
    return w / w.sum()


def convolve_reflect101(img, kernel2d):
    r = kernel2d.shape[0] // 2
    padded = np.pad(img.astype(np.float64), r, mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    for ky in range(kernel2d.shape[0]):
        for kx in range(kernel2d.shape[1]):
            w = kernel2d[ky, kx]
            if w != 0.0:
                out += w * padded[ky:ky + img.shape[0], kx:kx + img.shape[1]]
    return out


def report(name, gpu, expected, tol_mean, tol_max):
    diff = np.abs(gpu.astype(np.float64) - expected)
    mean, mx = diff.mean(), diff.max()
    over = int((diff > tol_max).sum())
    ok = mean <= tol_mean and over == 0
    print(f"  {name:<12} mean={mean:8.5f}  max={mx:5.1f}  "
          f"pixels>{tol_max}: {over:<8} {'OK' if ok else 'FAIL'}")
    return ok, mean, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--ksize", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=1.4)
    ap.add_argument("--tol-mean", type=float, default=1.0)
    ap.add_argument("--tol-max", type=float, default=3.0)
    args = ap.parse_args()

    paths = {k: os.path.join(args.dir, v) for k, v in {
        "orig": "stage0_original.png",
        "gray": "stage1_grayscale.png",
        "blur": "stage2_blurred.png",
        "edge": "stage3_edges.png",
    }.items()}

    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        sys.exit("error: missing stage PNGs (run with --save-frames first):\n  " +
                 "\n  ".join(missing))

    bgr = cv2.imread(paths["orig"], cv2.IMREAD_COLOR)
    gpu_gray = cv2.imread(paths["gray"], cv2.IMREAD_GRAYSCALE)
    gpu_blur = cv2.imread(paths["blur"], cv2.IMREAD_GRAYSCALE)
    gpu_edge = cv2.imread(paths["edge"], cv2.IMREAD_GRAYSCALE)
    if any(x is None for x in (bgr, gpu_gray, gpu_blur, gpu_edge)):
        sys.exit("error: one or more stage PNGs could not be decoded")

    h, w = gpu_gray.shape
    print(f"Verifying {w}x{h} against a NumPy float64 oracle "
          f"(ksize={args.ksize}, sigma={args.sigma})")

    # Stage 1 -- BT.601 on BGR bytes. Byte 0 is blue.
    B, G, R = (bgr[:, :, i].astype(np.float64) for i in range(3))
    exp_gray = np.rint(0.114 * B + 0.587 * G + 0.299 * R)

    # Stage 2 -- separable Gaussian as a full 2D outer product, reflect-101.
    k1 = gaussian_1d(args.ksize, args.sigma)
    exp_blur = np.rint(np.clip(convolve_reflect101(exp_gray, np.outer(k1, k1)), 0, 255))

    # Stage 3 -- Sobel magnitude on the blurred image, saturated to 8 bits.
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = convolve_reflect101(exp_blur, sx)
    gy = convolve_reflect101(exp_blur, sy)
    exp_edge = np.rint(np.clip(np.sqrt(gx * gx + gy * gy), 0, 255))

    results = [
        report("grayscale", gpu_gray, exp_gray, args.tol_mean, args.tol_max),
        report("blur", gpu_blur, exp_blur, args.tol_mean, args.tol_max),
        report("sobel", gpu_edge, exp_edge, args.tol_mean, args.tol_max),
    ]

    # Guard against a kernel that outputs a constant. Every stage above would
    # still "pass" a loose tolerance on a low-contrast image, and a blank output
    # is the single most likely way for a broken kernel to look plausible.
    print()
    degenerate = False
    for name, img in (("grayscale", gpu_gray), ("blur", gpu_blur), ("sobel", gpu_edge)):
        u = int(np.unique(img).size)
        rng = int(img.max()) - int(img.min())
        flag = "" if u > 8 else "   <-- SUSPICIOUS, nearly constant"
        print(f"  {name:<12} distinct values={u:<6} range={rng:<5}{flag}")
        if u <= 8:
            degenerate = True

    # Positive control: prove the grayscale comparison can detect a channel swap.
    swapped = np.rint(0.114 * R + 0.587 * G + 0.299 * B)
    swap_delta = float(np.abs(exp_gray - swapped).mean())
    print(f"\n  control: a B/R swap would shift grayscale by mean {swap_delta:.3f} "
          f"({'detectable' if swap_delta > args.tol_mean else 'NOT DETECTABLE - weak test image'})")

    ok = all(r[0] for r in results) and not degenerate and swap_delta > args.tol_mean
    print("\nVERDICT:", "ALL STAGES VERIFIED" if ok else "*** VERIFICATION FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
