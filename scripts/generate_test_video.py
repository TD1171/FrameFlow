#!/usr/bin/env python3
"""Generate a synthetic test clip for the FrameFlow pipeline.

The repository ships no video files: this script draws one procedurally, so the
project is reproducible with no external assets and no licensing questions.

The content is chosen to exercise each pipeline stage rather than to look nice:

  smooth gradients      Blur has something to smooth and Sobel should report a
                        near-zero gradient there. A bright band in this region
                        means the Sobel kernel is wrong.
  high-contrast shapes  Sharp polygon/circle borders give strong, unambiguous
                        edges that are easy to check by eye.
  fine detail           A checkerboard and a fan of thin lines. A correct 5x5
                        Gaussian visibly attenuates these; a broken blur either
                        leaves them crisp or erases everything.
  colour separation     Pure-blue, pure-green and pure-red patches. Under
                        BT.601 these must map to clearly different greys
                        (~29, ~150, ~76), which catches BGR/RGB channel swaps
                        that are otherwise invisible in the output.
  motion                Content moves every frame, so per-frame timings are not
                        measuring one cached, unchanging frame.

Usage:
    python scripts/generate_test_video.py --output data/sample.mp4 \
        --width 1920 --height 1080 --frames 300 --fps 30
"""

import argparse
import os
import sys

import cv2
import numpy as np


def build_background(width, height):
    """Diagonal gradient: large smooth areas where the edge map should stay dark."""
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    diag = np.clip(0.5 * xs + 0.5 * ys, 0.0, 1.0)

    bg = np.empty((height, width, 3), dtype=np.float32)
    bg[:, :, 0] = 40 + 150 * diag              # B
    bg[:, :, 1] = 30 + 120 * (1.0 - diag)      # G
    bg[:, :, 2] = 60 + 90 * np.abs(diag - 0.5) * 2.0  # R
    return bg.astype(np.uint8)


def draw_checkerboard(frame, x0, y0, size, cell):
    """Fine detail. Cell size is deliberately near the blur radius."""
    h, w = frame.shape[:2]
    x1, y1 = min(x0 + size, w), min(y0 + size, h)
    if x0 >= x1 or y0 >= y1:
        return
    patch = frame[y0:y1, x0:x1]
    ph, pw = patch.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw]
    mask = (((yy // cell) + (xx // cell)) % 2).astype(bool)
    patch[mask] = 245
    patch[~mask] = 10


def draw_colour_patches(frame, width, height):
    """Pure B/G/R blocks: BT.601 must turn these into distinctly different greys."""
    size = max(24, min(width, height) // 14)
    pad = size // 3
    y0 = pad
    for i, colour in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        x0 = pad + i * (size + pad)
        if x0 + size < width and y0 + size < height:
            cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), colour, -1)


def draw_line_fan(frame, cx, cy, radius, count, angle_offset):
    """Thin one-pixel lines radiating outward: the finest detail in the clip."""
    for i in range(count):
        theta = angle_offset + (2.0 * np.pi * i / count)
        x = int(cx + radius * np.cos(theta))
        y = int(cy + radius * np.sin(theta))
        cv2.line(frame, (int(cx), int(cy)), (x, y), (255, 255, 255), 1, cv2.LINE_8)


def rotated_square(cx, cy, half, angle):
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]],
                       dtype=np.float32)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = corners @ rot.T + np.array([cx, cy], dtype=np.float32)
    return pts.astype(np.int32)


def render_frame(idx, total, width, height, background):
    frame = background.copy()
    t = idx / max(total - 1, 1)
    scale = min(width, height)

    # Rotating filled square: long straight high-contrast borders.
    half = int(scale * 0.13)
    cx = int(width * (0.25 + 0.5 * (0.5 - 0.5 * np.cos(2 * np.pi * t))))
    cy = int(height * 0.55)
    cv2.fillPoly(frame, [rotated_square(cx, cy, half, 2 * np.pi * t)], (250, 250, 250))
    cv2.polylines(frame, [rotated_square(cx, cy, half, 2 * np.pi * t)], True, (5, 5, 5),
                  max(2, scale // 400))

    # Two circles crossing the frame out of phase.
    r = int(scale * 0.07)
    for k, phase in enumerate((0.0, 0.5)):
        px = int(width * ((t + phase) % 1.0))
        py = int(height * (0.30 + 0.12 * np.sin(2 * np.pi * (t + phase) * 2)))
        cv2.circle(frame, (px, py), r, (15, 15, 15) if k == 0 else (240, 240, 240), -1)
        cv2.circle(frame, (px, py), r, (255, 255, 255) if k == 0 else (0, 0, 0),
                   max(2, scale // 500))

    # Hard vertical bar sweeping horizontally: a clean step edge at every row.
    bar_w = max(4, scale // 90)
    bar_x = int(width * ((1.0 - t) % 1.0))
    cv2.rectangle(frame, (bar_x, 0), (min(bar_x + bar_w, width - 1), height - 1),
                  (255, 255, 255), -1)

    cell = max(2, scale // 220)
    board = max(40, scale // 5)
    draw_checkerboard(frame, int(width * 0.66), int(height * 0.62), board, cell)

    draw_line_fan(frame, width * 0.18, height * 0.82, scale * 0.11, 24, 2 * np.pi * t)

    draw_colour_patches(frame, width, height)
    return frame


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic FrameFlow test clip.")
    ap.add_argument("--output", default="data/sample.mp4", help="output video path")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.width < 64 or args.height < 64:
        sys.exit("error: --width and --height must be at least 64")
    if args.frames < 1:
        sys.exit("error: --frames must be at least 1")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        sys.exit(f"error: could not open VideoWriter for '{args.output}' "
                 f"(missing codec support?)")

    background = build_background(args.width, args.height)
    try:
        for i in range(args.frames):
            writer.write(render_frame(i, args.frames, args.width, args.height, background))
            if args.frames >= 20 and (i + 1) % max(1, args.frames // 10) == 0:
                pct = 100.0 * (i + 1) / args.frames
                print(f"  {i + 1}/{args.frames} frames ({pct:.0f}%)", flush=True)
    finally:
        writer.release()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output}: {args.width}x{args.height}, {args.frames} frames "
          f"@ {args.fps:g} fps ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
