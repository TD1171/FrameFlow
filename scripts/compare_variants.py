#!/usr/bin/env python3
"""Compare the output of two or more kernel variants against each other.

Every variant computes the same mathematical result by a different memory
strategy, so they must agree. Checking each variant against the NumPy oracle
individually is necessary but not sufficient for the thing that actually goes
wrong: a halo bug in a tiled kernel produces errors confined to tile borders,
which can be a small enough fraction of the image to hide inside a
whole-image mean while being catastrophically wrong where it occurs.

So this reports the WORST pixel and where it is, not just the average, and
fails on any single pixel exceeding tolerance.

Usage:
    python scripts/compare_variants.py a.png b.png [c.png ...] --tol 1
"""

import argparse
import itertools
import os
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="PNG per variant, named <variant>.png")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="max allowed absolute difference at any single pixel")
    args = ap.parse_args()

    if len(args.images) < 2:
        sys.exit("error: need at least two images to compare")

    loaded = {}
    for p in args.images:
        if not os.path.exists(p):
            sys.exit(f"error: missing {p}")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"error: could not decode {p}")
        loaded[os.path.splitext(os.path.basename(p))[0]] = img

    shapes = {v.shape for v in loaded.values()}
    if len(shapes) != 1:
        sys.exit(f"error: images differ in size: {shapes}")

    print(f"Comparing {len(loaded)} variants at {next(iter(shapes))[1]}x"
          f"{next(iter(shapes))[0]}, tolerance {args.tol} at any pixel")

    ok = True
    for (na, a), (nb, b) in itertools.combinations(loaded.items(), 2):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        mx = int(d.max())
        n_over = int((d > args.tol).sum())
        where = np.unravel_index(int(d.argmax()), d.shape) if mx > 0 else (0, 0)
        status = "OK" if n_over == 0 else "FAIL"
        print(f"  {na:<12} vs {nb:<12} mean={d.mean():8.5f} max={mx:4d} "
              f"at (row={where[0]},col={where[1]})  px>{args.tol}: {n_over:<8} {status}")
        if n_over:
            ok = False

    # A blank image would trivially agree with another blank image, so confirm
    # the variants are producing real content before trusting the agreement.
    print()
    for name, img in loaded.items():
        u = int(np.unique(img).size)
        print(f"  {name:<12} distinct values={u:<5}"
              f"{'   <-- SUSPICIOUS, nearly constant' if u <= 8 else ''}")
        if u <= 8:
            ok = False

    print("\nVERDICT:", "VARIANTS AGREE" if ok else "*** VARIANTS DISAGREE ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
