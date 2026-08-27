#!/usr/bin/env python3
"""Plot the measured benchmark results.

Reads results/benchmarks.csv exactly as written by `frameflow --benchmark`.
Nothing here invents, smooths or extrapolates a number: if a configuration was
skipped during the sweep it is simply absent from the plots.

Usage:
    python scripts/plot_benchmarks.py [--csv results/benchmarks.csv] [--out results]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # no display needed; write PNGs directly
import matplotlib.pyplot as plt

VARIANTS = ["naive", "separable", "shared"]
STAGES = ["grayscale", "blur", "sobel"]
COLORS = {"naive": "#c44e52", "separable": "#dd8452", "shared": "#4c72b0"}
STAGE_COLORS = {"grayscale": "#8172b3", "blur": "#4c72b0", "sobel": "#55a868"}


def load(path):
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found. Run: frameflow --input <video> --benchmark")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["width"] = int(r["width"])
            r["height"] = int(r["height"])
            for k in ("kernel_ms", "gpu_total_ms", "cpu_ms", "speedup_kernel", "speedup_total"):
                r[k] = float(r[k])
            rows.append(r)
    if not rows:
        sys.exit(f"error: {path} contains no data rows")
    return rows


def resolutions_in_order(rows):
    seen = {}
    for r in rows:
        seen.setdefault(r["resolution"], r["width"] * r["height"])
    return sorted(seen, key=lambda k: seen[k])


def plot_time_vs_resolution(rows, out_dir):
    """Log y-axis: kernel times span more than an order of magnitude across
    resolutions, and a linear axis would flatten every small resolution into
    the same invisible line at the bottom."""
    res_order = resolutions_in_order(rows)
    totals = {(r["resolution"], r["variant"]): r["kernel_ms"]
              for r in rows if r["stage"] == "total"}
    cpu = {r["resolution"]: r["cpu_ms"] for r in rows if r["stage"] == "total"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(res_order))
    for v in VARIANTS:
        ys = [totals.get((res, v)) for res in res_order]
        pts = [(i, y) for i, y in zip(x, ys) if y is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    label=f"GPU {v}", color=COLORS[v], linewidth=2)

    ys = [cpu.get(res) for res in res_order]
    pts = [(i, y) for i, y in zip(x, ys) if y is not None]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", linestyle="--",
                label="CPU baseline (OpenCV)", color="#937860", linewidth=2)

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(res_order)
    ax.set_xlabel("resolution")
    ax.set_ylabel("time per frame (ms, log scale)")
    ax.set_title("Kernel time per frame vs resolution\n(compute only, excludes H2D/D2H transfer)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "bench_time_vs_resolution.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_speedup_vs_naive(rows, out_dir):
    """Speedup relative to the naive kernel -- the project's own baseline, not
    the CPU. This is the plot that shows what each optimisation bought."""
    res_order = resolutions_in_order(rows)
    blur = {(r["resolution"], r["variant"]): r["kernel_ms"]
            for r in rows if r["stage"] == "blur"}

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    plotted = [v for v in VARIANTS if v != "naive"]
    for i, v in enumerate(plotted):
        xs, ys = [], []
        for j, res in enumerate(res_order):
            base, cur = blur.get((res, "naive")), blur.get((res, v))
            if base and cur:
                xs.append(j + (i - (len(plotted) - 1) / 2) * width)
                ys.append(base / cur)
        if xs:
            bars = ax.bar(xs, ys, width, label=v, color=COLORS[v])
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}x",
                        ha="center", va="bottom", fontsize=9)

    ax.axhline(1.0, color="#c44e52", linestyle="--", linewidth=1.2,
               label="naive (baseline)")
    ax.set_xticks(range(len(res_order)))
    ax.set_xticklabels(res_order)
    ax.set_ylabel("blur speedup vs naive (higher is better)")
    ax.set_title("Blur speedup relative to the naive kernel")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "bench_speedup_vs_naive.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_stage_breakdown(rows, out_dir):
    """Stacked per-stage time. Shows which stage actually dominates, which is
    not obvious: blur is by far the most expensive, and optimising anything
    else first would have been wasted effort."""
    res_order = resolutions_in_order(rows)
    per_stage = defaultdict(dict)
    for r in rows:
        if r["stage"] in STAGES:
            per_stage[(r["resolution"], r["variant"])][r["stage"]] = r["kernel_ms"]

    labels, bottoms, series = [], [], defaultdict(list)
    for res in res_order:
        for v in VARIANTS:
            d = per_stage.get((res, v))
            if not d:
                continue
            labels.append(f"{res}\n{v}")
            for s in STAGES:
                series[s].append(d.get(s, 0.0))

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(labels))
    bottoms = [0.0] * len(labels)
    for s in STAGES:
        ax.bar(x, series[s], 0.7, bottom=bottoms, label=s, color=STAGE_COLORS[s])
        bottoms = [b + v for b, v in zip(bottoms, series[s])]

    for i, total in enumerate(bottoms):
        ax.text(i, total, f"{total:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("time per frame (ms)")
    ax.set_title("Where the time goes: per-stage kernel time by resolution and variant")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="stage")
    fig.tight_layout()
    p = os.path.join(out_dir, "bench_stage_breakdown.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/benchmarks.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = load(args.csv)
    os.makedirs(args.out, exist_ok=True)

    written = [p for p in (plot_time_vs_resolution(rows, args.out),
                           plot_speedup_vs_naive(rows, args.out),
                           plot_stage_breakdown(rows, args.out)) if p]
    for p in written:
        print(f"wrote {p}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
