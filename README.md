# FrameFlow

A CUDA C++ video processing pipeline with custom grayscale, Gaussian blur and Sobel kernels. I implemented the blur three ways — naive, separable, and shared-memory tiled — validated all of them against OpenCV, and benchmarked how each memory optimization affected performance.

OpenCV is used only for video I/O and the CPU baseline; all GPU processing uses custom CUDA kernels.

| Input | Sobel output |
|---|---|
| ![](results/demo_original.jpg) | ![](results/demo_edges.png) |

<sub>Source photograph by Tabitha Mort, [CC0 1.0](https://commons.wikimedia.org/wiki/File:Long_Exposure_Photography_of_City_Buildings.jpg) (public domain), cropped to 1920×1080.</sub>

Every stage, on the synthetic clip the repository generates for reproducible benchmarking and validation:

| Original | Grayscale | Blurred | Edges |
|---|---|---|---|
| ![](results/stage0_original.png) | ![](results/stage1_grayscale.png) | ![](results/stage2_blurred.png) | ![](results/stage3_edges.png) |

<sub>The test clip contains pure blue, green and red patches deliberately: under BT.601 they map to distinct greys, so a BGR/RGB channel swap becomes visible rather than plausible.</sub>

---

## Performance

RTX 5060 Laptop GPU (Blackwell, `sm_120`), CUDA 12.8, Release build, 1920×1080.

**Gaussian blur — the stage the optimizations target.** From the benchmark sweep:

| variant | ms/frame | vs naive |
|---|---|---|
| naive — direct 2D convolution | 0.181 | — |
| separable — two 1D passes | 0.106 | 1.71× |
| shared — tiled with halo regions | **0.090** | **2.02×** |

![Blur speedup vs naive](results/bench_speedup_vs_naive.png)

**Whole pipeline at 1080p, ms/frame:**

| | processing only | + PCIe transfer | + decode and encode |
|---|---|---|---|
| OpenCV CPU | 4.372 | — | — |
| custom CUDA | **0.209** | 0.902 | 9.867 (**101 fps**) |

<sub>Processing figures come from the benchmark sweep (median over in-memory frames); the end-to-end figure from a separate full-clip run with decode interleaved. See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).</sub>

The source clip is 30 fps, so end-to-end throughput of 101 fps is roughly 3.4×
real time including CPU decode and encode. Kernel-only throughput is about 10×
higher and describes GPU processing, not pipeline throughput — the two are not
interchangeable. Decode and encode dominate the end-to-end figure.

Full sweep, 3 variants × 4 resolutions: [results/benchmarks.csv](results/benchmarks.csv), plots in `results/`.

---

## How it works

```
Video decode (CPU, OpenCV → BGR frames)
      ↓
  Host → Device            one upload per frame
      ↓
BGR → Grayscale kernel     d_bgr  → d_gray
      ↓
Blur kernel                d_gray → d_blur     naive | separable | shared
      ↓
Sobel kernel               d_blur → d_edges    naive | shared
      ↓
  Device → Host            one download per frame
      ↓
Video encode (CPU)
```

Device buffers are allocated once and reused across frames. The three kernels chain through device memory, so a frame crosses PCIe exactly twice regardless of how many stages run.

---

## CUDA optimizations

**Naive.** Direct 2D convolution, every tap read from global memory. At 5×5 that is 25 loads per output pixel. A 16×16 block issues 6,400 loads while touching only a 20×20 = 400 pixel region.

**Separable.** A 2D Gaussian is the outer product of two 1D Gaussians, so the K×K convolution becomes a horizontal pass followed by a vertical one — 2K taps instead of K², or 10 instead of 25 at the default size. This is an algorithmic win, independent of memory layout, and the advantage grows with kernel size. It works because the Gaussian is separable; most filters are not.

**Shared memory.** Each block loads its 16×16 tile plus a 2-pixel halo into `__shared__` memory once, synchronizes, then computes from on-chip data. Tiles are ~1.3 KB against 48 KB of shared memory per block, so tile size is bounded by occupancy rather than capacity.

**Constant memory.** Filter weights moved to `__constant__` memory. The tap index is uniform across a warp, which is the access pattern the constant broadcast cache is designed for. This gained 6–10% for the naive and separable variants but up to 47% for the tiled one — plausibly because once image data is in shared memory, weight reads are the main remaining global traffic.

### Result that went the other way

Shared-memory tiling made **Sobel slower**, 0.068 ms against 0.047 ms for the naive version, reproduced across three runs. A 3×3 stencil has much less overlap between neighbouring threads than a 5×5, so there is less reuse to recover, and staging through shared memory adds a load loop, a type conversion and a block-wide barrier. This is consistent with the redundant reads already being served largely from cache, though confirming that would require profiling.

The consequence is that the `shared` variant is marginally slower overall than `separable` (0.209 vs 0.205 ms kernel total), since the Sobel regression cancels the blur gain. The fastest measured combination is shared-memory blur with naive Sobel.

Detailed memory-traffic analysis, kernel-size sweeps and theory-vs-measurement comparison: **[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)**.

---

## GPU baseline: NVIDIA NPP

An OpenCV **CPU** baseline only shows that a GPU beats a CPU. This compares the
custom kernels against a vendor-optimised **GPU** library.

OpenCV CUDA was unavailable in the installed Windows build, so **NVIDIA NPP**
was used as the GPU-library baseline. NPP convolves with this project's own
Gaussian weights and uses the same timing, warmup and median statistic.

**1920×1080, ms/frame:**

| | grayscale | blur | sobel | compute | + transfer |
|---|---|---|---|---|---|
| **NPP** | 0.045 | **0.038** | 0.061 | **0.144** | **0.836** |
| custom naive | 0.053 | 0.181 | **0.047** | 0.280 | 0.982 |
| custom separable | 0.053 | 0.106 | **0.047** | 0.205 | 0.972 |
| custom shared | 0.051 | 0.090 | 0.068 | 0.209 | 0.902 |

**NPP ÷ custom** — above 1 means the custom kernel is faster:

| resolution | blur | sobel | compute | + transfer |
|---|---|---|---|---|
| 640×480 | 0.65 | **1.80** | 1.00 | **1.05** |
| 1280×720 | 0.48 | **1.43** | 0.82 | 0.97 |
| 1920×1080 | 0.43 | **1.30** | 0.70 | 0.93 |
| 3840×2160 | 0.24 | **1.72** | 0.58 | 0.89 |

**NPP wins the full compute pipeline from 720p upward** — 1.4× at 1080p, 1.7× at
4K — driven by its Gaussian blur. **The custom fused Sobel is faster at every
resolution, by 1.30× to 1.80×**, because NPP has no fused gradient-magnitude call
and needs two passes plus a combine step. Including PCIe transfers narrows the
difference to within roughly 10%.

Detailed methodology and analysis: [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).

## Profiling

Nsight Compute 2026.2.1, shared variant at 1080p:

| kernel | achieved occupancy | notes |
|---|---|---|
| grayscale | ~75% | ~95 GB/s; L1TEX and scoreboard stalls |
| blur, horizontal | ~86% | ~65% compute utilisation |
| blur, vertical | ~87% | ~99 GB/s |
| sobel | ~90% | |

At 75–90%, occupancy is not the limit. The latency and memory-dependency stalls
point at memory access patterns, and the separable float32 intermediate moves
20.74 MB per 1080p frame against NPP's 4.15 MB. **No optimization has been
applied on the basis of these numbers.**

## Validation

Correctness is checked three ways: against **OpenCV**, against a **NumPy
reference** written from the mathematical definition in float64, and against
**NVIDIA NPP**.

- Against OpenCV the comparison covers the full image with no excluded pixels, because the CUDA kernels implement the same `BORDER_REFLECT_101` border mode. Mean absolute error at 1080p: 0.00002 grayscale, 0.00495 blur, 0.02470 Sobel.
- Against the NumPy reference all three stages are **bit-exact**.
- The three kernel variants are checked against each other at a resolution that is not a multiple of the thread-block size, so partial edge tiles are exercised. Separable and shared agree bit-exactly.
- NPP comparisons account for differences in border handling and rounding, and are gated on a derived maximum-difference bound rather than a mean.

Methodology, error bounds and the NPP semantic differences are documented in
[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).

---

## Build and run

**Requirements:** NVIDIA GPU and matching driver · CUDA Toolkit 12.x (12.8+ for `sm_120`) · CMake 3.24+ · C++17 compiler (MSVC on Windows — `nvcc` requires it) · OpenCV 4.x with `videoio` · Python 3 with `numpy`, `matplotlib`, `opencv-python`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Targets `sm_120` by default; override with `-DCMAKE_CUDA_ARCHITECTURES=86`. On Windows, add `-DOpenCV_DIR=C:/path/to/opencv/build` if OpenCV is not at `C:/opencv/build`.

Generate a test clip (the repository ships no video files):

```bash
python scripts/generate_test_video.py --output data/sample.mp4 --width 1920 --height 1080 --frames 300
```

```bash
./frameflow --input data/sample.mp4 --output results/edges.mp4 --kernel shared
./frameflow --input data/sample.mp4 --frames 20 --validate
./frameflow --input data/sample.mp4 --benchmark
python scripts/plot_benchmarks.py
```

| Flag | |
|---|---|
| `--input <path>` | source video (required) |
| `--output <path>` | destination video; omit to skip encoding |
| `--kernel {naive,separable,shared}` | blur variant, default `shared` |
| `--frames N` | process only the first N frames |
| `--ksize N`, `--sigma S` | blur parameters, default 5 and 1.4 |
| `--warmup N` | untimed warmup iterations, default 30 |
| `--validate` | run the CPU baseline alongside and report error |
| `--benchmark` | sweep variants × resolutions, write CSV |
| `--save-frames`, `--frames-dir D` | dump per-stage PNGs |
| `--debug-sync` | synchronize after every kernel launch |

**Tests.** One command builds and runs every check, exiting non-zero on failure:

```bash
powershell -ExecutionPolicy Bypass -File scripts/build_and_test.ps1
```

Covers correctness against the NumPy oracle at aligned and unaligned resolutions, three-way variant agreement, and argument rejection.

Benchmark numbers were produced with launch-error checking only and no per-launch synchronize, using `cudaEvent` records with one synchronize at the end of the measured region. `--validate` and Debug builds synchronize after every launch instead, and label their timings as unsuitable for benchmarking.

---

## Project structure

```
src/kernels.cu        all CUDA kernels
src/pipeline.cu       device memory, kernel orchestration, event timing
src/main.cpp          CLI and frame loop
src/video_io.cpp      OpenCV decode/encode
src/cpu_baseline.cpp  OpenCV CPU reference
src/benchmark.cpp     resolution × variant sweep
scripts/              test clip generator, verification, plotting, test suite
results/              benchmark CSV, plots, sample frames
```

## Technologies

C++17 · CUDA C++ (Runtime API) · CMake · NVCC · OpenCV (I/O and CPU baseline only) · Python (NumPy, Matplotlib)

## Future work

- **NVDEC/NVENC hardware codecs** — decode and encode are roughly 88% of end-to-end time, making this the highest-value change
- **Pinned memory and CUDA streams** to overlap transfer with compute; PCIe transfer currently costs more than the kernels
- **Reduce the separable intermediate** from float32 to `uint8` or `__half`, cutting the 20.74 MB/frame the blur currently moves; the NPP comparison points here first
- **Process multiple output pixels per thread** in the blur kernels, to close part of the gap against NPP's 2D convolution
- **Full Canny** — non-maximum suppression and hysteresis for thin, connected edges
