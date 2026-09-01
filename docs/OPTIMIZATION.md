# Optimization analysis

Detail behind the summary in the README: memory-traffic arithmetic for each
variant, kernel-size sweeps, and where measurement diverged from prediction.

All measurements: RTX 5060 Laptop GPU (Blackwell, `sm_120`, 26 SMs, 48 KB shared
memory per block), CUDA 12.8, MSVC 19.44, Release, 1920×1080 unless stated. 30
untimed warmup iterations, median of per-frame samples, launch-error checking
only with no per-launch synchronize.

A note on what follows: timings are measured. Explanations involving cache
behaviour are inferences consistent with the measurements, not profiler results.
Nsight Compute was later run to collect occupancy and memory-throughput figures
(summarised in the README), but it was not used to confirm the cache hypotheses
below. Those remain marked as inferences where they appear.

---

## 1. Naive — direct 2D convolution

Every thread reads all K² neighbours from global memory. At the default 5×5 that
is 25 loads per output pixel.

```
16×16 block, 5×5 filter:
  256 outputs × 25 loads          = 6,400 global loads
  distinct data touched: 20×20    =   400 pixels
  redundancy factor                =    16×
```

**Measured:** 0.181 ms at 1080p.

## 2. Separable — two 1D passes

A 2D Gaussian is the outer product of two 1D Gaussians, so the K×K convolution
factors into a horizontal pass followed by a vertical one. Taps per output pixel
fall from K² to 2K.

```
K=5:   25 → 10 taps    predicted 2.5×
K=9:   81 → 18 taps    predicted 4.5×
K=15: 225 → 30 taps    predicted 7.5×
K=21: 441 → 42 taps    predicted 10.5×
```

**Measured blur time by kernel size** (constant-memory weights):

| ksize | naive | separable | measured | predicted (K/2) | ratio |
|---|---|---|---|---|---|
| 5 | 0.201 | 0.107 | 1.88× | 2.5× | 75% |
| 9 | 0.508 | 0.151 | 3.36× | 4.5× | 75% |
| 15 | 1.299 | 0.225 | 5.77× | 7.5× | 77% |
| 21 | 2.471 | 0.301 | 8.21× | 10.5× | 78% |

The speedup holds at 75–78% of prediction at every kernel size. The consistency
suggests a systematic effect rather than measurement noise. Three contributing
factors:

1. **The float intermediate.** Rounding between passes would quantize twice and
   make this variant disagree with the naive one by more than rounding, so the
   intermediate is `float` — 4 bytes per pixel where the naive kernel reads 1.
   A deliberate trade of some speed for numerical comparability.
2. **Two launches instead of one**, so fixed per-launch overhead is paid twice.
3. **The naive kernel likely never paid the full theoretical cost.** Its 25 loads
   per pixel overlap heavily between neighbouring threads, so caches plausibly
   absorb much of the redundancy. The predicted 2.5× assumes 25 independent DRAM
   accesses, which is unlikely to be what actually happens. *(Inference, not
   profiled.)*

## 3. Shared memory — tiling with halo regions

Each block loads its tile plus halo into `__shared__` memory once, synchronizes,
then computes from on-chip data.

```
16×16 tile + 5×5 filter (radius 2) → 20×20 shared region
  400 pixels loaded → 256 outputs produced

Because this variant is also separable, tiles are 1D-shaped:
  horizontal pass: 20×16 = 320 floats = 1,280 bytes
  vertical pass:   16×20 = 320 floats = 1,280 bytes
  Sobel (radius 1): 18×18 = 324 floats = 1,296 bytes
```

At 48 KB of shared memory per block these tiles use under 3% of the budget, so
tile size is bounded by occupancy rather than capacity.

**Halo loading.** A grid-stride loop over the tile extent:

```cuda
for (int i = threadIdx.x; i < tileW; i += blockDim.x)
```

With `blockDim.x = 16` and radius 2, `tileW = 20`: threads 0–15 load columns
0–15, then threads 0–3 loop again for 16–19. Chosen over the common "each thread
loads its own pixel, then threads 0..2R−1 load the halo" pattern because it needs
no separate halo index arithmetic and cannot leave a gap when the tile exceeds
twice the block width.

The `__syncthreads()` sits outside the bounds guard. Every thread in a block must
reach every barrier, so letting out-of-range threads return early would deadlock
the block.

**Blur speedup over naive, by resolution:**

| resolution | separable | shared |
|---|---|---|
| 640×480 | 1.38× | 1.53× |
| 1280×720 | 1.65× | 1.93× |
| 1920×1080 | 1.71× | **2.02×** |
| 3840×2160 | 1.26× | 1.44× |

**Tiling benefit grows with stencil size**, which is what the reuse argument
predicts:

| ksize | separable | shared | shared/separable |
|---|---|---|---|
| 5 | 0.107 | 0.094 | 1.14× |
| 9 | 0.151 | 0.106 | 1.42× |
| 15 | 0.225 | 0.122 | 1.84× |
| 21 | 0.301 | 0.152 | **1.98×** |

**The 4K regression.** At 3840×2160 the separable float intermediate is 33 MB.
The drop from 2.02× to 1.44× is consistent with the intermediate exceeding cache
capacity, so the extra traffic it introduces offsets more of what the tap
reduction saves. Confirming that this is the mechanism would require profiling
memory throughput and cache hit rates; it has not been done. Either way, the
measurement shows the optimization is workload-dependent, which a
single-resolution benchmark would have hidden.

## 4. Constant memory for filter weights

The tap index is uniform across a warp, so every thread reads the same weight at
the same time — the access pattern the constant broadcast cache is built for.

**Blur time, global weight buffer → `__constant__`:**

| ksize | naive | separable | shared |
|---|---|---|---|
| 5 | 0.214 → 0.201 (1.06×) | 0.111 → 0.107 (1.04×) | 0.103 → 0.094 (1.10×) |
| 9 | 0.554 → 0.508 (1.09×) | 0.160 → 0.151 (1.06×) | 0.130 → 0.106 (1.23×) |
| 15 | 1.428 → 1.299 (1.10×) | 0.242 → 0.225 (1.08×) | 0.169 → 0.122 (1.39×) |
| 21 | 2.721 → 2.471 (1.10×) | 0.325 → 0.301 (1.08×) | 0.224 → 0.152 (**1.47×**) |

Naive and separable gain a flat 6–10%; the tiled variant gains up to 47%. A
plausible reading is that once image data is staged in shared memory, weight
reads are the main remaining global traffic, so removing them removes a larger
share of what is left — whereas in the naive kernel they sit among 25 image loads
per pixel. *(Inference, not profiled.)*

Sobel is unaffected: its 3×3 coefficients are compile-time literals that nvcc
materialises as immediates, so there is no memory traffic to remove.

## 5. Tiled Sobel is slower than naive

| | naive | shared |
|---|---|---|
| Sobel, 1080p | **0.047 ms** | 0.068 ms |

Reproduced identically across three runs. The tile arithmetic looks favourable —
324 loads for 256 outputs versus 2,304 naive loads — but a 3×3 stencil has far
less overlap between neighbouring threads than a 5×5, so there is less reuse to
recover. Staging through shared memory adds a load loop, a `uint8`→`float`
conversion, and a block-wide barrier.

This is consistent with the naive kernel's redundant reads already being served
largely from cache, in which case shared memory replaces cheap hits with explicit
staging. Confirming that would require profiling. *(Inference, not profiled.)*

The consequence is that `shared` is marginally slower overall than `separable`
(0.209 vs 0.205 ms kernel total at 1080p), because the Sobel regression cancels
the blur gain. The fastest measured combination is shared-memory blur with naive
Sobel.

---

## What limits the pipeline

```
Kernel time     0.319 ms
GPU total       1.170 ms    + PCIe transfer      3.7× the kernel time
End-to-end      9.867 ms    + decode and encode   31× the kernel time
```

**Transfer costs more than compute.** H2D is 0.553 ms and D2H 0.271 ms against
0.319 ms of kernel time. The upload is larger because a BGR frame is 3 bytes per
pixel while the result is 1.

**Decode and encode dominate.** They are CPU-bound and account for roughly 8.7 ms
of the 9.867 ms end-to-end budget. This is why the honest end-to-end figure is
101 fps rather than 3,134 — the kernels stopped being the bottleneck several
optimizations ago. Hardware decode (NVDEC/NVENC) is the path to closing that gap.

**Against the CPU baseline.** OpenCV is SIMD-vectorised and multithreaded and was
not crippled for the comparison. Blur at 1080p:

| variant | GPU | OpenCV CPU | speedup |
|---|---|---|---|
| naive | 0.181 ms | 0.194 ms | 1.07× |
| separable | 0.106 ms | 0.194 ms | 1.83× |
| shared | 0.090 ms | 0.194 ms | **2.16×** |

The naive GPU blur barely beats well-vectorised CPU code at all; the optimizations
take the margin to 2.16×. Across the whole pipeline the speedup is 20.9× on
kernel time and 4.8× once PCIe transfers are included. The multithreaded OpenCV
CPU baseline is more sensitive to system load and can vary between runs, so
treat CPU-relative ratios as approximate. NPP is the primary GPU comparison.

---

## Measurement method

- 30 untimed warmup iterations before every measurement. Kernel time falls by
  more than 2× over the first ~100 frames as the GPU leaves its idle clock state,
  so a single warmup frame produces unstable numbers.
- Median of per-frame samples, not mean, so one frame delayed by OS scheduling
  does not shift the result.
- `cudaEvent` records bracketing the measured region with exactly one
  synchronize at its end. Host clocks are not used for device work.
- `kernel_ms` slightly over-estimates pure compute: uploads and downloads are
  synchronous copies from pageable memory, so the GPU is idle when the
  post-upload event is recorded and the first kernel's launch latency lands
  inside the span.

---

## Validation methodology

Three independent checks, all on 20 frames at 1920x1080 unless stated.

**Against OpenCV**, with matched kernel size, sigma and `BORDER_REFLECT_101`.
Because the CUDA kernels implement the same border mode, the comparison covers
the entire image with no excluded pixels.

```
stage        mean abs   max abs   px differing >1
Grayscale     0.00002         1                 0
Blur          0.00495         1                 0
Sobel         0.02470         8            15,817     PASSED (tolerance: mean <= 1.0)
```

The Sobel maximum of 8 is inherited rounding, not a defect: the Sobel weights sum
to 8 in absolute value, so a one-level difference in the blurred input can move
the gradient by up to 8 levels. Blur differs by at most 1 because OpenCV uses
fixed-point coefficients.

**Against a NumPy oracle** (`scripts/verify_stages.py`), written from the
mathematical definition in float64. OpenCV dispatches to fixed-point SIMD paths
that vary by build and CPU, so it is a useful reference but not a bit-exact one.
Against NumPy, all three stages are bit-exact.

**Variants against each other** (`scripts/compare_variants.py`) at 642x482 --
deliberately not a multiple of the 16x16 block, so partial edge tiles are
exercised. Separable and shared agree bit-exactly; naive differs by one level at
a single pixel, from floating-point associativity.

**Against NPP**, with two differences that cannot be eliminated:

- **Border mode.** NPP's convolution and Sobel accept only
  `NPP_BORDER_REPLICATE`; `MIRROR`, `CONSTANT` and `WRAP` return
  `NPP_NOT_SUPPORTED_MODE_ERROR`. Blur and Sobel are therefore compared on the
  interior with a 4-pixel margin excluded, 1.15% of pixels at 1080p. Grayscale
  reads no neighbours and is compared over the full image.
- **Rounding.** NPP truncates where these kernels round. Measured against a
  float64 reference, `NPP - round(exact)` is `-1` on 50.3% of pixels and `0` on
  the rest, never `+1`. These kernels round half to even because that matches
  OpenCV.

Because that rounding difference is systematic, the NPP check is gated on the
**maximum** difference against a derived bound rather than the mean: 1 level for
grayscale and blur, since rounding cannot exceed one level, and 8 for Sobel, the
sum of the Sobel weights. Measured at 1080p: max 1, 1 and 5. The blur result sits
exactly at its limit, so any further disagreement fails the check.

## Why two different kernel timings appear

The benchmark sweep reports 0.209 ms of kernel time at 1080p while the
end-to-end run reports 0.319 ms. They are different measurements: the sweep takes
a median over repeated passes on in-memory frames, the end-to-end run takes a
mean with video decode interleaved between frames. Both are real; neither
corrects the other.

4K end-to-end was never measured. The 4K rows in `results/benchmarks.csv` are
compute-only, on resized frames.

## NPP compared to the custom naive kernel

The most informative comparison for judging the custom code is `custom naive`
against NPP: same algorithm, same weights, same byte traffic, and NPP is 2.3x to
5.9x faster. That isolates the difference as implementation quality rather than
algorithm choice. The custom optimizations remain real and measurable -- blur
falls from 0.181 to 0.090 ms at 1080p, a 2.02x improvement -- but they do not
close the gap to NPP.
