#include "kernels.cuh"

#include "cuda_check.h"

#include <cmath>
#include <stdexcept>

namespace frameflow {

// Defined here rather than in a header so there is exactly one instance.
// Debug builds default to per-launch synchronization; see cuda_check.h.
#ifdef FRAMEFLOW_DEBUG_SYNC
bool g_debug_sync = true;
#else
bool g_debug_sync = false;
#endif

// ---------------------------------------------------------------------------
// Host-side filter generation
// ---------------------------------------------------------------------------

std::vector<float> gaussian_kernel_1d(int ksize, double sigma) {
    if (ksize < 1 || ksize % 2 == 0) {
        throw std::runtime_error("gaussian kernel size must be positive and odd");
    }
    if (sigma <= 0.0) {
        // Same fallback OpenCV uses when sigma is unspecified, so the two agree
        // even when the caller does not supply one.
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
    }

    std::vector<float> w(static_cast<size_t>(ksize));
    const double centre = (ksize - 1) * 0.5;
    const double scale = -0.5 / (sigma * sigma);

    // Accumulate in double, then normalize. Normalizing by the actual sum
    // rather than the analytic 1/(sigma*sqrt(2pi)) matters: the kernel is
    // truncated at ksize taps, so the analytic constant would not sum to 1 and
    // would darken or brighten the whole image.
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i) {
        const double x = i - centre;
        const double t = std::exp(scale * x * x);
        w[static_cast<size_t>(i)] = static_cast<float>(t);
        sum += t;
    }
    for (int i = 0; i < ksize; ++i) {
        w[static_cast<size_t>(i)] = static_cast<float>(w[static_cast<size_t>(i)] / sum);
    }
    return w;
}

std::vector<float> gaussian_kernel_2d(int ksize, double sigma) {
    const std::vector<float> w1 = gaussian_kernel_1d(ksize, sigma);
    std::vector<float> w2(static_cast<size_t>(ksize) * static_cast<size_t>(ksize));
    for (int y = 0; y < ksize; ++y) {
        for (int x = 0; x < ksize; ++x) {
            w2[static_cast<size_t>(y) * ksize + x] = w1[static_cast<size_t>(y)] *
                                                     w1[static_cast<size_t>(x)];
        }
    }
    return w2;
}

namespace {

// BT.601 luminance weights, indexed to match BGR byte order.
constexpr float kWeightB = 0.114f;
constexpr float kWeightG = 0.587f;
constexpr float kWeightR = 0.299f;

// ---------------------------------------------------------------------------
// Border handling: BORDER_REFLECT_101
//
// Pixels at the image edge have no neighbours, and the choice made here must
// match the CPU baseline or validation reports an error that has nothing to do
// with kernel correctness. OpenCV's convolution default is BORDER_REFLECT_101,
// which mirrors about the edge pixel WITHOUT repeating it:
//
//     row:            a b c d e
//     extended:   c b|a b c d e|d c
//
// so index -1 maps to 1 and index n maps to n-2. (Plain REFLECT would give
// a b|a b c d e|e d -- repeating the edge sample. Getting this wrong shifts
// every border pixel by one source sample.)
//
// The loop handles the case where a tap reaches past the mirrored range too,
// which only occurs for images narrower than the filter. n == 1 is guarded
// because it would otherwise spin forever.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

__device__ __forceinline__ uint8_t to_u8(float v) {
    // rintf rounds half to even, matching OpenCV's rounding, so the CPU/GPU
    // comparison is not offset by a systematic half-level bias.
    return static_cast<uint8_t>(fminf(fmaxf(rintf(v), 0.0f), 255.0f));
}

// ---------------------------------------------------------------------------
// One thread per output pixel. Purely element-wise: no neighbour access, no
// shared memory, nothing to synchronize. This is the warm-up kernel whose job
// is to establish the indexing and bounds-checking pattern the convolution
// kernels reuse.
//
// Thread mapping: x comes from threadIdx.x so consecutive threads in a warp
// handle consecutive columns. Images are row-major, so those threads then touch
// consecutive addresses and the warp's loads coalesce into a small number of
// transactions. Mapping x to threadIdx.y instead would make each thread in a
// warp stride a whole row apart, turning one coalesced access into 32 separate
// ones.
// ---------------------------------------------------------------------------
__global__ void bgrToGrayscale(const uint8_t* __restrict__ bgr,
                               uint8_t* __restrict__ gray,
                               int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid dimensions are rounded up, so the last block in each dimension runs
    // threads past the image edge. Without this guard they would read and write
    // out of bounds -- the defect that hides behind power-of-two test images.
    if (x >= width || y >= height) return;

    const int src = (y * width + x) * 3;
    const float b = static_cast<float>(bgr[src + 0]);
    const float g = static_cast<float>(bgr[src + 1]);
    const float r = static_cast<float>(bgr[src + 2]);

    gray[y * width + x] = to_u8(kWeightB * b + kWeightG * g + kWeightR * r);
}

// ---------------------------------------------------------------------------
// Gaussian blur, naive variant: direct 2D convolution from global memory.
//
// Memory traffic, worked out for the default 5x5 filter: each thread issues
// ksize*ksize = 25 global loads to produce one output pixel. A 16x16 block
// produces 256 outputs and therefore issues 256*25 = 6,400 global loads, even
// though the distinct data it touches is only a 20x20 region = 400 pixels. The
// same bytes cross the bus roughly 16x more often than strictly necessary.
//
// In practice the L1/L2 caches absorb much of that redundancy, which is exactly
// why the measured speedup from tiling is smaller than 6400/400 would suggest.
// The tiled variant in a later stage makes the reuse explicit instead of
// hoping the cache catches it.
//
// Weights are read from global memory here on purpose. Moving them to
// __constant__ memory is a separate optimization, measured separately, so its
// contribution is not silently folded into the tiling result.
// ---------------------------------------------------------------------------
__global__ void gaussianBlurNaive(const uint8_t* __restrict__ src,
                                  uint8_t* __restrict__ dst,
                                  const float* __restrict__ weights,
                                  int ksize, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int radius = ksize / 2;
    float acc = 0.0f;

    for (int ky = -radius; ky <= radius; ++ky) {
        const int sy = reflect101(y + ky, height);
        for (int kx = -radius; kx <= radius; ++kx) {
            const int sx = reflect101(x + kx, width);
            acc += weights[(ky + radius) * ksize + (kx + radius)] *
                   static_cast<float>(src[sy * width + sx]);
        }
    }

    dst[y * width + x] = to_u8(acc);
}

// ---------------------------------------------------------------------------
// Sobel edge detection, naive variant.
//
// Standard 3x3 operators. Sign convention matches cv::Sobel, so gx is positive
// for a left-to-right increase in intensity:
//
//     gx = [-1  0  1]        gy = [-1 -2 -1]
//          [-2  0  2]             [ 0  0  0]
//          [-1  0  1]             [ 1  2  1]
//
// The result is the true magnitude sqrt(gx^2 + gy^2), not the |gx| + |gy|
// approximation. The approximation is cheaper but overestimates diagonal edges
// by up to 41%, which would show up as a real disagreement against an OpenCV
// baseline rather than as rounding.
//
// SATURATION: gx and gy each span [-1020, 1020] for 8-bit input, so the
// magnitude can reach ~1443 while the output is 8-bit. Values above 255 are
// clamped, not scaled. Strong edges therefore saturate to white and lose their
// relative ordering. This is deliberate -- it matches what the CPU baseline
// does when converting to CV_8U -- but it means the output is an edge map, not
// a quantitative gradient field.
// ---------------------------------------------------------------------------
__global__ void sobelNaive(const uint8_t* __restrict__ src,
                           uint8_t* __restrict__ dst,
                           int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float kx3[3][3] = {{-1.f, 0.f, 1.f}, {-2.f, 0.f, 2.f}, {-1.f, 0.f, 1.f}};
    const float ky3[3][3] = {{-1.f, -2.f, -1.f}, {0.f, 0.f, 0.f}, {1.f, 2.f, 1.f}};

    float gx = 0.0f;
    float gy = 0.0f;

    for (int dy = -1; dy <= 1; ++dy) {
        const int sy = reflect101(y + dy, height);
        for (int dx = -1; dx <= 1; ++dx) {
            const int sx = reflect101(x + dx, width);
            const float v = static_cast<float>(src[sy * width + sx]);
            gx += kx3[dy + 1][dx + 1] * v;
            gy += ky3[dy + 1][dx + 1] * v;
        }
    }

    dst[y * width + x] = to_u8(sqrtf(gx * gx + gy * gy));
}

// 16x16 = 256 threads: a multiple of the 32-thread warp size, and small enough
// to keep several blocks resident per SM for latency hiding.
constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

inline dim3 grid_for(int width, int height) {
    return dim3(static_cast<unsigned>((width + kBlockX - 1) / kBlockX),
                static_cast<unsigned>((height + kBlockY - 1) / kBlockY));
}

}  // namespace

void launch_bgr_to_grayscale(const uint8_t* d_bgr, uint8_t* d_gray,
                             int width, int height, cudaStream_t stream) {
    bgrToGrayscale<<<grid_for(width, height), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_bgr, d_gray, width, height);
    CUDA_CHECK_KERNEL("bgrToGrayscale");
}

void launch_gaussian_blur_naive(const uint8_t* d_src, uint8_t* d_dst,
                                const float* d_weights2d, int ksize,
                                int width, int height, cudaStream_t stream) {
    gaussianBlurNaive<<<grid_for(width, height), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_src, d_dst, d_weights2d, ksize, width, height);
    CUDA_CHECK_KERNEL("gaussianBlurNaive");
}

void launch_sobel_naive(const uint8_t* d_src, uint8_t* d_dst,
                        int width, int height, cudaStream_t stream) {
    sobelNaive<<<grid_for(width, height), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_src, d_dst, width, height);
    CUDA_CHECK_KERNEL("sobelNaive");
}

}  // namespace frameflow
