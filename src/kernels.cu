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

// Filter weights in __constant__ memory. The tap index is uniform across a
// warp, so every thread reads the same value at the same time, which is the
// access pattern the constant broadcast cache is built for.
//
// Sized for the maximum supported kernel (31, enforced in GpuPipeline): 124
// bytes for the 1D filter and 3,844 for the 2D, against a 64 KB bank.
// __constant__ arrays must be statically sized at file scope, which is why
// these are not passed as kernel parameters like the image buffers.
namespace {
constexpr int kMaxKsize = 31;
}

__constant__ float c_weights1d[kMaxKsize];
__constant__ float c_weights2d[kMaxKsize * kMaxKsize];

void upload_filter_weights(const std::vector<float>& w1d, const std::vector<float>& w2d) {
    if (w1d.size() > static_cast<size_t>(kMaxKsize) ||
        w2d.size() > static_cast<size_t>(kMaxKsize) * kMaxKsize) {
        throw std::runtime_error("filter too large for the constant memory reservation");
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights1d, w1d.data(), w1d.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights2d, w2d.data(), w2d.size() * sizeof(float)));
}

namespace {

// BT.601 luminance weights, indexed to match BGR byte order.
constexpr float kWeightB = 0.114f;
constexpr float kWeightG = 0.587f;
constexpr float kWeightR = 0.299f;

// BORDER_REFLECT_101, matching the OpenCV convolution default so validation is
// not dominated by border disagreement. Mirrors about the edge pixel without
// repeating it:
//
//     row:            a b c d e
//     extended:   c b|a b c d e|d c
//
// so index -1 maps to 1 and index n maps to n-2. Plain REFLECT repeats the edge
// sample instead, shifting every border pixel by one. The loop covers taps
// reaching past the mirrored range, which happens only when the image is
// narrower than the filter; n == 1 would otherwise spin forever.
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

// One thread per output pixel, element-wise, no neighbour access.
//
// x comes from threadIdx.x so consecutive threads in a warp handle consecutive
// columns. Images are row-major, so those accesses are contiguous and coalesce.
// Mapping x to threadIdx.y would make each thread in a warp stride a full row
// apart, splitting one coalesced access into 32.
__global__ void bgrToGrayscale(const uint8_t* __restrict__ bgr,
                               uint8_t* __restrict__ gray,
                               int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid dimensions round up, so the last block in each dimension overhangs
    // the image and those threads must not read or write.
    if (x >= width || y >= height) return;

    const int src = (y * width + x) * 3;
    const float b = static_cast<float>(bgr[src + 0]);
    const float g = static_cast<float>(bgr[src + 1]);
    const float r = static_cast<float>(bgr[src + 2]);

    gray[y * width + x] = to_u8(kWeightB * b + kWeightG * g + kWeightR * r);
}

// Direct 2D convolution: every tap read from global memory.
//
// Traffic for the default 5x5: 25 loads per output pixel. A 16x16 block issues
// 256*25 = 6,400 loads while touching only a 20x20 = 400 pixel region, so the
// same bytes are requested roughly 16x more than necessary. Measured gain from
// tiling is far below 16x, which suggests caching already recovers much of it.
__global__ void gaussianBlurNaive(const uint8_t* __restrict__ src,
                                  uint8_t* __restrict__ dst,
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
            acc += c_weights2d[(ky + radius) * ksize + (kx + radius)] *
                   static_cast<float>(src[sy * width + sx]);
        }
    }

    dst[y * width + x] = to_u8(acc);
}

// Separable blur, horizontal pass: uint8 in, float out.
// Only x is reflected; y is a plain row index, so accesses stay coalesced.
__global__ void gaussianBlurHorizontal(const uint8_t* __restrict__ src,
                                       float* __restrict__ dst,
                                       int ksize, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int radius = ksize / 2;
    const int row = y * width;
    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        acc += c_weights1d[k + radius] * static_cast<float>(src[row + reflect101(x + k, width)]);
    }
    dst[row + x] = acc;  // kept in float; see the note in kernels.cuh
}

// Separable blur, vertical pass: float in, uint8 out.
//
// Taps stride a full row apart, so a warp's loads touch ksize different rows.
// Each individual load is still coalesced because consecutive threads read
// consecutive columns, which is why x stays the fastest-varying index here.
__global__ void gaussianBlurVertical(const float* __restrict__ src,
                                     uint8_t* __restrict__ dst,
                                     int ksize, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int radius = ksize / 2;
    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        acc += c_weights1d[k + radius] * src[reflect101(y + k, height) * width + x];
    }
    dst[y * width + x] = to_u8(acc);
}

// Separable blur, horizontal pass, shared-memory tiled.
// Shared tile: [blockDim.y][blockDim.x + 2*radius]. Only x needs a halo here.
//
// HALO LOADING. A grid-stride loop over the tile width has each thread load one
// interior pixel, with low-index threads looping again for the halo columns.
// At blockDim.x = 16 and radius 2, tileW = 20: threads 0-15 load columns 0-15,
// then threads 0-3 load 16-19. This needs no separate halo index arithmetic and
// cannot leave a gap when the tile exceeds twice the block width.
//
// The __syncthreads() is mandatory: thread 15 reads columns written by other
// threads. It sits OUTSIDE the bounds guard because every thread must reach
// every barrier, so letting out-of-range threads return early would deadlock.
__global__ void gaussianBlurHorizontalShared(const uint8_t* __restrict__ src,
                                             float* __restrict__ dst,
                                             int ksize, int width, int height) {
    extern __shared__ float tile[];

    const int radius = ksize / 2;
    const int tileW = blockDim.x + 2 * radius;
    const int x0 = blockIdx.x * blockDim.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height) {
        for (int i = threadIdx.x; i < tileW; i += blockDim.x) {
            const int sx = reflect101(x0 - radius + i, width);
            tile[threadIdx.y * tileW + i] = static_cast<float>(src[y * width + sx]);
        }
    }
    __syncthreads();

    const int x = x0 + threadIdx.x;
    if (x >= width || y >= height) return;

    float acc = 0.0f;
    for (int k = 0; k < ksize; ++k) {
        acc += c_weights1d[k] * tile[threadIdx.y * tileW + threadIdx.x + k];
    }
    dst[y * width + x] = acc;
}

// Separable blur, vertical pass, shared-memory tiled.
// Shared tile: [blockDim.y + 2*radius][blockDim.x]. Only y needs a halo.
//
// Untiled, each tap strides a full row away, so one output pixel's taps land in
// ksize different cache lines. Tiled, those rows are read once and reused by the
// whole column. x stays the fastest-varying index so global loads still coalesce
// even though the tile is tall rather than wide.
__global__ void gaussianBlurVerticalShared(const float* __restrict__ src,
                                           uint8_t* __restrict__ dst,
                                           int ksize, int width, int height) {
    extern __shared__ float tile[];

    const int radius = ksize / 2;
    const int tileH = blockDim.y + 2 * radius;
    const int y0 = blockIdx.y * blockDim.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width) {
        for (int j = threadIdx.y; j < tileH; j += blockDim.y) {
            const int sy = reflect101(y0 - radius + j, height);
            tile[j * blockDim.x + threadIdx.x] = src[sy * width + x];
        }
    }
    __syncthreads();

    const int y = y0 + threadIdx.y;
    if (x >= width || y >= height) return;

    float acc = 0.0f;
    for (int k = 0; k < ksize; ++k) {
        acc += c_weights1d[k] * tile[(threadIdx.y + k) * blockDim.x + threadIdx.x];
    }
    dst[y * width + x] = to_u8(acc);
}

// Sobel, shared-memory tiled. Radius 1, so an 18x18 tile for a 16x16 block.
// Both dimensions need a halo, so the load is a 2D grid-stride loop. Every
// thread participates regardless of whether its own output pixel is in bounds,
// because the barrier requires the whole block.
__global__ void sobelShared(const uint8_t* __restrict__ src,
                            uint8_t* __restrict__ dst,
                            int width, int height) {
    extern __shared__ float tile[];

    const int tileW = blockDim.x + 2;
    const int tileH = blockDim.y + 2;
    const int x0 = blockIdx.x * blockDim.x;
    const int y0 = blockIdx.y * blockDim.y;

    for (int j = threadIdx.y; j < tileH; j += blockDim.y) {
        const int sy = reflect101(y0 - 1 + j, height);
        for (int i = threadIdx.x; i < tileW; i += blockDim.x) {
            const int sx = reflect101(x0 - 1 + i, width);
            tile[j * tileW + i] = static_cast<float>(src[sy * width + sx]);
        }
    }
    __syncthreads();

    const int x = x0 + threadIdx.x;
    const int y = y0 + threadIdx.y;
    if (x >= width || y >= height) return;

    const float kx3[3][3] = {{-1.f, 0.f, 1.f}, {-2.f, 0.f, 2.f}, {-1.f, 0.f, 1.f}};
    const float ky3[3][3] = {{-1.f, -2.f, -1.f}, {0.f, 0.f, 0.f}, {1.f, 2.f, 1.f}};

    float gx = 0.0f;
    float gy = 0.0f;
    for (int dy = 0; dy < 3; ++dy) {
        for (int dx = 0; dx < 3; ++dx) {
            // +dy/+dx rather than -1..+1: the tile already starts one pixel
            // above and left of the block's output region.
            const float v = tile[(threadIdx.y + dy) * tileW + (threadIdx.x + dx)];
            gx += kx3[dy][dx] * v;
            gy += ky3[dy][dx] * v;
        }
    }

    dst[y * width + x] = to_u8(sqrtf(gx * gx + gy * gy));
}

// Sobel edge detection, naive variant. Sign convention matches cv::Sobel, so
// gx is positive for a left-to-right increase in intensity:
//
//     gx = [-1  0  1]        gy = [-1 -2 -1]
//          [-2  0  2]             [ 0  0  0]
//          [-1  0  1]             [ 1  2  1]
//
// Uses the true magnitude sqrt(gx^2 + gy^2) rather than the |gx| + |gy|
// approximation, which overestimates diagonal edges by up to 41% and would
// disagree with the OpenCV baseline by more than rounding.
//
// SATURATION: gx and gy span [-1020, 1020] for 8-bit input, so the magnitude
// reaches ~1443 against an 8-bit output. Values above 255 are clamped, not
// scaled, matching what the CPU baseline does converting to CV_8U. The result
// is an edge map, not a quantitative gradient field.
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

void launch_gaussian_blur_naive(const uint8_t* d_src, uint8_t* d_dst, int ksize,
                                int width, int height, cudaStream_t stream) {
    gaussianBlurNaive<<<grid_for(width, height), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_src, d_dst, ksize, width, height);
    CUDA_CHECK_KERNEL("gaussianBlurNaive");
}

void launch_gaussian_blur_separable(const uint8_t* d_src, float* d_tmp, uint8_t* d_dst,
                                    int ksize,
                                    int width, int height, cudaStream_t stream) {
    const dim3 grid = grid_for(width, height);
    const dim3 block(kBlockX, kBlockY);

    gaussianBlurHorizontal<<<grid, block, 0, stream>>>(d_src, d_tmp, ksize, width, height);
    CUDA_CHECK_KERNEL("gaussianBlurHorizontal");

    // The vertical pass reads what the horizontal pass wrote. Both are issued
    // into the same stream, and work in a stream executes in order, so no
    // explicit synchronization is needed between them.
    gaussianBlurVertical<<<grid, block, 0, stream>>>(d_tmp, d_dst, ksize, width, height);
    CUDA_CHECK_KERNEL("gaussianBlurVertical");
}

void launch_gaussian_blur_shared(const uint8_t* d_src, float* d_tmp, uint8_t* d_dst,
                                 int ksize,
                                 int width, int height, cudaStream_t stream) {
    const dim3 grid = grid_for(width, height);
    const dim3 block(kBlockX, kBlockY);
    const int radius = ksize / 2;

    // Dynamic shared memory, sized per pass. The horizontal tile is wide, the
    // vertical tile is tall; sizing each separately avoids reserving the
    // maximum of the two for both.
    const size_t h_bytes = static_cast<size_t>(kBlockY) *
                           static_cast<size_t>(kBlockX + 2 * radius) * sizeof(float);
    const size_t v_bytes = static_cast<size_t>(kBlockY + 2 * radius) *
                           static_cast<size_t>(kBlockX) * sizeof(float);

    gaussianBlurHorizontalShared<<<grid, block, h_bytes, stream>>>(
        d_src, d_tmp, ksize, width, height);
    CUDA_CHECK_KERNEL("gaussianBlurHorizontalShared");

    gaussianBlurVerticalShared<<<grid, block, v_bytes, stream>>>(
        d_tmp, d_dst, ksize, width, height);
    CUDA_CHECK_KERNEL("gaussianBlurVerticalShared");
}

void launch_sobel_shared(const uint8_t* d_src, uint8_t* d_dst,
                         int width, int height, cudaStream_t stream) {
    const size_t bytes = static_cast<size_t>(kBlockY + 2) *
                         static_cast<size_t>(kBlockX + 2) * sizeof(float);
    sobelShared<<<grid_for(width, height), dim3(kBlockX, kBlockY), bytes, stream>>>(
        d_src, d_dst, width, height);
    CUDA_CHECK_KERNEL("sobelShared");
}

void launch_sobel_naive(const uint8_t* d_src, uint8_t* d_dst,
                        int width, int height, cudaStream_t stream) {
    sobelNaive<<<grid_for(width, height), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_src, d_dst, width, height);
    CUDA_CHECK_KERNEL("sobelNaive");
}

}  // namespace frameflow
