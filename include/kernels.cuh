#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace frameflow {

// ---------------------------------------------------------------------------
// Filter generation (host side)
//
// Weights are computed at startup from the Gaussian function rather than being
// hardcoded, so --ksize and --sigma actually mean something. The formula
// deliberately mirrors cv::getGaussianKernel: same exponent, same
// normalize-by-sum. If the two disagreed, every blur comparison against OpenCV
// would show an error that has nothing to do with the kernel being tested.
// ---------------------------------------------------------------------------
std::vector<float> gaussian_kernel_1d(int ksize, double sigma);

// Outer product of the 1D kernel with itself -- the full ksize*ksize filter.
// Used by the naive variant, which performs a genuine 2D convolution. That a
// 2D Gaussian factors into this outer product at all is exactly what the
// separable variant exploits later.
std::vector<float> gaussian_kernel_2d(int ksize, double sigma);

// Copies the filter weights into __constant__ memory. Must be called before any
// blur launch, and again whenever ksize or sigma changes.
//
// IMPORTANT: __constant__ memory is a single per-module resource, not per
// object. Two GpuPipeline instances with different filters would overwrite each
// other's weights, and the second construction would silently corrupt the
// first's results. GpuPipeline enforces that only one instance is live at a
// time rather than leaving that to convention.
void upload_filter_weights(const std::vector<float>& w1d, const std::vector<float>& w2d);

// ---------------------------------------------------------------------------
// Stage 1: BGR -> grayscale
//
// INPUT CHANNEL ORDER IS BGR, NOT RGB. cv::VideoCapture delivers frames with
// byte 0 = blue, byte 1 = green, byte 2 = red. The kernel is named for what it
// actually consumes so the assumption cannot drift; feeding it RGB would swap
// the blue and red weights and produce a plausible-looking but wrong image.
//
// Luminance is ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B, with each
// coefficient applied to the correct byte.
// ---------------------------------------------------------------------------
void launch_bgr_to_grayscale(const uint8_t* d_bgr, uint8_t* d_gray,
                             int width, int height, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 2: Gaussian blur -- naive variant
//
// Direct 2D convolution. Every thread reads all ksize*ksize neighbours from
// global memory: 25 reads per output pixel at the default 5x5. Adjacent threads
// re-read almost the same neighbourhood, so the same bytes cross the memory bus
// many times. This is the baseline the later variants are measured against.
// ---------------------------------------------------------------------------
void launch_gaussian_blur_naive(const uint8_t* d_src, uint8_t* d_dst, int ksize,
                                int width, int height, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 2: Gaussian blur -- separable variant
//
// A 2D Gaussian is the outer product of two 1D Gaussians, so convolving with
// the KxK filter is equivalent to a horizontal 1D pass followed by a vertical
// one. That reduces the taps per output pixel from K*K to 2K: at K=5, from 25
// to 10, a 2.5x reduction in loads. At K=9 it would be 81 vs 18, a 4.5x
// reduction -- the advantage grows with kernel size.
//
// This is an ALGORITHMIC win, independent of memory layout or hardware, and it
// composes with the tiling optimisation rather than competing with it.
//
// It works because the Gaussian is separable. Most filters are not: a general
// KxK filter matrix has rank > 1 and cannot be factored into two 1D passes.
// Sobel happens to be separable too ([1,2,1] x [-1,0,1]), but at 3x3 the saving
// is 9 taps vs 6 and not worth the extra pass and intermediate buffer.
//
// The intermediate is float, not uint8. Rounding the horizontal pass to 8 bits
// before the vertical pass would quantize twice and make this variant disagree
// with the naive one by more than rounding. That costs 4 bytes per intermediate
// pixel instead of 1, which partly offsets the traffic saved -- an honest
// trade-off, made in favour of the variants being numerically comparable.
// ---------------------------------------------------------------------------
void launch_gaussian_blur_separable(const uint8_t* d_src, float* d_tmp, uint8_t* d_dst,
                                    int ksize, int width, int height,
                                    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 2: Gaussian blur -- shared-memory tiled variant (separable + tiling)
//
// Each block cooperatively loads the pixels its threads need into __shared__
// memory once, calls __syncthreads(), then every thread computes reading only
// from on-chip memory. The extra border of pixels that edge threads need but
// that lies outside the block's own output region is the HALO.
//
// Geometry, 16x16 block with a 5x5 filter (radius 2):
//   naive 2D:   256 outputs, 25 global loads each        = 6,400 loads
//                distinct data actually touched: 20x20   =   400 pixels
//   tiled 2D:   loads that 20x20 region once             =   400 loads, 16x fewer
//
// Because this variant is also separable, the tiles are 1D-shaped rather than
// square, which is cheaper still:
//   horizontal pass: (16 + 2*2) x 16 = 20x16 = 320 floats = 1,280 bytes
//   vertical pass:   16 x (16 + 2*2) = 16x20 = 320 floats = 1,280 bytes
//
// At 48 KB of shared memory per block on this GPU, those tiles are nowhere near
// the limit -- occupancy is bounded by other resources, not by shared memory,
// so there is no reason to shrink the tile.
//
// Shared memory is allocated dynamically because the filter radius is a runtime
// value (--ksize). Sizing a static array for the maximum supported radius would
// reserve shared memory that small filters never use.
// ---------------------------------------------------------------------------
void launch_gaussian_blur_shared(const uint8_t* d_src, float* d_tmp, uint8_t* d_dst,
                                 int ksize, int width, int height,
                                 cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 3: Sobel -- shared-memory tiled variant
//
// 16x16 block, radius 1, so an 18x18 tile: 324 loads to produce 256 outputs,
// versus 256*9 = 2,304 in the naive kernel. The saving ratio is smaller than
// the blur's because a 3x3 stencil has far less overlap between neighbouring
// threads to begin with -- which is a useful prediction to check against the
// measurement.
// ---------------------------------------------------------------------------
void launch_sobel_shared(const uint8_t* d_src, uint8_t* d_dst,
                         int width, int height, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 3: Sobel edge detection -- naive variant
//
// 3x3 horizontal and vertical gradients, combined as the true magnitude
// sqrt(gx^2 + gy^2) rather than the |gx| + |gy| approximation, so the result
// matches an OpenCV baseline computed the same way.
// ---------------------------------------------------------------------------
void launch_sobel_naive(const uint8_t* d_src, uint8_t* d_dst,
                        int width, int height, cudaStream_t stream);

}  // namespace frameflow
