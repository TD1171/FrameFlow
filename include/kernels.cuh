#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace frameflow {

// Gaussian weights, generated from the function rather than hardcoded so that
// --ksize and --sigma are meaningful. Matches cv::getGaussianKernel's formula
// and normalisation so the GPU and CPU paths agree.
std::vector<float> gaussian_kernel_1d(int ksize, double sigma);

// Outer product of the 1D kernel with itself, used by the naive variant. That
// this factorisation exists is what the separable variant exploits.
std::vector<float> gaussian_kernel_2d(int ksize, double sigma);

// Copies filter weights into __constant__ memory. Must be called before any
// blur launch, and again if ksize or sigma changes.
//
// __constant__ memory belongs to the module, not to an object, so two live
// pipelines with different filters would overwrite each other. GpuPipeline
// enforces a single live instance.
void upload_filter_weights(const std::vector<float>& w1d, const std::vector<float>& w2d);

// INPUT IS BGR, NOT RGB -- cv::VideoCapture delivers byte 0 = blue. The kernel
// is named for what it consumes so the assumption cannot drift. Applies BT.601
// luminance (Y = 0.299R + 0.587G + 0.114B) to the correct bytes.
void launch_bgr_to_grayscale(const uint8_t* d_bgr, uint8_t* d_gray,
                             int width, int height, cudaStream_t stream);

// Direct 2D convolution: ksize*ksize taps per output pixel, all from global
// memory. The baseline the other variants are measured against.
void launch_gaussian_blur_naive(const uint8_t* d_src, uint8_t* d_dst, int ksize,
                                int width, int height, cudaStream_t stream);

// Two 1D passes instead of one 2D pass, reducing taps per output pixel from
// K*K to 2K -- 25 to 10 at the default 5x5, and the advantage grows with K.
// An algorithmic win, independent of memory layout, and it composes with
// tiling. Works because the Gaussian is separable; most filters are not.
//
// The intermediate is float rather than uint8 so the two passes do not quantize
// twice, which would make this variant disagree with the naive one by more than
// rounding. That costs 4 bytes per intermediate pixel instead of 1.
void launch_gaussian_blur_separable(const uint8_t* d_src, float* d_tmp, uint8_t* d_dst,
                                    int ksize, int width, int height,
                                    cudaStream_t stream);

// Separable, plus each block stages its tile and halo in __shared__ memory and
// computes from on-chip data.
//
// Geometry for a 16x16 block with a 5x5 filter: the horizontal pass tiles
// 20x16 and the vertical 16x20, 1,280 bytes each against 48 KB of shared memory
// per block -- so tile size is bounded by occupancy, not capacity. Shared
// memory is allocated dynamically because the radius is a runtime value.
void launch_gaussian_blur_shared(const uint8_t* d_src, float* d_tmp, uint8_t* d_dst,
                                 int ksize, int width, int height,
                                 cudaStream_t stream);

// 3x3 gradients combined as the true magnitude sqrt(gx^2 + gy^2). Output
// saturates at 255; see the kernel for what that costs.
void launch_sobel_naive(const uint8_t* d_src, uint8_t* d_dst,
                        int width, int height, cudaStream_t stream);

// Tiled Sobel: an 18x18 tile produces 256 outputs against 2,304 global loads in
// the naive kernel. Measured slower than naive on this hardware -- a 3x3 stencil
// has little reuse to recover. See docs/OPTIMIZATION.md.
void launch_sobel_shared(const uint8_t* d_src, uint8_t* d_dst,
                         int width, int height, cudaStream_t stream);

}  // namespace frameflow
