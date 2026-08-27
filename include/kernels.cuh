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
void launch_gaussian_blur_naive(const uint8_t* d_src, uint8_t* d_dst,
                                const float* d_weights2d, int ksize,
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
