#pragma once

#include "pipeline.h"

#include <opencv2/core.hpp>

#include <cstdint>

namespace frameflow {

// GPU reference implementation built on NVIDIA Performance Primitives (NPP),
// which ships with the CUDA Toolkit.
//
// This exists because comparing custom CUDA kernels against an OpenCV CPU
// baseline only shows that a GPU beats a CPU, which is not in question. The
// interesting comparison is against a vendor-optimised GPU library.
//
// OpenCV's own CUDA module is not used because the official OpenCV Windows
// binaries ship without it; building it from source would take hours and its
// filters call NPP for several of these operations anyway.
//
// Timings reuse FrameTimings so they are directly comparable to GpuPipeline:
// the same cudaEvent placement, the same span definitions, the same single
// synchronize at the end of the measured region.
//
// SEMANTIC DIFFERENCES FROM THE CUSTOM PIPELINE -- all measured, not assumed:
//
//   Border mode. NPP's convolution and Sobel accept only NPP_BORDER_REPLICATE;
//   MIRROR, CONSTANT and WRAP return NPP_NOT_SUPPORTED_MODE_ERROR. The custom
//   kernels use BORDER_REFLECT_101 to match OpenCV. Outputs therefore differ
//   within one filter radius of the image edge and are compared on the interior
//   only, with the excluded margin reported.
//
//   Blur algorithm. NPP's generic convolution is a single 2D pass, so it is
//   algorithmically equivalent to the custom 'naive' variant rather than to
//   'separable' or 'shared'. Both comparisons are reported.
//
//   Sobel. NPP has no fused gradient-magnitude call, so it produces two 16-bit
//   gradient images and the magnitude is combined by a small kernel in this
//   file. That kernel's time is counted in NPP's total; excluding it would
//   flatter NPP.
//
//   Kernel orientation. NPP convolves (flipping the kernel) where the custom
//   kernels correlate. The Gaussian is symmetric, so this makes no difference
//   here, but it would for an asymmetric filter.
class NppBaseline {
public:
    // Uses the same Gaussian weights as the custom pipeline: gaussian_kernel_2d
    // from kernels.cuh, so the two are convolving with identical numbers rather
    // than with NPP's built-in Gaussian, whose coefficients differ.
    NppBaseline(int width, int height, int ksize, double sigma);
    ~NppBaseline();

    NppBaseline(const NppBaseline&) = delete;
    NppBaseline& operator=(const NppBaseline&) = delete;

    // Uploads one BGR frame, runs grayscale -> blur -> Sobel through NPP, and
    // downloads the edge map. Device buffers are allocated once in the
    // constructor and reused, so nothing is allocated inside the timed region.
    void process(const cv::Mat& bgr, cv::Mat& out, FrameTimings& timings);

    // Intermediates, for per-stage validation. Never called inside a timed loop.
    void download_grayscale(cv::Mat& out) const;
    void download_blurred(cv::Mat& out) const;

    // Pixels within this distance of the edge may legitimately differ from the
    // custom pipeline because of the border-mode difference above. Comparisons
    // should exclude them and report the count.
    int border_margin() const { return ksize_ / 2 + 2; }

private:
    void free_all() noexcept;

    int width_ = 0;
    int height_ = 0;
    int ksize_ = 5;

    uint8_t* d_bgr_ = nullptr;    // 8u C3
    uint8_t* d_gray_ = nullptr;   // 8u C1
    uint8_t* d_blur_ = nullptr;   // 8u C1
    uint8_t* d_edges_ = nullptr;  // 8u C1
    int16_t* d_gx_ = nullptr;     // 16s C1, signed gradient
    int16_t* d_gy_ = nullptr;     // 16s C1
    float* d_kernel_ = nullptr;   // ksize*ksize convolution weights

    // NPP allocations are pitched for row alignment, so every buffer carries
    // its own step rather than assuming width*channels.
    int bgr_step_ = 0;
    int gray_step_ = 0;
    int blur_step_ = 0;
    int edges_step_ = 0;
    int gx_step_ = 0;
    int gy_step_ = 0;

    void* ev_[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

}  // namespace frameflow
