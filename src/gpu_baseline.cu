#include "gpu_baseline.h"

#include "cuda_check.h"
#include "kernels.cuh"

#include <npp.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace frameflow {

namespace {

inline cudaEvent_t ev(void* p) { return static_cast<cudaEvent_t>(p); }

enum EventSlot { kStart = 0, kAfterUpload, kAfterGray, kAfterBlur, kAfterSobel, kEnd };

void npp_check(NppStatus s, const char* what) {
    if (s != NPP_SUCCESS) {
        throw std::runtime_error(std::string("NPP call failed: ") + what +
                                 " (status " + std::to_string(static_cast<int>(s)) + ")");
    }
}

// NPP produces the two gradients but has no fused magnitude, so this combines
// them. Deliberately identical arithmetic to sobelNaive in kernels.cu --
// sqrt(gx^2 + gy^2), round half to even, saturate at 255 -- so any difference
// between the two pipelines comes from NPP's filters, not from this step.
__global__ void magnitude16s(const int16_t* __restrict__ gx, int gx_pitch,
                             const int16_t* __restrict__ gy, int gy_pitch,
                             uint8_t* __restrict__ dst, int dst_pitch,
                             int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int16_t* rx = reinterpret_cast<const int16_t*>(
        reinterpret_cast<const char*>(gx) + static_cast<size_t>(y) * gx_pitch);
    const int16_t* ry = reinterpret_cast<const int16_t*>(
        reinterpret_cast<const char*>(gy) + static_cast<size_t>(y) * gy_pitch);
    uint8_t* row = reinterpret_cast<uint8_t*>(
        reinterpret_cast<char*>(dst) + static_cast<size_t>(y) * dst_pitch);

    const float fx = static_cast<float>(rx[x]);
    const float fy = static_cast<float>(ry[x]);
    row[x] = static_cast<uint8_t>(fminf(fmaxf(rintf(sqrtf(fx * fx + fy * fy)), 0.0f), 255.0f));
}

}  // namespace

void NppBaseline::free_all() noexcept {
    if (d_bgr_) { nppiFree(d_bgr_); d_bgr_ = nullptr; }
    if (d_gray_) { nppiFree(d_gray_); d_gray_ = nullptr; }
    if (d_blur_) { nppiFree(d_blur_); d_blur_ = nullptr; }
    if (d_edges_) { nppiFree(d_edges_); d_edges_ = nullptr; }
    if (d_gx_) { nppiFree(d_gx_); d_gx_ = nullptr; }
    if (d_gy_) { nppiFree(d_gy_); d_gy_ = nullptr; }
    if (d_kernel_) { cudaFree(d_kernel_); d_kernel_ = nullptr; }
    for (auto& e : ev_) {
        if (e) { cudaEventDestroy(ev(e)); e = nullptr; }
    }
}

NppBaseline::NppBaseline(int width, int height, int ksize, double sigma)
    : width_(width), height_(height), ksize_(ksize) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("NppBaseline requires a positive frame size");
    }
    if (ksize <= 0 || ksize % 2 == 0) {
        throw std::runtime_error("NppBaseline requires a positive odd kernel size");
    }

    try {
        d_bgr_ = nppiMalloc_8u_C3(width, height, &bgr_step_);
        d_gray_ = nppiMalloc_8u_C1(width, height, &gray_step_);
        d_blur_ = nppiMalloc_8u_C1(width, height, &blur_step_);
        d_edges_ = nppiMalloc_8u_C1(width, height, &edges_step_);
        d_gx_ = nppiMalloc_16s_C1(width, height, &gx_step_);
        d_gy_ = nppiMalloc_16s_C1(width, height, &gy_step_);
        if (!d_bgr_ || !d_gray_ || !d_blur_ || !d_edges_ || !d_gx_ || !d_gy_) {
            throw std::runtime_error("nppiMalloc failed for a " + std::to_string(width) +
                                     "x" + std::to_string(height) + " frame");
        }

        // The custom pipeline's own weights, so both convolve with identical
        // numbers. NPP's built-in nppiFilterGauss uses a fixed mask that does
        // not correspond to this sigma, which would not be a fair comparison.
        const std::vector<float> w2 = gaussian_kernel_2d(ksize, sigma);
        CUDA_CHECK(cudaMalloc(&d_kernel_, w2.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernel_, w2.data(), w2.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        for (auto& slot : ev_) {
            cudaEvent_t e = nullptr;
            CUDA_CHECK(cudaEventCreate(&e));
            slot = e;
        }
    } catch (...) {
        free_all();
        throw;
    }
}

NppBaseline::~NppBaseline() { free_all(); }

void NppBaseline::process(const cv::Mat& bgr, cv::Mat& out, FrameTimings& timings) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("NppBaseline::process expects a non-empty CV_8UC3 frame");
    }
    if (bgr.cols != width_ || bgr.rows != height_) {
        throw std::runtime_error("frame size changed mid-stream; NPP buffers are sized once");
    }

    out.create(height_, width_, CV_8UC1);

    const NppiSize roi{width_, height_};
    const NppiSize srcsz{width_, height_};
    const NppiPoint off{0, 0};

    CUDA_CHECK(cudaEventRecord(ev(ev_[kStart])));

    CUDA_CHECK(cudaMemcpy2D(d_bgr_, static_cast<size_t>(bgr_step_), bgr.data, bgr.step,
                            static_cast<size_t>(width_) * 3, static_cast<size_t>(height_),
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ev(ev_[kAfterUpload])));

    // BT.601 weights in BGR order -- coefficient i applies to channel i, and
    // channel 0 of an OpenCV frame is blue. Passing RGB-ordered coefficients
    // here would swap the blue and red contributions.
    const Npp32f coeffs[3] = {0.114f, 0.587f, 0.299f};
    npp_check(nppiColorToGray_8u_C3C1R(d_bgr_, bgr_step_, d_gray_, gray_step_, roi, coeffs),
              "nppiColorToGray_8u_C3C1R");
    CUDA_CHECK(cudaEventRecord(ev(ev_[kAfterGray])));

    // Single 2D convolution. NPP's generic filter accepts only
    // NPP_BORDER_REPLICATE; MIRROR and CONSTANT return
    // NPP_NOT_SUPPORTED_MODE_ERROR, verified by probe.
    const NppiSize ksz{ksize_, ksize_};
    const NppiPoint anchor{ksize_ / 2, ksize_ / 2};
    npp_check(nppiFilterBorder32f_8u_C1R(d_gray_, gray_step_, srcsz, off,
                                         d_blur_, blur_step_, roi,
                                         d_kernel_, ksz, anchor, NPP_BORDER_REPLICATE),
              "nppiFilterBorder32f_8u_C1R");
    CUDA_CHECK(cudaEventRecord(ev(ev_[kAfterBlur])));

    // NPP names these by the edge orientation they detect, not by the
    // derivative axis: Vert is d/dx and Horiz is d/dy. Verified on an x-ramp,
    // where Vert returns 80 and Horiz returns 0 -- the same scale the custom
    // 3x3 Sobel produces, so no rescaling is needed.
    npp_check(nppiFilterSobelVertBorder_8u16s_C1R(d_blur_, blur_step_, srcsz, off,
                                                  d_gx_, gx_step_, roi,
                                                  NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE),
              "nppiFilterSobelVertBorder_8u16s_C1R");
    npp_check(nppiFilterSobelHorizBorder_8u16s_C1R(d_blur_, blur_step_, srcsz, off,
                                                   d_gy_, gy_step_, roi,
                                                   NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE),
              "nppiFilterSobelHorizBorder_8u16s_C1R");

    const dim3 block(16, 16);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    magnitude16s<<<grid, block>>>(d_gx_, gx_step_, d_gy_, gy_step_,
                                  d_edges_, edges_step_, width_, height_);
    CUDA_CHECK_KERNEL("magnitude16s");
    CUDA_CHECK(cudaEventRecord(ev(ev_[kAfterSobel])));

    CUDA_CHECK(cudaMemcpy2D(out.data, out.step, d_edges_, static_cast<size_t>(edges_step_),
                            static_cast<size_t>(width_), static_cast<size_t>(height_),
                            cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev(ev_[kEnd])));
    CUDA_CHECK(cudaEventSynchronize(ev(ev_[kEnd])));

    auto span = [&](EventSlot a, EventSlot b) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev(ev_[a]), ev(ev_[b])));
        return ms;
    };

    timings.upload_ms = span(kStart, kAfterUpload);
    timings.gray_ms = span(kAfterUpload, kAfterGray);
    timings.blur_ms = span(kAfterGray, kAfterBlur);
    timings.sobel_ms = span(kAfterBlur, kAfterSobel);
    timings.kernel_ms = span(kAfterUpload, kAfterSobel);
    timings.download_ms = span(kAfterSobel, kEnd);
    timings.gpu_total_ms = span(kStart, kEnd);
}

void NppBaseline::download_grayscale(cv::Mat& out) const {
    out.create(height_, width_, CV_8UC1);
    CUDA_CHECK(cudaMemcpy2D(out.data, out.step, d_gray_, static_cast<size_t>(gray_step_),
                            static_cast<size_t>(width_), static_cast<size_t>(height_),
                            cudaMemcpyDeviceToHost));
}

void NppBaseline::download_blurred(cv::Mat& out) const {
    out.create(height_, width_, CV_8UC1);
    CUDA_CHECK(cudaMemcpy2D(out.data, out.step, d_blur_, static_cast<size_t>(blur_step_),
                            static_cast<size_t>(width_), static_cast<size_t>(height_),
                            cudaMemcpyDeviceToHost));
}

}  // namespace frameflow
