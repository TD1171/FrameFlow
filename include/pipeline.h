#pragma once

#include <opencv2/core.hpp>

#include <cstdint>

namespace frameflow {

// Accessors for the per-launch synchronization mode. Host translation units use
// these rather than touching the flag directly, so that main.cpp does not have
// to include <cuda_runtime.h> -- it is compiled by the host compiler, which has
// no CUDA include path. See cuda_check.h for what the mode actually changes.
void set_debug_sync(bool enabled);
bool debug_sync();

// Per-frame GPU timings, in milliseconds, measured with cudaEvent records.
// Host clocks are not used for GPU work: kernel launches are asynchronous, so a
// host timer around a launch measures the launch call, not the kernel.
struct FrameTimings {
    float upload_ms = 0.0f;   // host -> device
    float kernel_ms = 0.0f;   // all kernels, transfers excluded
    float download_ms = 0.0f; // device -> host
    float gpu_total_ms = 0.0f;// upload + kernels + download

    void accumulate(const FrameTimings& t) {
        upload_ms += t.upload_ms;
        kernel_ms += t.kernel_ms;
        download_ms += t.download_ms;
        gpu_total_ms += t.gpu_total_ms;
    }
    void divide(double n) {
        upload_ms = static_cast<float>(upload_ms / n);
        kernel_ms = static_cast<float>(kernel_ms / n);
        download_ms = static_cast<float>(download_ms / n);
        gpu_total_ms = static_cast<float>(gpu_total_ms / n);
    }
};

// Owns every device allocation for the pipeline and reuses them across frames.
// Allocating per frame would put a cudaMalloc/cudaFree pair -- both implicitly
// synchronizing -- inside the measured loop and dominate the timings.
class GpuPipeline {
public:
    GpuPipeline(int width, int height);
    ~GpuPipeline();

    GpuPipeline(const GpuPipeline&) = delete;
    GpuPipeline& operator=(const GpuPipeline&) = delete;

    // Uploads one BGR frame, runs the pipeline on-device, and downloads the
    // result. `out` is resized to CV_8UC1. No intermediate result returns to
    // the host: every stage reads and writes device memory.
    void process(const cv::Mat& bgr, cv::Mat& out, FrameTimings& timings);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_ = 0;
    int height_ = 0;

    uint8_t* d_bgr_ = nullptr;   // width*height*3
    uint8_t* d_gray_ = nullptr;  // width*height

    // Reused across frames; creating events per frame is a measurable cost.
    void* ev_start_ = nullptr;
    void* ev_after_upload_ = nullptr;
    void* ev_after_kernels_ = nullptr;
    void* ev_end_ = nullptr;
};

}  // namespace frameflow
