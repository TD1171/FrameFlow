#include "pipeline.h"

#include "cuda_check.h"
#include "kernels.cuh"

#include <stdexcept>

namespace frameflow {

namespace {

inline cudaEvent_t as_event(void* p) { return static_cast<cudaEvent_t>(p); }

}  // namespace

void set_debug_sync(bool enabled) { g_debug_sync = enabled; }
bool debug_sync() { return g_debug_sync; }

GpuPipeline::GpuPipeline(int width, int height) : width_(width), height_(height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("GpuPipeline requires a positive frame size");
    }

    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    CUDA_CHECK(cudaMalloc(&d_bgr_, pixels * 3));
    CUDA_CHECK(cudaMalloc(&d_gray_, pixels));

    cudaEvent_t a, b, c, d;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventCreate(&c));
    CUDA_CHECK(cudaEventCreate(&d));
    ev_start_ = a;
    ev_after_upload_ = b;
    ev_after_kernels_ = c;
    ev_end_ = d;
}

GpuPipeline::~GpuPipeline() {
    // Destructors must not throw, so failures here are ignored rather than
    // checked. There is nothing useful to do about a failed free during teardown.
    if (d_bgr_) cudaFree(d_bgr_);
    if (d_gray_) cudaFree(d_gray_);
    if (ev_start_) cudaEventDestroy(as_event(ev_start_));
    if (ev_after_upload_) cudaEventDestroy(as_event(ev_after_upload_));
    if (ev_after_kernels_) cudaEventDestroy(as_event(ev_after_kernels_));
    if (ev_end_) cudaEventDestroy(as_event(ev_end_));
}

void GpuPipeline::process(const cv::Mat& bgr, cv::Mat& out, FrameTimings& timings) {
    if (bgr.empty()) throw std::runtime_error("GpuPipeline::process got an empty frame");
    if (bgr.type() != CV_8UC3) {
        throw std::runtime_error("GpuPipeline::process expects CV_8UC3 (8-bit BGR)");
    }
    if (bgr.cols != width_ || bgr.rows != height_) {
        throw std::runtime_error("frame size changed mid-stream; device buffers are "
                                 "sized once at construction");
    }

    out.create(height_, width_, CV_8UC1);

    CUDA_CHECK(cudaEventRecord(as_event(ev_start_)));

    // cudaMemcpy2D, not cudaMemcpy: a cv::Mat may pad each row, so bgr.step can
    // exceed width*3. Copying bgr.total()*3 bytes linearly would shear the image
    // whenever padding is present. Device buffers are tightly packed.
    CUDA_CHECK(cudaMemcpy2D(d_bgr_, static_cast<size_t>(width_) * 3,
                            bgr.data, bgr.step,
                            static_cast<size_t>(width_) * 3, static_cast<size_t>(height_),
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(as_event(ev_after_upload_)));

    launch_bgr_to_grayscale(d_bgr_, d_gray_, width_, height_, /*stream=*/0);

    CUDA_CHECK(cudaEventRecord(as_event(ev_after_kernels_)));

    CUDA_CHECK(cudaMemcpy2D(out.data, out.step,
                            d_gray_, static_cast<size_t>(width_),
                            static_cast<size_t>(width_), static_cast<size_t>(height_),
                            cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(as_event(ev_end_)));

    // Exactly one synchronization for the whole measured region. Elapsed times
    // between events are only valid once the last event has actually occurred.
    CUDA_CHECK(cudaEventSynchronize(as_event(ev_end_)));

    CUDA_CHECK(cudaEventElapsedTime(&timings.upload_ms,
                                    as_event(ev_start_), as_event(ev_after_upload_)));
    CUDA_CHECK(cudaEventElapsedTime(&timings.kernel_ms,
                                    as_event(ev_after_upload_), as_event(ev_after_kernels_)));
    CUDA_CHECK(cudaEventElapsedTime(&timings.download_ms,
                                    as_event(ev_after_kernels_), as_event(ev_end_)));
    CUDA_CHECK(cudaEventElapsedTime(&timings.gpu_total_ms,
                                    as_event(ev_start_), as_event(ev_end_)));
}

}  // namespace frameflow
