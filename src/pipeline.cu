#include "pipeline.h"

#include "cuda_check.h"
#include "kernels.cuh"

#include <stdexcept>
#include <vector>

namespace frameflow {

namespace {

inline cudaEvent_t ev(void* p) { return static_cast<cudaEvent_t>(p); }

enum EventSlot { kStart = 0, kAfterUpload, kAfterGray, kAfterBlur, kAfterSobel, kEnd };

}  // namespace

void set_debug_sync(bool enabled) { g_debug_sync = enabled; }
bool debug_sync() { return g_debug_sync; }

const char* variant_name(Variant v) {
    switch (v) {
        case Variant::Naive: return "naive";
        case Variant::Separable: return "separable";
        case Variant::Shared: return "shared";
    }
    return "unknown";
}

const char* variant_description(Variant v) {
    switch (v) {
        case Variant::Naive:
            return "direct 2D convolution, all taps from global memory";
        case Variant::Separable:
            return "two 1D passes, 2K taps instead of K*K";
        case Variant::Shared:
            return "separable + shared-memory tiling with halo regions";
    }
    return "unknown";
}

bool parse_variant(const std::string& s, Variant& out) {
    if (s == "naive") { out = Variant::Naive; return true; }
    if (s == "separable") { out = Variant::Separable; return true; }
    if (s == "shared") { out = Variant::Shared; return true; }
    return false;
}

void FrameTimings::accumulate(const FrameTimings& t) {
    upload_ms += t.upload_ms;
    gray_ms += t.gray_ms;
    blur_ms += t.blur_ms;
    sobel_ms += t.sobel_ms;
    kernel_ms += t.kernel_ms;
    download_ms += t.download_ms;
    gpu_total_ms += t.gpu_total_ms;
}

void FrameTimings::divide(double n) {
    const float d = static_cast<float>(n);
    upload_ms /= d;
    gray_ms /= d;
    blur_ms /= d;
    sobel_ms /= d;
    kernel_ms /= d;
    download_ms /= d;
    gpu_total_ms /= d;
}

void GpuPipeline::free_all() noexcept {
    // Used both by the destructor and by the constructor's failure path, so it
    // must tolerate being called with a partially built object.
    if (d_bgr_) { cudaFree(d_bgr_); d_bgr_ = nullptr; }
    if (d_gray_) { cudaFree(d_gray_); d_gray_ = nullptr; }
    if (d_blur_) { cudaFree(d_blur_); d_blur_ = nullptr; }
    if (d_edges_) { cudaFree(d_edges_); d_edges_ = nullptr; }
    if (d_tmp_) { cudaFree(d_tmp_); d_tmp_ = nullptr; }
    for (auto& e : ev_) {
        if (e) { cudaEventDestroy(ev(e)); e = nullptr; }
    }
}

// Filter weights live in __constant__ memory, which belongs to the module
// rather than to any object. A second live pipeline with a different ksize or
// sigma would overwrite the first one's weights, and the first would then
// silently produce results for the wrong filter. Constructing two is refused
// rather than left as an undocumented rule.
int GpuPipeline::live_instances_ = 0;

GpuPipeline::GpuPipeline(int width, int height, int ksize, double sigma, Variant variant)
    : width_(width), height_(height), ksize_(ksize), sigma_(sigma), variant_(variant) {
    if (live_instances_ > 0) {
        throw std::runtime_error("only one GpuPipeline may exist at a time: filter "
                                 "weights are held in __constant__ memory, which is "
                                 "shared by the whole module");
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("GpuPipeline requires a positive frame size");
    }
    if (ksize <= 0 || ksize % 2 == 0) {
        throw std::runtime_error("--ksize must be positive and odd, got " +
                                 std::to_string(ksize));
    }
    if (ksize > 31) {
        throw std::runtime_error("--ksize above 31 is not supported");
    }

    // Everything after this point can throw. If it does, the destructor will
    // NOT run -- the object never finished constructing -- so any allocation
    // already made would leak with no handle left to free it. Catching here and
    // releasing before rethrowing is what makes partial construction safe.
    try {
        const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        CUDA_CHECK(cudaMalloc(&d_bgr_, pixels * 3));
        CUDA_CHECK(cudaMalloc(&d_gray_, pixels));
        CUDA_CHECK(cudaMalloc(&d_blur_, pixels));
        CUDA_CHECK(cudaMalloc(&d_edges_, pixels));

        // The separable intermediate is float rather than uint8 so the two
        // 1D passes do not quantize twice. See kernels.cuh for the trade-off.
        CUDA_CHECK(cudaMalloc(&d_tmp_, pixels * sizeof(float)));

        // Weights go to __constant__ memory, which is a single per-module
        // resource -- see the live-instance guard above.
        upload_filter_weights(gaussian_kernel_1d(ksize, sigma),
                              gaussian_kernel_2d(ksize, sigma));

        for (auto& slot : ev_) {
            cudaEvent_t e = nullptr;
            CUDA_CHECK(cudaEventCreate(&e));
            slot = e;  // stored immediately, so a later failure can still free it
        }
    } catch (...) {
        free_all();
        throw;
    }

    // Incremented last: a pipeline that failed to construct never ran its
    // destructor, so counting it earlier would leak the slot permanently.
    ++live_instances_;
}

GpuPipeline::~GpuPipeline() {
    free_all();
    if (live_instances_ > 0) --live_instances_;
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

    CUDA_CHECK(cudaEventRecord(ev(ev_[kStart])));

    // cudaMemcpy2D, not cudaMemcpy: a cv::Mat may pad each row, so bgr.step can
    // exceed width*3. Copying linearly would shear the image whenever padding
    // is present. Device buffers are tightly packed.
    CUDA_CHECK(cudaMemcpy2D(d_bgr_, static_cast<size_t>(width_) * 3,
                            bgr.data, bgr.step,
                            static_cast<size_t>(width_) * 3, static_cast<size_t>(height_),
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ev(ev_[kAfterUpload])));

    // Three kernels, no host round-trip between them: grayscale writes d_gray_,
    // blur reads it and writes d_blur_, Sobel reads that and writes d_edges_.
    launch_bgr_to_grayscale(d_bgr_, d_gray_, width_, height_, /*stream=*/0);
    CUDA_CHECK(cudaEventRecord(ev(ev_[kAfterGray])));

    switch (variant_) {
        case Variant::Naive:
            launch_gaussian_blur_naive(d_gray_, d_blur_, ksize_, width_, height_,
                                       /*stream=*/0);
            break;
        case Variant::Separable:
            launch_gaussian_blur_separable(d_gray_, d_tmp_, d_blur_, ksize_,
                                           width_, height_, /*stream=*/0);
            break;
        case Variant::Shared:
            launch_gaussian_blur_shared(d_gray_, d_tmp_, d_blur_, ksize_,
                                        width_, height_, /*stream=*/0);
            break;
    }
    CUDA_CHECK(cudaEventRecord(ev(ev_[kAfterBlur])));

    // Sobel is tiled only in the 'shared' variant. Per-stage timings are
    // reported separately, so pairing both changes under one variant name does
    // not hide which stage the improvement came from.
    if (variant_ == Variant::Shared) {
        launch_sobel_shared(d_blur_, d_edges_, width_, height_, /*stream=*/0);
    } else {
        launch_sobel_naive(d_blur_, d_edges_, width_, height_, /*stream=*/0);
    }
    CUDA_CHECK(cudaEventRecord(ev(ev_[kAfterSobel])));

    CUDA_CHECK(cudaMemcpy2D(out.data, out.step,
                            d_edges_, static_cast<size_t>(width_),
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

void GpuPipeline::download_grayscale(cv::Mat& out) const {
    out.create(height_, width_, CV_8UC1);
    CUDA_CHECK(cudaMemcpy2D(out.data, out.step, d_gray_, static_cast<size_t>(width_),
                            static_cast<size_t>(width_), static_cast<size_t>(height_),
                            cudaMemcpyDeviceToHost));
}

void GpuPipeline::download_blurred(cv::Mat& out) const {
    out.create(height_, width_, CV_8UC1);
    CUDA_CHECK(cudaMemcpy2D(out.data, out.step, d_blur_, static_cast<size_t>(width_),
                            static_cast<size_t>(width_), static_cast<size_t>(height_),
                            cudaMemcpyDeviceToHost));
}

}  // namespace frameflow
