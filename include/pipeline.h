#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

namespace frameflow {

// Kernel implementation selected at runtime by --kernel. The progression is the
// point of the project: each variant computes the same result by a different
// memory strategy, so their timings are directly comparable.
enum class Variant {
    Naive,      // direct 2D convolution, every tap from global memory
    Separable,  // two 1D passes: 2K taps instead of K*K
    Shared,     // separable + shared-memory tiling with halo regions
};

const char* variant_name(Variant v);
const char* variant_description(Variant v);

// Returns false if `s` is not a known variant name.
bool parse_variant(const std::string& s, Variant& out);

// Accessors for the per-launch synchronization mode. Host translation units use
// these rather than touching the flag directly, so that main.cpp does not have
// to include <cuda_runtime.h> -- it is compiled by the host compiler, which has
// no CUDA include path. See cuda_check.h for what the mode actually changes.
void set_debug_sync(bool enabled);
bool debug_sync();

// Per-frame GPU timings in milliseconds, measured with cudaEvent records.
// Host clocks are not used for GPU work: kernel launches are asynchronous, so a
// host timer around a launch measures the launch call, not the kernel.
//
// CAVEAT on kernel_ms: the uploads and downloads here are synchronous copies
// from pageable host memory, so the GPU is idle at the moment the post-upload
// event is recorded. kernel_ms therefore includes the host-side launch latency
// of the first kernel in the chain, and is a slight OVER-estimate of pure
// compute. Pinned memory plus async copies would remove this; that is Tier 3.
struct FrameTimings {
    float upload_ms = 0.0f;
    float gray_ms = 0.0f;
    float blur_ms = 0.0f;
    float sobel_ms = 0.0f;
    float kernel_ms = 0.0f;    // gray + blur + sobel
    float download_ms = 0.0f;
    float gpu_total_ms = 0.0f; // upload + kernels + download

    void accumulate(const FrameTimings& t);
    void divide(double n);
};

// Owns every device allocation for the pipeline and reuses them across frames.
// Allocating per frame would put a cudaMalloc/cudaFree pair -- both implicitly
// synchronizing -- inside the measured loop and dominate the timings.
class GpuPipeline {
public:
    // ksize must be positive and odd. sigma <= 0 selects OpenCV's default
    // sigma-from-ksize rule so the two implementations still agree.
    GpuPipeline(int width, int height, int ksize, double sigma, Variant variant);
    ~GpuPipeline();

    GpuPipeline(const GpuPipeline&) = delete;
    GpuPipeline& operator=(const GpuPipeline&) = delete;

    // Uploads one BGR frame, runs grayscale -> blur -> Sobel entirely on the
    // device, and downloads the final edge map. `out` is resized to CV_8UC1.
    // No intermediate result returns to the host.
    void process(const cv::Mat& bgr, cv::Mat& out, FrameTimings& timings);

    // Copies an intermediate stage back to the host. For --save-frames and for
    // per-stage validation only; never called inside a timed loop.
    void download_grayscale(cv::Mat& out) const;
    void download_blurred(cv::Mat& out) const;

    int width() const { return width_; }
    int height() const { return height_; }
    int ksize() const { return ksize_; }
    double sigma() const { return sigma_; }
    Variant variant() const { return variant_; }

private:
    void free_all() noexcept;

    int width_ = 0;
    int height_ = 0;
    int ksize_ = 5;
    double sigma_ = 1.4;
    Variant variant_ = Variant::Naive;

    uint8_t* d_bgr_ = nullptr;    // width*height*3
    uint8_t* d_gray_ = nullptr;   // width*height
    uint8_t* d_blur_ = nullptr;   // width*height
    uint8_t* d_edges_ = nullptr;  // width*height
    float* d_tmp_ = nullptr;      // width*height, separable intermediate

    // Filter weights are NOT stored per object: they live in __constant__
    // memory, which belongs to the module. This counter enforces that only one
    // pipeline is live at a time, so a second one cannot overwrite the first's
    // weights and silently make it compute the wrong filter.
    static int live_instances_;

    // Reused across frames; creating events per frame is a measurable cost.
    void* ev_[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

}  // namespace frameflow
