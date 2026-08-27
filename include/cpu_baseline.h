#pragma once

#include <opencv2/core.hpp>

namespace frameflow {

// Per-frame CPU timings in milliseconds, measured with a host clock. A host
// clock is correct here: OpenCV's calls are synchronous, unlike kernel launches.
struct CpuTimings {
    double gray_ms = 0.0;
    double blur_ms = 0.0;
    double sobel_ms = 0.0;
    double total_ms = 0.0;

    void accumulate(const CpuTimings& t);
    void divide(double n);
};

// CPU reference implementation of the same three stages, built on OpenCV.
//
// This serves two distinct purposes and it matters not to conflate them:
//
//   PERFORMANCE BASELINE. OpenCV's cvtColor/GaussianBlur/Sobel are SIMD
//   vectorised and multithreaded. They are deliberately NOT crippled -- no
//   forced single thread, no debug build, no naive scalar rewrite. If the GPU
//   speedup comes out modest against a well-optimised CPU, that is the honest
//   result and it is reported as such.
//
//   VALIDATION REFERENCE. Useful, but NOT a bit-exact oracle: cvtColor
//   dispatches to fixed-point integer paths whose coefficients differ slightly
//   from the exact BT.601 constants, and which vary by build and CPU. So GPU
//   vs OpenCV is compared against a stated tolerance, not for exact equality.
//   Bit-exact correctness is established separately, against a float64 NumPy
//   oracle written from the mathematical definition (scripts/verify_stages.py).
//
// Border mode: every OpenCV call here uses BORDER_REFLECT_101, which is both
// OpenCV's convolution default and what the CUDA kernels implement. Matching
// this is what allows validation over the full image with zero excluded pixels.
class CpuBaseline {
public:
    CpuBaseline(int ksize, double sigma);

    // Runs all three stages, filling each intermediate so per-stage error can
    // be attributed rather than only the final result being compared.
    void process(const cv::Mat& bgr, cv::Mat& gray, cv::Mat& blurred,
                 cv::Mat& edges, CpuTimings& timings) const;

private:
    int ksize_;
    double sigma_;
};

// Absolute-difference statistics between two single-channel 8-bit images.
struct StageError {
    double mean_abs = 0.0;
    double max_abs = 0.0;
    long long pixels = 0;
    long long over_one = 0;  // pixels differing by more than one level
};

StageError compare_u8(const cv::Mat& a, const cv::Mat& b);

}  // namespace frameflow
