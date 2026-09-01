#include "cpu_baseline.h"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <stdexcept>

namespace frameflow {

namespace {

using Clock = std::chrono::steady_clock;

inline double ms_since(const Clock::time_point& t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

}  // namespace

void CpuTimings::accumulate(const CpuTimings& t) {
    gray_ms += t.gray_ms;
    blur_ms += t.blur_ms;
    sobel_ms += t.sobel_ms;
    total_ms += t.total_ms;
}

void CpuTimings::divide(double n) {
    gray_ms /= n;
    blur_ms /= n;
    sobel_ms /= n;
    total_ms /= n;
}

CpuBaseline::CpuBaseline(int ksize, double sigma) : ksize_(ksize), sigma_(sigma) {
    if (ksize <= 0 || ksize % 2 == 0) {
        throw std::runtime_error("CpuBaseline requires a positive odd kernel size");
    }
}

void CpuBaseline::process(const cv::Mat& bgr, cv::Mat& gray, cv::Mat& blurred,
                          cv::Mat& edges, CpuTimings& timings) const {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("CpuBaseline::process expects a non-empty CV_8UC3 frame");
    }

    const auto t_all = Clock::now();

    auto t = Clock::now();
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    timings.gray_ms = ms_since(t);

    // Explicit BORDER_REFLECT_101 rather than relying on the default, so the
    // match with the CUDA kernels is visible at the call site.
    t = Clock::now();
    cv::GaussianBlur(gray, blurred, cv::Size(ksize_, ksize_), sigma_, sigma_,
                     cv::BORDER_REFLECT_101);
    timings.blur_ms = ms_since(t);

    // Gradients are computed at CV_32F, not CV_16S. At 8-bit input the
    // gradients span [-1020, 1020], and the magnitude reaches ~1443 -- taking
    // the magnitude in a saturating integer type would clip the components
    // before they are combined, producing a different answer from the GPU
    // rather than merely a rounded one.
    t = Clock::now();
    cv::Mat gx, gy, mag;
    cv::Sobel(blurred, gx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REFLECT_101);
    cv::Sobel(blurred, gy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REFLECT_101);
    cv::magnitude(gx, gy, mag);  // true sqrt(gx^2 + gy^2), matching the kernel
    // convertTo rounds and saturates to [0,255], the same clamp the kernel does.
    mag.convertTo(edges, CV_8U);
    timings.sobel_ms = ms_since(t);

    timings.total_ms = ms_since(t_all);
}

StageError compare_u8_interior(const cv::Mat& a, const cv::Mat& b, int margin) {
    if (a.size() != b.size() || a.type() != b.type() || a.type() != CV_8UC1) {
        throw std::runtime_error("compare_u8_interior requires two same-size CV_8UC1 images");
    }
    if (margin < 0) margin = 0;
    if (a.cols <= 2 * margin || a.rows <= 2 * margin) {
        throw std::runtime_error("margin leaves no interior region to compare");
    }

    StageError e;
    double sum = 0.0;
    int worst = 0;
    long long over_one = 0;
    long long n = 0;

    for (int y = margin; y < a.rows - margin; ++y) {
        const uint8_t* pa = a.ptr<uint8_t>(y);
        const uint8_t* pb = b.ptr<uint8_t>(y);
        for (int x = margin; x < a.cols - margin; ++x) {
            const int d = std::abs(static_cast<int>(pa[x]) - static_cast<int>(pb[x]));
            sum += d;
            if (d > worst) worst = d;
            if (d > 1) ++over_one;
            ++n;
        }
    }

    e.pixels = n;
    e.mean_abs = sum / static_cast<double>(n);
    e.max_abs = worst;
    e.over_one = over_one;
    return e;
}

StageError compare_u8(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type() || a.type() != CV_8UC1) {
        throw std::runtime_error("compare_u8 requires two same-size CV_8UC1 images");
    }

    StageError e;
    e.pixels = static_cast<long long>(a.total());

    double sum = 0.0;
    int worst = 0;
    long long over_one = 0;

    for (int y = 0; y < a.rows; ++y) {
        const uint8_t* pa = a.ptr<uint8_t>(y);
        const uint8_t* pb = b.ptr<uint8_t>(y);
        for (int x = 0; x < a.cols; ++x) {
            const int d = std::abs(static_cast<int>(pa[x]) - static_cast<int>(pb[x]));
            sum += d;
            if (d > worst) worst = d;
            if (d > 1) ++over_one;
        }
    }

    e.mean_abs = sum / static_cast<double>(e.pixels);
    e.max_abs = worst;
    e.over_one = over_one;
    return e;
}

}  // namespace frameflow
