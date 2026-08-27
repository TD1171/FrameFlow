#include "benchmark.h"

#include "cpu_baseline.h"
#include "video_io.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace frameflow {

namespace {

struct Resolution {
    int width;
    int height;
    const char* label;
};

// 4K is attempted last so that an out-of-memory failure there does not cost the
// results for every smaller configuration.
const Resolution kResolutions[] = {
    {640, 480, "640x480"},
    {1280, 720, "1280x720"},
    {1920, 1080, "1920x1080"},
    {3840, 2160, "3840x2160"},
};

const Variant kVariants[] = {Variant::Naive, Variant::Separable, Variant::Shared};

// The median, not the mean. A single frame delayed by an OS scheduling hiccup
// or a driver housekeeping pass shifts a mean noticeably while leaving the
// median where it belongs, and those outliers are not what the kernel costs.
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const double hi = v[mid];
    if (v.size() % 2 != 0) return hi;
    std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
    return 0.5 * (hi + v[mid - 1]);
}

struct StageStats {
    double gray = 0.0, blur = 0.0, sobel = 0.0, total = 0.0;
};

// One row per stage keeps the CSV in long format, which is what a plotting
// script wants: no column has to be renamed when a stage is added.
void write_rows(std::ofstream& csv, const Resolution& res, const char* variant,
                const StageStats& gpu, const StageStats& cpu, double gpu_total_ms) {
    const struct { const char* name; double g, c; } rows[] = {
        {"grayscale", gpu.gray, cpu.gray},
        {"blur", gpu.blur, cpu.blur},
        {"sobel", gpu.sobel, cpu.sobel},
        {"total", gpu.total, cpu.total},
    };
    for (const auto& r : rows) {
        const double sp_kernel = r.g > 0.0 ? r.c / r.g : 0.0;
        const double sp_total = gpu_total_ms > 0.0 ? cpu.total / gpu_total_ms : 0.0;
        csv << res.label << ',' << res.width << ',' << res.height << ',' << variant << ','
            << r.name << ',' << r.g << ',' << gpu_total_ms << ',' << r.c << ','
            << sp_kernel << ',' << sp_total << '\n';
    }
}

}  // namespace

int run_benchmark(const BenchmarkOptions& opt) {
    VideoReader reader(opt.input);
    const VideoInfo& info = reader.info();

    // Decode once into memory. Re-decoding per configuration would put video
    // decode inside the sweep and make the whole thing decode-bound.
    std::vector<cv::Mat> source;
    cv::Mat frame;
    while (static_cast<int>(source.size()) < opt.frames && reader.read(frame)) {
        source.push_back(frame.clone());
    }
    if (source.empty()) {
        std::fprintf(stderr, "error: no frames decoded from '%s'\n", opt.input.c_str());
        return EXIT_FAILURE;
    }

    std::printf("\nFrameFlow Benchmark\n");
    std::printf("------------------------\n");
    std::printf("Source: %s (%dx%d)\n", info.path.c_str(), info.width, info.height);
    std::printf("Filter: %dx%d, sigma=%.2f\n", opt.ksize, opt.ksize, opt.sigma);
    std::printf("Per configuration: %zu frames x %d repeats, %d warmup iterations\n",
                source.size(), opt.repeats, opt.warmup);
    std::printf("Statistic: median of per-frame samples\n");
    std::printf("Sync mode: launch check only (no per-launch synchronize)\n\n");

    std::ofstream csv(opt.csv_path);
    if (!csv) {
        std::fprintf(stderr, "error: cannot write '%s'\n", opt.csv_path.c_str());
        return EXIT_FAILURE;
    }
    csv << "resolution,width,height,variant,stage,kernel_ms,gpu_total_ms,cpu_ms,"
           "speedup_kernel,speedup_total\n";
    csv.setf(std::ios::fixed);
    csv.precision(6);

    std::printf("%-11s %-10s %8s %8s %8s %9s %10s %9s\n", "resolution", "variant",
                "gray", "blur", "sobel", "kernel", "gpu_total", "cpu");
    std::printf("%-11s %-10s %8s %8s %8s %9s %10s %9s\n", "----------", "-------",
                "----", "----", "-----", "------", "---------", "---");

    int completed = 0;

    for (const Resolution& res : kResolutions) {
        // Resizing happens once per resolution, outside every timed region.
        std::vector<cv::Mat> frames;
        frames.reserve(source.size());
        for (const cv::Mat& f : source) {
            cv::Mat r;
            cv::resize(f, r, cv::Size(res.width, res.height), 0, 0, cv::INTER_AREA);
            frames.push_back(r);
        }

        // The CPU baseline depends only on resolution, so it is measured once
        // per resolution rather than once per variant.
        StageStats cpu{};
        {
            CpuBaseline baseline(opt.ksize, opt.sigma);
            cv::Mat g, b, e;
            CpuTimings t{};
            for (int i = 0; i < 3; ++i) baseline.process(frames[0], g, b, e, t);  // warm

            std::vector<double> cg, cb, cs, ct;
            for (int rep = 0; rep < opt.repeats; ++rep) {
                for (const cv::Mat& f : frames) {
                    baseline.process(f, g, b, e, t);
                    cg.push_back(t.gray_ms);
                    cb.push_back(t.blur_ms);
                    cs.push_back(t.sobel_ms);
                    ct.push_back(t.total_ms);
                }
            }
            cpu = {median(cg), median(cb), median(cs), median(ct)};
        }

        for (Variant variant : kVariants) {
            try {
                // Scoped so the pipeline is destroyed before the next one is
                // built: filter weights live in __constant__ memory and only
                // one pipeline may be live at a time.
                StageStats gpu{};
                double gpu_total = 0.0;
                {
                    GpuPipeline pipeline(res.width, res.height, opt.ksize, opt.sigma,
                                         variant);
                    cv::Mat out;
                    FrameTimings t{};

                    for (int i = 0; i < opt.warmup; ++i) pipeline.process(frames[0], out, t);

                    std::vector<double> gg, gb, gs, gk, gt;
                    for (int rep = 0; rep < opt.repeats; ++rep) {
                        for (const cv::Mat& f : frames) {
                            pipeline.process(f, out, t);
                            gg.push_back(t.gray_ms);
                            gb.push_back(t.blur_ms);
                            gs.push_back(t.sobel_ms);
                            gk.push_back(t.kernel_ms);
                            gt.push_back(t.gpu_total_ms);
                        }
                    }
                    gpu = {median(gg), median(gb), median(gs), median(gk)};
                    gpu_total = median(gt);
                }

                write_rows(csv, res, variant_name(variant), gpu, cpu, gpu_total);
                std::printf("%-11s %-10s %8.3f %8.3f %8.3f %9.3f %10.3f %9.3f\n",
                            res.label, variant_name(variant), gpu.gray, gpu.blur,
                            gpu.sobel, gpu.total, gpu_total, cpu.total);
                ++completed;
            } catch (const std::exception& e) {
                // A resolution that will not fit is reported and skipped rather
                // than aborting the sweep, so 4K failing does not discard the
                // results already gathered.
                std::printf("%-11s %-10s SKIPPED: %s\n", res.label,
                            variant_name(variant), e.what());
            }
        }
    }

    csv.close();
    if (completed == 0) {
        std::fprintf(stderr, "\nerror: no configuration completed\n");
        return EXIT_FAILURE;
    }

    std::printf("\nWrote %d configurations to %s\n", completed, opt.csv_path.c_str());
    std::printf("All times are milliseconds per frame. 'kernel' excludes H2D/D2H\n");
    std::printf("transfer; 'gpu_total' includes it; neither includes decode or encode.\n\n");
    return EXIT_SUCCESS;
}

}  // namespace frameflow
