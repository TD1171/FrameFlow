#include "benchmark.h"
#include "cpu_baseline.h"
#include "pipeline.h"
#include "video_io.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct Options {
    std::string input;
    std::string output;       // empty => skip encoding entirely
    long long frames = -1;    // -1 => process the whole file
    int ksize = 5;
    double sigma = 1.4;
    int warmup = 30;
    // Default is naive until the shared variant lands, so the default path
    // is always one that actually exists.
    frameflow::Variant variant = frameflow::Variant::Naive;
    bool save_frames = false;
    // Where --save-frames writes. Configurable so the test suite can dump
    // into a scratch directory instead of overwriting the committed
    // README sample frames in results/.
    std::string frames_dir = "results";
    bool debug_sync = false;
    bool validate = false;
    bool benchmark = false;
};

// GPU/CPU agreement threshold, on an 8-bit scale. Exact equality is not
// achievable and not the goal: OpenCV's cvtColor uses fixed-point coefficients
// that differ slightly from the exact BT.601 constants, and float summation
// order differs between the two implementations. A mean absolute error below
// one level means the two agree to within rounding on essentially every pixel.
//
// This threshold is NOT to be relaxed to make a failing kernel pass. If a stage
// exceeds it, the kernel is wrong.
constexpr double kValidationTolerance = 1.0;

void print_usage() {
    std::printf(
        "FrameFlow CUDA Video Pipeline\n"
        "\n"
        "Usage: frameflow --input <path> [options]\n"
        "\n"
        "  --input <path>    source video (required)\n"
        "  --output <path>   destination video; omit to skip encoding\n"
        "  --frames N        process only the first N frames\n"
        "  --kernel V        blur variant: naive | separable | shared\n"
        "  --ksize N         Gaussian kernel size, odd (default 5)\n"
        "  --sigma S         Gaussian sigma (default 1.4)\n"
        "  --warmup N        untimed warmup iterations before measuring\n"
        "                    (default 30; clock speeds need this to settle)\n"
        "  --save-frames     write a lossless PNG of each pipeline stage for\n"
        "                    the first processed frame\n"
        "  --frames-dir D    directory for --save-frames (default results)\n"
        "  --validate        run the OpenCV CPU baseline alongside, report\n"
        "                    per-stage error and CPU/GPU timings, then exit\n"
        "  --benchmark       sweep every variant across resolutions, write\n"
        "                    results/benchmarks.csv, then exit\n"
        "  --debug-sync      synchronize after every kernel launch (slower;\n"
        "                    attributes a fault to the launch that caused it)\n"
        "  --help            show this message\n");
}

bool parse_args(int argc, char** argv, Options& opt, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto value_for = [&](const char* name, std::string& out) -> bool {
            if (i + 1 >= argc) {
                err = std::string("missing value after ") + name;
                return false;
            }
            out = argv[++i];
            return true;
        };

        auto int_value = [&](const char* name, int& out) -> bool {
            std::string v;
            if (!value_for(name, v)) return false;
            try {
                out = std::stoi(v);
            } catch (const std::exception&) {
                err = std::string(name) + " expects an integer, got '" + v + "'";
                return false;
            }
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--input") {
            if (!value_for("--input", opt.input)) return false;
        } else if (arg == "--output") {
            if (!value_for("--output", opt.output)) return false;
        } else if (arg == "--kernel") {
            std::string v;
            if (!value_for("--kernel", v)) return false;
            if (!frameflow::parse_variant(v, opt.variant)) {
                err = "--kernel expects naive, separable or shared; got '" + v + "'";
                return false;
            }
        } else if (arg == "--save-frames") {
            opt.save_frames = true;
        } else if (arg == "--frames-dir") {
            if (!value_for("--frames-dir", opt.frames_dir)) return false;
        } else if (arg == "--validate") {
            opt.validate = true;
        } else if (arg == "--benchmark") {
            opt.benchmark = true;
        } else if (arg == "--debug-sync") {
            opt.debug_sync = true;
        } else if (arg == "--ksize") {
            if (!int_value("--ksize", opt.ksize)) return false;
            if (opt.ksize <= 0 || opt.ksize % 2 == 0) {
                err = "--ksize must be positive and odd, got " + std::to_string(opt.ksize);
                return false;
            }
        } else if (arg == "--warmup") {
            if (!int_value("--warmup", opt.warmup)) return false;
            if (opt.warmup < 0) {
                err = "--warmup cannot be negative";
                return false;
            }
        } else if (arg == "--sigma") {
            std::string v;
            if (!value_for("--sigma", v)) return false;
            try {
                opt.sigma = std::stod(v);
            } catch (const std::exception&) {
                err = "--sigma expects a number, got '" + v + "'";
                return false;
            }
            if (opt.sigma <= 0.0) {
                err = "--sigma must be positive, got '" + v + "'";
                return false;
            }
        } else if (arg == "--frames") {
            std::string v;
            if (!value_for("--frames", v)) return false;
            try {
                opt.frames = std::stoll(v);
            } catch (const std::exception&) {
                err = "--frames expects an integer, got '" + v + "'";
                return false;
            }
            if (opt.frames <= 0) {
                err = "--frames must be positive, got '" + v + "'";
                return false;
            }
        } else {
            err = "unrecognized argument '" + arg + "'";
            return false;
        }
    }

    if (opt.input.empty()) {
        err = "--input is required";
        return false;
    }
    return true;
}

void print_header(const frameflow::VideoInfo& info, const Options& opt) {
    std::printf("\nFrameFlow CUDA Video Pipeline\n\n");
    std::printf("Input\n");
    std::printf("------------------------\n");
    std::printf("File: %s\n", info.path.c_str());
    std::printf("Resolution: %dx%d\n", info.width, info.height);
    if (info.frame_count > 0) {
        std::printf("Frames: %lld\n", info.frame_count);
    } else {
        std::printf("Frames: (not reported by container)\n");
    }
    std::printf("FPS (source): %.1f\n", info.fps);
    if (opt.frames > 0) std::printf("Frame limit: %lld\n", opt.frames);

    std::printf("\nPipeline\n");
    std::printf("------------------------\n");
    std::printf("Stage 1: BGR -> Grayscale\n");
    std::printf("Stage 2: Gaussian Blur (%dx%d, sigma=%.2f)\n", opt.ksize, opt.ksize,
                opt.sigma);
    std::printf("Stage 3: Sobel Edge Detection\n");
    std::printf("Kernel variant: %s (%s)\n", frameflow::variant_name(opt.variant),
                frameflow::variant_description(opt.variant));
    std::printf("Border mode: BORDER_REFLECT_101 (matches OpenCV default)\n");
    std::printf("Sync mode: %s\n",
                frameflow::debug_sync() ? "per-launch synchronize (debug)"
                                        : "launch check only (benchmark)");
    std::printf("Warmup iterations: %d\n", opt.warmup);
    std::printf("Output: %s\n",
                opt.output.empty() ? "(none, decode + process only)" : opt.output.c_str());
}

void print_fps_line(const char* label, double ms, const char* note) {
    std::printf("%-25s %8.3f ms  (%7.1f fps)   %s\n", label, ms,
                ms > 0.0 ? 1000.0 / ms : 0.0, note);
}

// Runs the CPU baseline alongside the GPU and reports per-stage agreement.
//
// Per-stage rather than final-only: an error in grayscale propagates through
// blur into Sobel, so comparing only the edge map tells you something is wrong
// but not which kernel is responsible.
int run_validate(const Options& opt) {
    frameflow::VideoReader reader(opt.input);
    const frameflow::VideoInfo& info = reader.info();

    // Per-launch synchronization while validating, so a fault is reported at
    // the launch that caused it. This makes the GPU timings below unsuitable
    // as benchmark figures, which is stated where they are printed.
    frameflow::set_debug_sync(true);
    print_header(info, opt);

    frameflow::GpuPipeline pipeline(info.width, info.height, opt.ksize, opt.sigma,
                                    opt.variant);
    frameflow::CpuBaseline cpu(opt.ksize, opt.sigma);

    const long long limit = opt.frames > 0 ? opt.frames : 20;

    cv::Mat frame, gpu_gray, gpu_blur, gpu_edges, cpu_gray, cpu_blur, cpu_edges;
    frameflow::FrameTimings gt{}, gpu_total{};
    frameflow::CpuTimings ct{}, cpu_total{};

    const char* names[3] = {"Grayscale", "Blur", "Sobel"};
    double sum_mean[3] = {0.0, 0.0, 0.0};
    double worst_max[3] = {0.0, 0.0, 0.0};
    long long worst_over1[3] = {0, 0, 0};
    long long n = 0;

    while (n < limit && reader.read(frame)) {
        pipeline.process(frame, gpu_edges, gt);
        pipeline.download_grayscale(gpu_gray);
        pipeline.download_blurred(gpu_blur);

        cpu.process(frame, cpu_gray, cpu_blur, cpu_edges, ct);

        const frameflow::StageError e[3] = {
            frameflow::compare_u8(gpu_gray, cpu_gray),
            frameflow::compare_u8(gpu_blur, cpu_blur),
            frameflow::compare_u8(gpu_edges, cpu_edges),
        };
        for (int i = 0; i < 3; ++i) {
            sum_mean[i] += e[i].mean_abs;
            worst_max[i] = std::max(worst_max[i], e[i].max_abs);
            worst_over1[i] = std::max(worst_over1[i], e[i].over_one);
        }

        gpu_total.accumulate(gt);
        cpu_total.accumulate(ct);
        ++n;
    }

    if (n == 0) {
        std::fprintf(stderr, "\nerror: no frames were decoded from '%s'\n",
                     opt.input.c_str());
        return EXIT_FAILURE;
    }

    gpu_total.divide(static_cast<double>(n));
    cpu_total.divide(static_cast<double>(n));

    bool passed = true;
    for (int i = 0; i < 3; ++i) {
        if (sum_mean[i] / static_cast<double>(n) > kValidationTolerance) passed = false;
    }

    std::printf("\nValidation  (%lld frames, GPU vs OpenCV CPU baseline)\n", n);
    std::printf("------------------------\n");
    std::printf("Border mode: BORDER_REFLECT_101 on both sides\n");
    std::printf("Border pixels excluded: 0 (full-image comparison, %lld px/frame)\n",
                static_cast<long long>(info.width) * info.height);
    std::printf("Tolerance: mean absolute error <= %.1f on an 8-bit scale\n\n",
                kValidationTolerance);
    std::printf("%-12s %12s %12s %16s\n", "stage", "mean abs", "max abs", "px differing >1");
    for (int i = 0; i < 3; ++i) {
        std::printf("%-12s %12.5f %12.0f %16lld\n", names[i],
                    sum_mean[i] / static_cast<double>(n), worst_max[i], worst_over1[i]);
    }
    std::printf("\nGPU vs CPU (OpenCV): %s\n", passed ? "PASSED" : "FAILED");

    std::printf("\nTiming  (per frame; GPU figures are NOT benchmark numbers --\n");
    std::printf("         --validate forces a synchronize after every launch)\n");
    std::printf("------------------------\n");
    std::printf("%-12s %10s %10s %10s\n", "stage", "GPU ms", "CPU ms", "speedup");
    const double g[3] = {gpu_total.gray_ms, gpu_total.blur_ms, gpu_total.sobel_ms};
    const double c[3] = {cpu_total.gray_ms, cpu_total.blur_ms, cpu_total.sobel_ms};
    for (int i = 0; i < 3; ++i) {
        std::printf("%-12s %10.3f %10.3f %9.1fx\n", names[i], g[i], c[i],
                    g[i] > 0.0 ? c[i] / g[i] : 0.0);
    }
    std::printf("%-12s %10.3f %10.3f %9.1fx\n", "TOTAL", gpu_total.kernel_ms,
                cpu_total.total_ms,
                gpu_total.kernel_ms > 0.0 ? cpu_total.total_ms / gpu_total.kernel_ms : 0.0);
    std::printf("\n");

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    std::string err;
    if (!parse_args(argc, argv, opt, err)) {
        std::fprintf(stderr, "error: %s\n\n", err.c_str());
        print_usage();
        return EXIT_FAILURE;
    }
    frameflow::set_debug_sync(opt.debug_sync);

    try {
        if (opt.benchmark) {
            frameflow::BenchmarkOptions b;
            b.input = opt.input;
            b.ksize = opt.ksize;
            b.sigma = opt.sigma;
            b.warmup = opt.warmup;
            if (opt.frames > 0) b.frames = static_cast<int>(opt.frames);
            return frameflow::run_benchmark(b);
        }

        if (opt.validate) return run_validate(opt);

        frameflow::VideoReader reader(opt.input);
        const frameflow::VideoInfo& info = reader.info();
        print_header(info, opt);

        frameflow::GpuPipeline pipeline(info.width, info.height, opt.ksize, opt.sigma,
                                    opt.variant);

        std::unique_ptr<frameflow::VideoWriter> writer;
        if (!opt.output.empty()) {
            writer = std::make_unique<frameflow::VideoWriter>(
                opt.output, info.width, info.height, info.fps);
        }

        cv::Mat frame, result;
        frameflow::FrameTimings total{}, one{};
        long long processed = 0;

        if (!reader.read(frame)) {
            std::fprintf(stderr, "\nerror: no frames were decoded from '%s'\n",
                         opt.input.c_str());
            return EXIT_FAILURE;
        }

        // Warm up on the first frame, repeatedly and untimed. One iteration is
        // not enough: measured kernel time falls by more than 2x over the first
        // ~100 frames as the GPU leaves its idle clock state. Warming on a
        // single frame avoids consuming the stream, so frame 0 is still the
        // first frame measured and written.
        for (int i = 0; i < opt.warmup; ++i) {
            pipeline.process(frame, result, one);
        }

        // t0 starts AFTER warmup. Starting it before would fold context
        // creation and clock ramp-up into end-to-end time, which at small
        // --frames values dominates the figure it is supposed to report.
        const auto t0 = std::chrono::steady_clock::now();

        do {
            pipeline.process(frame, result, one);
            total.accumulate(one);

            if (opt.save_frames && processed == 0) {
                cv::Mat gray, blurred;
                pipeline.download_grayscale(gray);
                pipeline.download_blurred(blurred);
                const std::string dir = opt.frames_dir + "/";
                // imwrite returns false rather than throwing when the directory
                // does not exist. Silently writing nothing would let a test that
                // depends on these files fail somewhere far less obvious.
                const std::pair<const char*, const cv::Mat*> outputs[] = {
                    {"stage0_original.png", &frame},
                    {"stage1_grayscale.png", &gray},
                    {"stage2_blurred.png", &blurred},
                    {"stage3_edges.png", &result},
                };
                for (const auto& o : outputs) {
                    if (!cv::imwrite(dir + o.first, *o.second)) {
                        throw std::runtime_error("could not write " + dir + o.first +
                                                 " (does the directory exist?)");
                    }
                }
                std::printf("\nSaved stage PNGs to %s\n", opt.frames_dir.c_str());
            }

            if (writer) writer->write(result);
            ++processed;
            if (opt.frames > 0 && processed >= opt.frames) break;
        } while (reader.read(frame));

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total.divide(static_cast<double>(processed));
        const double end_to_end_ms = elapsed_ms / static_cast<double>(processed);

        std::printf("\nPerformance  (per frame, mean over %lld frames)\n", processed);
        std::printf("------------------------\n");
        std::printf("%-25s %8.3f ms\n", "  Grayscale:", total.gray_ms);
        std::printf("%-25s %8.3f ms\n", "  Blur:", total.blur_ms);
        std::printf("%-25s %8.3f ms\n", "  Sobel:", total.sobel_ms);
        print_fps_line("Kernel total:", total.kernel_ms, "[compute only]");
        print_fps_line("GPU total:", total.gpu_total_ms, "[includes H2D/D2H transfer]");
        print_fps_line("End-to-end:", end_to_end_ms,
                       writer ? "[includes decode + encode]" : "[includes decode]");
        std::printf("%-25s %8.3f ms / %8.3f ms\n", "  H2D / D2H:", total.upload_ms,
                    total.download_ms);
        std::printf("%-25s %8.1f ms\n\n", "Total wall time:", elapsed_ms);

        if (writer) {
            std::printf("Wrote %lld frames to %s\n\n", writer->frames_written(),
                        opt.output.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
