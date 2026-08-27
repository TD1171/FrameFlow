#include "pipeline.h"
#include "video_io.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

namespace {

struct Options {
    std::string input;
    std::string output;       // empty => skip encoding entirely
    long long frames = -1;    // -1 => process the whole file
    int ksize = 5;
    double sigma = 1.4;
    int warmup = 30;
    bool save_frames = false;
    bool debug_sync = false;
};

void print_usage() {
    std::printf(
        "FrameFlow CUDA Video Pipeline\n"
        "\n"
        "Usage: frameflow --input <path> [options]\n"
        "\n"
        "  --input <path>    source video (required)\n"
        "  --output <path>   destination video; omit to skip encoding\n"
        "  --frames N        process only the first N frames\n"
        "  --ksize N         Gaussian kernel size, odd (default 5)\n"
        "  --sigma S         Gaussian sigma (default 1.4)\n"
        "  --warmup N        untimed warmup iterations before measuring\n"
        "                    (default 30; clock speeds need this to settle)\n"
        "  --save-frames     write a lossless PNG of each pipeline stage for\n"
        "                    the first processed frame, into results/\n"
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
        } else if (arg == "--save-frames") {
            opt.save_frames = true;
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
    std::printf("Kernel variant: naive (direct 2D convolution, global memory)\n");
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
        frameflow::VideoReader reader(opt.input);
        const frameflow::VideoInfo& info = reader.info();
        print_header(info, opt);

        frameflow::GpuPipeline pipeline(info.width, info.height, opt.ksize, opt.sigma);

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
                cv::imwrite("results/stage0_original.png", frame);
                cv::imwrite("results/stage1_grayscale.png", gray);
                cv::imwrite("results/stage2_blurred.png", blurred);
                cv::imwrite("results/stage3_edges.png", result);
                std::printf("\nSaved stage PNGs to results/\n");
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
