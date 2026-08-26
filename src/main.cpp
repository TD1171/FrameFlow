#include "video_io.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

namespace {

struct Options {
    std::string input;
    std::string output;      // empty => skip encoding entirely
    long long frames = -1;   // -1 => process the whole file
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
        "  --help            show this message\n");
}

// Returns false if the argument list is malformed; `err` explains why.
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

        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--input") {
            if (!value_for("--input", opt.input)) return false;
        } else if (arg == "--output") {
            if (!value_for("--output", opt.output)) return false;
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
    if (opt.frames > 0) {
        std::printf("Frame limit: %lld\n", opt.frames);
    }
    std::printf("\nPipeline\n");
    std::printf("------------------------\n");
    std::printf("Stage 0: passthrough (decode -> encode, no processing)\n");
    std::printf("Output: %s\n", opt.output.empty() ? "(none, decode only)" : opt.output.c_str());
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

    try {
        frameflow::VideoReader reader(opt.input);
        const frameflow::VideoInfo& info = reader.info();
        print_header(info, opt);

        std::unique_ptr<frameflow::VideoWriter> writer;
        if (!opt.output.empty()) {
            writer = std::make_unique<frameflow::VideoWriter>(
                opt.output, info.width, info.height, info.fps);
        }

        cv::Mat frame;
        long long processed = 0;
        const auto t0 = std::chrono::steady_clock::now();

        while (reader.read(frame)) {
            if (writer) writer->write(frame);
            ++processed;
            if (opt.frames > 0 && processed >= opt.frames) break;
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (processed == 0) {
            std::fprintf(stderr, "\nerror: no frames were decoded from '%s'\n",
                         opt.input.c_str());
            return EXIT_FAILURE;
        }

        const double per_frame_ms = elapsed_ms / static_cast<double>(processed);
        std::printf("\nPerformance\n");
        std::printf("------------------------\n");
        std::printf("Frames processed:        %lld\n", processed);
        std::printf("End-to-end per frame:    %.3f ms  (%.1f fps)   [decode%s]\n",
                    per_frame_ms, 1000.0 / per_frame_ms,
                    writer ? " + encode" : " only");
        std::printf("Total wall time:         %.1f ms\n\n", elapsed_ms);

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
