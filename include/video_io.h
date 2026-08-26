#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

namespace frameflow {

// Properties of a decoded source. frame_count is best-effort: some containers
// report 0 or a wrong value, so it is used for display and progress only and
// never as a loop bound.
struct VideoInfo {
    std::string path;
    int width = 0;
    int height = 0;
    double fps = 0.0;
    long long frame_count = 0;
};

// Decodes frames on the CPU via OpenCV. Frames arrive as 8-bit 3-channel BGR
// (CV_8UC3) in that byte order -- byte 0 is blue. Every downstream kernel
// depends on this and says so at its own declaration.
class VideoReader {
public:
    // Throws std::runtime_error if the file is missing, unreadable, uses an
    // unsupported codec, or reports a degenerate resolution.
    explicit VideoReader(const std::string& path);

    // Returns false at end of stream. `frame` is CV_8UC3 BGR on success.
    bool read(cv::Mat& frame);

    const VideoInfo& info() const { return info_; }

private:
    cv::VideoCapture cap_;
    VideoInfo info_;
};

// Encodes frames on the CPU via OpenCV. Encoding is optional throughout the
// pipeline: when no output path is given, no writer is constructed at all, so
// benchmark runs are not charged for encode time.
class VideoWriter {
public:
    // Throws std::runtime_error if the writer cannot be opened.
    VideoWriter(const std::string& path, int width, int height, double fps);

    // `frame` may be single-channel; it is expanded to 3 channels because most
    // players will not accept a grayscale mp4.
    void write(const cv::Mat& frame);

    long long frames_written() const { return frames_written_; }

private:
    cv::VideoWriter writer_;
    std::string path_;
    cv::Size size_;
    cv::Mat bgr_scratch_;          // reused across frames to avoid per-frame allocation
    long long frames_written_ = 0;
};

}  // namespace frameflow
