#include "video_io.h"

#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <string>

namespace frameflow {

namespace {

std::string quoted(const std::string& s) { return "'" + s + "'"; }

}  // namespace

VideoReader::VideoReader(const std::string& path) {
    if (!cap_.open(path)) {
        throw std::runtime_error(
            "could not open input video " + quoted(path) +
            " (file missing, or the codec is not supported by this OpenCV build)");
    }

    info_.path = path;
    info_.width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    info_.height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    info_.fps = cap_.get(cv::CAP_PROP_FPS);
    info_.frame_count = static_cast<long long>(cap_.get(cv::CAP_PROP_FRAME_COUNT));

    if (info_.width <= 0 || info_.height <= 0) {
        throw std::runtime_error("input video " + quoted(path) +
                                 " reported a degenerate resolution (" +
                                 std::to_string(info_.width) + "x" +
                                 std::to_string(info_.height) + ")");
    }

    // A container that opens but yields nothing would otherwise surface much
    // later as an empty output file, so probe the first frame now and rewind.
    cv::Mat probe;
    if (!cap_.read(probe) || probe.empty()) {
        throw std::runtime_error("input video " + quoted(path) +
                                 " opened but contains no decodable frames");
    }
    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);

    if (info_.fps <= 0.0 || info_.fps > 1000.0) {
        info_.fps = 30.0;  // container lied; fall back so the writer stays valid
    }
    if (info_.frame_count < 0) {
        info_.frame_count = 0;
    }
}

bool VideoReader::read(cv::Mat& frame) {
    if (!cap_.read(frame)) return false;
    return !frame.empty();
}

VideoWriter::VideoWriter(const std::string& path, int width, int height, double fps)
    : path_(path), size_(width, height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("refusing to open a writer for a " +
                                 std::to_string(width) + "x" +
                                 std::to_string(height) + " video");
    }
    if (fps <= 0.0) fps = 30.0;

    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    if (!writer_.open(path, fourcc, fps, size_, /*isColor=*/true)) {
        throw std::runtime_error(
            "could not open output video " + quoted(path) +
            " for writing (unwritable path, or no mp4v encoder in this OpenCV build)");
    }
}

void VideoWriter::write(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::runtime_error("attempted to write an empty frame to " + quoted(path_));
    }
    if (frame.size() != size_) {
        throw std::runtime_error(
            "frame size " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows) +
            " does not match writer size " + std::to_string(size_.width) + "x" +
            std::to_string(size_.height) + " for " + quoted(path_));
    }

    // Pipeline output after Sobel is single-channel. Most players reject a
    // grayscale mp4, so replicate to 3 channels before encoding.
    if (frame.channels() == 1) {
        cv::cvtColor(frame, bgr_scratch_, cv::COLOR_GRAY2BGR);
        writer_.write(bgr_scratch_);
    } else {
        writer_.write(frame);
    }
    ++frames_written_;
}

}  // namespace frameflow
