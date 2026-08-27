#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace frameflow {

// ---------------------------------------------------------------------------
// Stage 1: BGR -> grayscale
//
// INPUT CHANNEL ORDER IS BGR, NOT RGB. cv::VideoCapture delivers frames with
// byte 0 = blue, byte 1 = green, byte 2 = red. The kernel is named for what it
// actually consumes so the assumption cannot drift; feeding it RGB would swap
// the blue and red weights and produce a plausible-looking but wrong image.
//
// Luminance is ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B, with each
// coefficient applied to the correct byte.
//
// Buffers are tightly packed: the BGR image has a row stride of width*3 bytes
// and the grayscale image width bytes. Host frames may be padded, so uploads
// use cudaMemcpy2D to reconcile the two strides.
// ---------------------------------------------------------------------------
void launch_bgr_to_grayscale(const uint8_t* d_bgr, uint8_t* d_gray,
                             int width, int height, cudaStream_t stream);

}  // namespace frameflow
