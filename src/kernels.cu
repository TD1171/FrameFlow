#include "kernels.cuh"

#include "cuda_check.h"

namespace frameflow {

// Defined here rather than in a header so there is exactly one instance.
// Debug builds default to per-launch synchronization; see cuda_check.h.
#ifdef FRAMEFLOW_DEBUG_SYNC
bool g_debug_sync = true;
#else
bool g_debug_sync = false;
#endif

namespace {

// BT.601 luminance weights, indexed to match BGR byte order.
constexpr float kWeightB = 0.114f;
constexpr float kWeightG = 0.587f;
constexpr float kWeightR = 0.299f;

// ---------------------------------------------------------------------------
// One thread per output pixel. Purely element-wise: no neighbour access, no
// shared memory, nothing to synchronize. This is the warm-up kernel whose job
// is to establish the indexing and bounds-checking pattern the later
// convolution kernels reuse.
//
// Thread mapping: x is threadIdx.x so that consecutive threads in a warp handle
// consecutive columns. Images are row-major, so those threads then touch
// consecutive addresses and the warp's loads coalesce into a small number of
// memory transactions. Mapping x to threadIdx.y instead would make each thread
// in a warp stride a whole row apart, turning one coalesced access into 32
// separate ones.
//
// The grayscale write is perfectly coalesced (1 byte per thread, consecutive).
// The BGR read is 3 bytes per thread, so a warp covers 96 contiguous bytes --
// still contiguous, but not a single aligned transaction. That inefficiency is
// inherent to interleaved 3-channel data and is left as-is: it is honest, and
// converting to a planar layout would move work rather than remove it.
// ---------------------------------------------------------------------------
__global__ void bgrToGrayscale(const uint8_t* __restrict__ bgr,
                               uint8_t* __restrict__ gray,
                               int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid dimensions are rounded up, so the last block in each dimension runs
    // threads past the image edge. Without this guard they would read and write
    // out of bounds -- the defect that hides behind power-of-two test images.
    if (x >= width || y >= height) return;

    const int src = (y * width + x) * 3;

    const float b = static_cast<float>(bgr[src + 0]);
    const float g = static_cast<float>(bgr[src + 1]);
    const float r = static_cast<float>(bgr[src + 2]);

    // rintf rounds half to even, matching how OpenCV's integer path rounds, so
    // the CPU/GPU comparison in --validate is not offset by a systematic +-0.5.
    const float y_lum = kWeightB * b + kWeightG * g + kWeightR * r;
    gray[y * width + x] = static_cast<uint8_t>(rintf(y_lum));
}

}  // namespace

void launch_bgr_to_grayscale(const uint8_t* d_bgr, uint8_t* d_gray,
                             int width, int height, cudaStream_t stream) {
    // 16x16 = 256 threads. A multiple of the 32-thread warp size, and small
    // enough to keep several blocks resident per SM for latency hiding.
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    bgrToGrayscale<<<grid, block, 0, stream>>>(d_bgr, d_gray, width, height);
    CUDA_CHECK_KERNEL("bgrToGrayscale");
}

}  // namespace frameflow
