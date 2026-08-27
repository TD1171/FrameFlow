#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace frameflow {

// Controls whether a device synchronize follows every kernel launch.
//
// There are two distinct needs here and conflating them produces either
// unattributable crashes or dishonest benchmarks:
//
//   Debug / validation -- sync after every launch so a fault is reported at the
//   launch that caused it rather than at some later, unrelated API call. Kernel
//   launches are asynchronous, so without this the error surfaces wherever the
//   host next happens to synchronize.
//
//   Benchmark -- check the launch for configuration errors (this is a host-side
//   check and costs nothing) but do NOT force a sync. A cudaDeviceSynchronize()
//   after every launch serializes the pipeline, folds per-launch latency into
//   the measurement, and would defeat any later stream overlap. Timing comes
//   from cudaEvent records bracketing the measured region, with exactly one
//   synchronize at the end of it.
//
// Defaults to true in Debug builds (FRAMEFLOW_DEBUG_SYNC) and is forced true at
// runtime by --validate. The README states which mode published numbers used.
extern bool g_debug_sync;

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                 ": " + expr + " failed: " + cudaGetErrorName(err) +
                                 " -- " + cudaGetErrorString(err));
    }
}

// Checks a kernel launch. Always cheap; synchronizes only when g_debug_sync.
inline void cuda_check_kernel(const char* name, const char* file, int line) {
    // Catches launch-configuration faults (bad grid/block, too much shared
    // memory). Host-side and essentially free -- always worth doing.
    cuda_check(cudaGetLastError(), name, file, line);
    if (g_debug_sync) {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize", file, line);
    }
}

}  // namespace frameflow

#define CUDA_CHECK(expr) ::frameflow::cuda_check((expr), #expr, __FILE__, __LINE__)

#define CUDA_CHECK_KERNEL(name) ::frameflow::cuda_check_kernel((name), __FILE__, __LINE__)
