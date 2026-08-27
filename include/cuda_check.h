#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace frameflow {

// Whether a device synchronize follows every kernel launch.
//
// Launches are asynchronous, so without a sync an error surfaces at whatever
// unrelated API call happens to synchronize next. Debugging wants the sync;
// benchmarking does not, since it serializes the pipeline and folds per-launch
// latency into the measurement. Timing instead uses cudaEvent records with one
// synchronize at the end of the measured region.
//
// Defaults true in Debug builds (FRAMEFLOW_DEBUG_SYNC), and --validate forces
// it on at runtime.
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
    // Catches launch-configuration faults. Host-side and essentially free.
    cuda_check(cudaGetLastError(), name, file, line);
    if (g_debug_sync) {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize", file, line);
    }
}

}  // namespace frameflow

#define CUDA_CHECK(expr) ::frameflow::cuda_check((expr), #expr, __FILE__, __LINE__)

#define CUDA_CHECK_KERNEL(name) ::frameflow::cuda_check_kernel((name), __FILE__, __LINE__)
