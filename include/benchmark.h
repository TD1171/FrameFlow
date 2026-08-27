#pragma once

#include "pipeline.h"

#include <string>

namespace frameflow {

struct BenchmarkOptions {
    std::string input;
    std::string csv_path = "results/benchmarks.csv";
    int ksize = 5;
    double sigma = 1.4;
    int frames = 20;      // distinct frames per configuration
    int repeats = 3;      // passes over those frames
    int warmup = 30;      // untimed iterations before each configuration
};

// Sweeps every kernel variant across a set of resolutions, writes a CSV, and
// prints a summary table.
//
// Returns EXIT_SUCCESS, or EXIT_FAILURE if no configuration completed.
int run_benchmark(const BenchmarkOptions& opt);

}  // namespace frameflow
