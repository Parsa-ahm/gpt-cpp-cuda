#pragma once
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Build/toolchain sanity: reports the visible CUDA devices.
// Infra only (not part of the from-scratch ML core). Returns device count.
int report_cuda_devices();

// ---------------------------------------------------------------------------
// CUDA_CHECK(expr) — wrap any CUDA runtime call that returns cudaError_t.
//
//     CUDA_CHECK(cudaMalloc(&ptr, bytes));
//
// On failure it prints file, line, the call, and the error string, then exits.
// Bare CUDA calls with the return value discarded are the single most common
// way to spend an afternoon debugging a kernel that never ran.
//
// The do/while(0) wrapper is a C idiom, not decoration: it makes the macro one
// statement, so `if (x) CUDA_CHECK(...); else ...` parses the way you expect.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n  in: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(err_), #expr);                                    \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA_CHECK_KERNEL() — call immediately after a <<<>>> launch.
//
// A launch is asynchronous, so errors arrive in two places and you need both:
//   cudaGetLastError()      launch-config problems (bad block dim, too much
//                           shared memory). Available immediately.
//   cudaDeviceSynchronize() errors from execution itself (out-of-bounds write,
//                           illegal address). Only available once it has run.
//
// Synchronizing after every launch is a correctness-first choice and it does
// cost throughput. Fine for Rung 2 tests; the training loop in Rung 3 will drop
// the sync from the hot path and check at step boundaries instead.
// ---------------------------------------------------------------------------
#define CUDA_CHECK_KERNEL()                \
    do {                                   \
        CUDA_CHECK(cudaGetLastError());    \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
