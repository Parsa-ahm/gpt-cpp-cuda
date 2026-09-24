#pragma once
#include <cuda_runtime.h>
#include "cuda_check.hpp"

__global__ void residual_fwd(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    out[i] = a[i] + b[i];
}

inline void launch_residual_fwd(const float* d_a, const float* d_b, float* d_out, int d_n) {
    const int threads = 256;
    const int blocks = (d_n + threads - 1) / threads;
    residual_fwd<<<blocks, threads>>>(d_a, d_b, d_out, d_n);
    CUDA_CHECK_KERNEL();
}

__global__ void residual_bwd(const float* dy, float* d_a, float* d_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    d_a[i] += dy[i];
    d_b[i] += dy[i];
}

inline void launch_residual_bwd(const float* d_dy, float* d_a, float* d_b, int d_n) {
    const int threads = 256;
    const int blocks = (d_n + threads - 1) / threads;
    residual_bwd<<<blocks, threads>>>(d_dy, d_a, d_b, d_n);
    CUDA_CHECK_KERNEL();
}