#pragma once
#include <cuda_runtime.h>
#include <cuda_check.hpp>

__global__ void gelu_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float xi = x[i];
    float inner = 0.7978845608028654f * (xi + 0.044715f * xi * xi * xi);
    float t = tanhf(inner);
    y[i] = .5f * xi * (1.f + t);
}

inline void launch_gelu_fwd(const float* d_x, float* d_y, int n) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    gelu_fwd<<<blocks, threads>>>(d_x, d_y, n);

    CUDA_CHECK_KERNEL();
}

__global__ void gelu_bwd(const float* x, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float c = 0.044715f;
    const float s = 0.7978845608028654f;
    const float xi = x[i];
    float u = s * (xi + c * xi * xi * xi);
    float t = tanhf(u);
    float gelu_d =
        0.5f * (1.0f + t) + 0.5f * xi * (1.0f - (t * t)) * s * (1.0f + 3.0f * c * xi * xi);
    dx[i] = dy[i] * gelu_d;
}

inline void launch_gelu_bwd(const float* d_x, const float* d_dy, float* d_dx, int n) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    gelu_bwd<<<blocks, threads>>>(d_x, d_dy, d_dx, n);

    CUDA_CHECK_KERNEL();
}