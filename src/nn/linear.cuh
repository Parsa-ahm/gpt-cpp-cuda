#pragma once
#include <cuda_runtime.h>

#include "core/gemm_cublas.cuh"
#include "core/cuda_check.hpp"


__global__ void bias_fwd(float* out, const float* b, int N, int OC){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * OC) return;
    out[i] += b[i % OC];
}

inline void launch_linear_fwd(
    const float* x,
    const float* W, 
    const float* b, 
    float* out,
    int N, int C, int OC 
){
    const int threads = 256;
    const int blocks = ((N * OC + threads - 1) / threads);
    sgemm_rm(false, false, N, OC, C, 1.0f, x, W, 0.0f, out);
    bias_fwd<<<blocks, threads>>>(out, b, N, OC);
    CUDA_CHECK_KERNEL();
}
__global__ void bias_bwd(const float* dy, float* db, int N, int OC){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= OC) return;
    float total = 0;
    for ( int n = 0; n < N; n ++){
        total += dy[n * OC + i];
    }
    db[i] += total;

}

inline void launch_linear_bwd(
    const float* x,
    const float* W, 
    const float* dy, 
    float* dx, 
    float* dW, 
    float* db, 
    int N, int C, int OC 
){
    const int threads = 256;
    const int blocks = ((OC + threads - 1) / threads);
    sgemm_rm(false, true, N, C, OC, 1.0f, dy, W, 0.0f, dx);
    sgemm_rm(true, false, C, OC, N, 1.0f, x, dy, 1.0f, dW);

    bias_bwd<<<blocks, threads>>>(dy, db, N, OC);
    CUDA_CHECK_KERNEL();
}