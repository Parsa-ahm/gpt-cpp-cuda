#pragma once
#include <cuda_runtime.h>
#include "cuda_check.hpp"

__global__ void embedding_fwd(
    const int* ids, 
    const float* wte, 
    const float* wpe, 
    float* out, 
    int B, int T, int C){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B*T*C) return;
    int b = i / (T * C);
    int t = (i / C) % T;
    int c = i % C;
    int id = ids[b * T + t];
    out[i] = wte[id*C + c] + wpe[t*C + c];

}
inline void launch_embedding_fwd(const int* d_ids, const float* d_wte, const float* d_wpe, float* d_out, int d_B, int d_T, int d_C) {
    const int threads = 256;
    const int blocks = (d_B*d_T*d_C + threads - 1) / threads;

    embedding_fwd<<<blocks, threads>>>(d_ids, d_wte, d_wpe, d_out, d_B, d_T, d_C);

    CUDA_CHECK_KERNEL();
}

__global__ void embedding_bwd_wte(
    const int* ids, 
    const float* dout,
    float* dwte, 
    int B, int T, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= B*T*C) return;

    int b = i / (T*C);
    int t = (i / C) % T;
    int c = i % C;
    int id = ids[b*T + t];
    
    atomicAdd(&dwte[id*C + c], dout[i]);
}

__global__ void embedding_bwd_wpe(
    const float* dout,
    float* dwpe, 
    int B, int T, int C) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= T*C) return;

    int t = j / C;
    int c = j % C;
    float acc = 0;

    for (int b = 0; b < B; b++)
        acc += dout[b*T*C + t*C + c];

    dwpe[t*C + c] += acc;
}

inline void launch_embedding_bwd(
    const int* ids, 
    const float* dout,
    float* dwte, 
    float* dwpe,
    int B, int T, int C) {

    const int threads = 256;
    embedding_bwd_wte<<<(B*T*C+threads-1)/threads, threads>>>(ids, dout, dwte, B, T, C);
    CUDA_CHECK_KERNEL();

    embedding_bwd_wpe<<<(T*C+threads-1)/threads, threads>>>(dout, dwpe, B, T, C);
    CUDA_CHECK_KERNEL();
}