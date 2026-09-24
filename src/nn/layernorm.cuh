#pragma once
#include <cuda_runtime.h>
#include "cuda_check.hpp"

__global__ void layernorm_fwd(
    const float* x, // input (N, C)
    const float* w, // gain (C)
    const float* b, // bias (C)
    float* out,     // output (N, C)
    float* mean,    // cached (C)
    float* rstd,    // cached (C)
    int N, int C
){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float eps = 1e-5f;

    float total = 0;
    for (int c = 0; c < C; c++){
        total += x[n * C + c];
    }
    mean[n] = total / C;
    float var = 0;
    for (int c = 0; c < C; c++){
        float d = x[n * C + c] - mean[n];
        var += d * d ;
    } 
    var = var / C;
    rstd[n] = 1.0f / sqrtf(var + eps);

    for (int c = 0; c < C; c++){
        float xhat = (x[n * C + c] - mean[n]) * rstd[n];
        out[n * C + c] = xhat * w[c] + b[c];
    } 
}
inline void launch_layernorm_fwd(    
    const float* d_x, // input (N, C)
    const float* d_w, // gain (C)
    const float* d_b, // bias (C)
    float* d_out,     // output (N, C)
    float* d_mean,    // cached (C)
    float* d_rstd,    // cached (C)
    int d_N, int d_C) {
    const int threads = 256;
    const int blocks = (d_N + threads - 1) / threads;

    layernorm_fwd<<<blocks, threads>>>(d_x, d_w, d_b, d_out, d_mean, d_rstd, d_N, d_C);

    CUDA_CHECK_KERNEL();
}


__global__ void layernorm_bwd_dx(
    const float* dout, // upstream grad (N, C)
    const float* x, // original input (N, C)
    const float* w, // gain (C)
    const float* mean, // cached mean (N)
    const float* rstd, // cached rstd (N)
    float* dx,         // output grad (N, C)
    int N, int C
){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float s1 = 0, s2 = 0;
    for (int c = 0; c < C; c ++){
        float xhat = (x[n * C + c] - mean[n]) * rstd[n];
        float dxhat = dout[n * C + c] * w[c];
        s1 += dxhat;
        s2 += dxhat * xhat;
    } 
    s1 /= C;
    s2 /= C;
    for (int c = 0; c < C; c ++){
        float xhat = (x[n * C + c] - mean[n]) * rstd[n];
        float dxhat = dout[n * C + c] * w[c];
        dx[n * C + c] = rstd[n] * (dxhat - s1 - xhat * s2);
    }
}
inline void launch_layernorm_bwd_dx(    
    const float* d_dout, 
    const float* d_x, 
    const float* d_w, 
    const float* d_mean,    
    const float* d_rstd, 
    float* d_dx,   
    int d_N, int d_C) {
    const int threads = 256;
    const int blocks = (d_N + threads - 1) / threads;

    layernorm_bwd_dx<<<blocks, threads>>>(d_dout, d_x, d_w, d_mean, d_rstd, d_dx, d_N, d_C);

    CUDA_CHECK_KERNEL();
}

__global__ void layernorm_bwd_dwdb(
    const float* dout, // upstream grad (N, C)
    const float* x, // original input (N, C)
    const float* mean, // cached mean (N)
    const float* rstd, // cached rstd (N)
    float* dw,         // output grad (C)
    float* db,         // output grad (C)
    int N, int C
){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float adw = 0, adb = 0;
    for (int n = 0; n < N; n++){
        float xhat = (x[n * C + c] - mean[n]) * rstd[n];
        adb += dout[n * C + c];
        adw += dout[n * C + c] * xhat;
    }
    db[c] += adb;
    dw[c] += adw;
    
}
inline void launch_layernorm_bwd_dwdb(    
    const float* d_dout, 
    const float* d_x,    
    const float* d_mean, 
    const float* d_rstd, 
    float* d_dw,    
    float* d_db,           
    int d_N, int d_C) {
    const int threads = 256;
    const int blocks = (d_C + threads - 1) / threads;

    layernorm_bwd_dwdb<<<blocks, threads>>>(d_dout, d_x, d_mean, d_rstd, d_dw, d_db, d_N, d_C);

    CUDA_CHECK_KERNEL();
}

inline void launch_layernorm_bwd(
    const float* dout,
    const float* x,
    const float* w,
    const float* mean,
    const float* rstd,
    float* dx,
    float* dw,
    float* db,
    int N, int C)
    {
        launch_layernorm_bwd_dx(dout, x, w, mean, rstd, dx, N, C);
        launch_layernorm_bwd_dwdb(dout, x, mean, rstd, dw, db, N, C);
    }
    