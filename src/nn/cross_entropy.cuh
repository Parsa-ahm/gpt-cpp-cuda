#pragma once
#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

__global__ void crossentropy_fwd(
    const float* logits,  
    const int* targets,  // Tokens
    float* losses,          
    float* lse, 
    int N, int V)  {

        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N) return;
        int y = targets[n];
        int base = n * V;
        float m = -INFINITY;    // largest logits val
        for (int v = 0; v < V; v++){
            m = fmaxf(m, logits[base+v]);
        }
        float s = 0;            // Exp sum for softmaxing
        for (int v = 0; v < V; v++){
            s += expf(logits[base + v] - m);
        }
        lse[n] = m + logf(s);
        losses[n] = lse[n] - logits[base+y];
}   

__global__ void crossentropy_bwd(
    const float* logits, 
    const float* lse, 
    const int* targets, 
    float* dlogits, 
    float dloss,  
    int N, int V){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N*V) return;
        int n = i / V;
        int v = i % V;
        float p = expf(logits[i] - lse[n]);
        float indicator = (v == targets[n]) ? 1.0f : 0.0f;
        dlogits[i] = (p - indicator) * dloss;
}

inline void launch_crossentropy_fwd(
    const float* logits, 
    const int* targets, 
    float* losses, 
    float* lse, 
    int N, int V){
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        crossentropy_fwd<<<blocks, threads>>>(logits, targets, losses, lse, N, V);
        CUDA_CHECK_KERNEL();
    }

inline void launch_crossentropy_bwd(
    const float* logits, 
    const float* lse, 
    const int* targets, 
    float* dlogits, 
    float dloss,  
    int N, int V){
        int threads = 256;
        int blocks = (N * V + threads - 1) / threads;
        crossentropy_bwd<<<blocks, threads>>>(logits, lse, targets, dlogits, dloss, N, V);
        CUDA_CHECK_KERNEL();
}