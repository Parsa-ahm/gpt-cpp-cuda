#pragma once
#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

__global__ void adamw(
    float* p, // parameters 
    const float* g, // Gradients 
    float* m, // first moment
    float* v, // second moment
    int n,   // element count
    float lr, float beta1, float beta2, float eps, float wd, // hyper perams
    int t   // Step count
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n) return;

    m[i] = beta1 * m[i] + ( 1 - beta1) * g[i];
    v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];

    float mhat = m[i] / (1 - powf(beta1, t));
    float vhat = v[i] / (1 - powf(beta2, t));

    p[i] -= lr* ( mhat / (sqrtf(vhat) + eps) + wd * p[i]);

}

inline void launch_adamw_step(
    float* p, // parameters 
    const float* g, // Gradients 
    float* m, // first moment
    float* v, // second moment
    int n,   // element count
    float lr, float beta1, float beta2, float eps, float wd, // hyper perams
    int t   // Step count
)
{
    int threads = 256;
    int blocks = (n + threads - 1) /threads;

    adamw<<<blocks,  threads>>>(p, g, m, v, n, lr, beta1, beta2, eps, wd, t);
    CUDA_CHECK_KERNEL();
}