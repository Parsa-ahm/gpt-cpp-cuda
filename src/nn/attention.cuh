#pragma once
#include <cuda_runtime.h>

#include "core/gemm_cublas.cuh"
#include "core/cuda_check.hpp"


//Helper functions

__device__ inline float dot(const float* a, const float* b, int n){
    float acc = 0.0f;
    for(int j = 0; j < n; j++){
        acc += a[j] * b[j];
    }
    return acc;
}

__global__ void attention_fwd(
    const float* qkv, // query then key then value. its a B, T, 3C as the 3 matrecies are appended one after each other. 
    float* att,  // (B, NH, T, T) for each batch and each head of size T , T, each row is how much does this token care about each earlier token, this is normalized with 0s for all future tokens,  
    float* out,  // B, T, C the product of the 
    int B,      // batch / sequence
    int T,      // Time (Tokens)
    int C,      // Chanells = NH * HS
    int NH      // Number of heads
){

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= B * NH * T) return;
        int t = i % T;          //
        int h = (i / T) % NH;
        int b = i / (NH *T);
        int HS = C / NH;       // Head size
        float scale = 1.0f / sqrtf((float)HS);
        const float* q = qkv + b*T*3*C + t*3*C + h*HS + 0*C; // points at hs floats its the querry 
        float* attrow = att + b*NH*T*T + h*T*T + t*T; // points at T gloats 
        float* outp = out + b*T*C + t*C + h*HS; // points at HS floats
        float maxv = -INFINITY;
        for (int t2 = 0; t2 <= t; t2++){
            const float* k = qkv + b*T*3*C + t2*3*C + h*HS + 1*C; // same as q but t2 and 1*C
            float score = scale * dot(q, k, HS);
            maxv = fmaxf(maxv, score);
            attrow[t2] = score;
        }
        float sum = 0.0f;
        for(int t2 = 0; t2 <= t; t2++){
            float e = expf(attrow[t2] - maxv);
            attrow[t2] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for ( int t2 = 0; t2 <= t; t2++){
            attrow[t2] *= inv; 
        }

        for ( int t2 = t+1; t2 < T; t2++){
            attrow[t2] = 0.0f; 
        }
        for (int j = 0; j < HS; j++){
            float acc = 0.0f;
            for (int t2 = 0; t2 <= t; t2++){
                const float* v = qkv + b*T*3*C + t2*3*C + h*HS + 2*C;
                acc += attrow[t2] * v[j];
            }
            outp[j] = acc;
        }


    }
__global__ void attention_bwd_datt_dv(
    const float* qkv, 
    const float* att, 
    const float* dout, 
    float* datt, 
    float* dqkv, 
    int B, int T, int C, int NH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * T) return;
    int t2 = idx % T;
    int h = (idx / T) % NH;
    int b = idx / (NH * T);
    int HS = C / NH;

    const float* v = qkv  + b*T*3*C + t2*3*C + h*HS + 2*C;
    float* dvp = dqkv + b*T*3*C + t2*3*C + h*HS + 2*C;
    int colbase = b*NH*T*T + h*T*T + t2;

    for (int t = t2; t < T; t++) {
        const float* dO = dout + b*T*C + t*C + h*HS;
        datt[colbase + t*T] = dot(dO, v, HS);
    }
    for (int t = 0; t < t2; t++) datt[colbase + t*T] = 0.0f;

    for (int i = 0; i < HS; i++) {
        float acc = 0.0f;
        for (int t = t2; t < T; t++) {
            const float* dO = dout + b*T*C + t*C + h*HS;
            acc += att[colbase + t*T] * dO[i];
        }
        dvp[i] = acc;
    }
}

__global__ void attention_bwd_softmax(
    const float* att, 
    const float* datt,
    float* dpreatt, 
    int B, int T, int C, int NH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * T) return;
    int t = idx % T;
    int h = (idx / T) % NH;
    int b = idx / (NH * T);

    int rowbase = b*NH*T*T + h*T*T + t*T;

    float rowdot = 0.0f;
    for (int t3 = 0; t3 <= t; t3++)
        rowdot += att[rowbase + t3] * datt[rowbase + t3];

    for (int t2 = 0; t2 <= t; t2++)
        dpreatt[rowbase + t2] = att[rowbase + t2] * (datt[rowbase + t2] - rowdot);

    for (int t2 = t + 1; t2 < T; t2++)
        dpreatt[rowbase + t2] = 0.0f;
}

__global__ void attention_bwd_dq_dk(
    const float* qkv, 
    const float* dpreatt,
    float* dqkv, 
    int B, int T, int C, int NH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * T) return;
    int x = idx % T;
    int h = (idx / T) % NH;
    int b = idx / (NH * T);
    int HS = C / NH;
    float scale = 1.0f / sqrtf((float)HS);

    float* dqp = dqkv + b*T*3*C + x*3*C + h*HS + 0*C;
    float* dkp = dqkv + b*T*3*C + x*3*C + h*HS + 1*C;
    int hbase = b*NH*T*T + h*T*T;

    for (int i = 0; i < HS; i++) {
        float acc = 0.0f;
        for (int t2 = 0; t2 <= x; t2++) {
            const float* k = qkv + b*T*3*C + t2*3*C + h*HS + 1*C;
            acc += dpreatt[hbase + x*T + t2] * k[i];
        }
        dqp[i] = scale * acc;
    }

    for (int i = 0; i < HS; i++) {
        float acc = 0.0f;
        for (int t = x; t < T; t++) {
            const float* q = qkv + b*T*3*C + t*3*C + h*HS + 0*C;
            acc += dpreatt[hbase + t*T + x] * q[i];
        }
        dkp[i] = scale * acc;
    }
}


inline void launch_attention_fwd(
    const float* qkv,
    float* att,  
    float* out,  // B, T, C the product of the 
    int B,      // batch / sequence
    int T,      // Time (Tokens)
    int C,      // Chanells = NH * HS
    int NH      // Number of heads
){
    int threads = 256;
    int blocks = (B * NH * T + threads - 1) / threads;

    attention_fwd<<<blocks, threads>>>(qkv, att, out, B, T, C, NH);
    CUDA_CHECK_KERNEL();
}



inline void launch_attention_bwd(
    const float* qkv,  
    const float* att,
    const float* dout,
    float* datt,  
    float* dpreatt,
    float* dqkv,  // B, T, C the product of the 
    int B,      // batch / sequence
    int T,      // Time (Tokens)
    int C,      // Chanells = NH * HS
    int NH      // Number of heads
){
    int threads = 256;
    int blocks = (B * NH * T + threads - 1) / threads;

    attention_bwd_datt_dv<<<blocks, threads>>>(qkv, att, dout, datt, dqkv, B, T, C, NH);
    CUDA_CHECK_KERNEL();
    attention_bwd_softmax<<<blocks, threads>>>(att, datt, dpreatt, B, T, C, NH);
    CUDA_CHECK_KERNEL();
    attention_bwd_dq_dk<<<blocks, threads>>>(qkv, dpreatt, dqkv, B, T, C, NH);
    CUDA_CHECK_KERNEL();
}