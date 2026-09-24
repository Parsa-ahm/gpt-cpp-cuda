#pragma once

#include <cuda_runtime.h>

__global__ void gemm(const float* a, const float* b, float* c, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

inline void launch_gemm(const float* dA, const float* dB, float* dC, int M, int K, int N) {
    dim3 threads = {16, 16};
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    gemm<<<blocks, threads>>>(dA, dB, dC, M, K, N);
}

__global__ void gemm_tiled(const float* a, const float* b, float* c, int M, int K, int N) {
    constexpr int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;
    float sum = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + tx;
        As[ty][tx] = (row < M && a_col < K) ? a[row * K + a_col] : 0.0f;

        int b_row = t + ty;
        Bs[ty][tx] = (col < N && b_row < K) ? b[b_row * N + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}
inline void launch_gemm_tiled(const float* dA, const float* dB, float* dC, int M, int K, int N) {
    dim3 threads = {16, 16};
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    gemm_tiled<<<blocks, threads>>>(dA, dB, dC, M, K, N);
}

__global__ void gemm_reg(const float* a, const float* b, float* c, int M, int K, int N) {
    constexpr int Bn = 64;
    constexpr int Bm = 64;
    constexpr int Bk = 8;
    __shared__ float As[Bn][Bk];
    __shared__ float Bs[Bk][Bm];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * 64;
    int block_col = blockIdx.x * 64;

    int row = block_row + ty * 8;
    int col = block_col + tx * 8;

    float acc[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0;
        }
    }

    int tid = ty * 8 + tx;
    for (int k0 = 0; k0 < K; k0 += Bk) {
        for (int i = 0; i < 8; i++) {
            int e = tid + 64 * i;
            int r = e / Bk;
            int c = e % Bk;
            As[r][c] = a[(block_row + r) * K + (k0 + c)];

            r = e / Bn;
            c = e % Bn;
            Bs[r][c] = b[(k0 + r) * N + (block_col + c)];
        }
        __syncthreads();
        float a_reg[8];
        float b_reg[8];
        for (int kk = 0; kk < Bk; kk++) {
            for (int i = 0; i < 8; i++) {
                a_reg[i] = As[ty * 8 + i][kk];
                b_reg[i] = Bs[kk][tx * 8 + i];
            }
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            c[(row + i) * N + col + j] = acc[i][j];
        }
    }
}
inline void launch_gemm_reg(const float* dA, const float* dB, float* dC, int M, int K, int N) {
    dim3 threads = {8, 8};
    dim3 blocks((N + 63) / 64, (M + 63) / 64);
    gemm_reg<<<blocks, threads>>>(dA, dB, dC, M, K, N);
}