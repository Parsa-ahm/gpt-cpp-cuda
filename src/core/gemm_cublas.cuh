#pragma once
#include <cublas_v2.h>

#include "core/cublas_ctx.hpp"

inline void sgemm_rm(bool trans_a, bool trans_b, int M, int N, int K, float alpha, const float* dA,
                     const float* dB, float beta, float* dC) {
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = trans_a ? M : K;
    int ldb = trans_b ? K : N;

    CUBLAS_CHECK(
        cublasSgemm(cublas_handle(), op_b, op_a, N, M, K, &alpha, dB, ldb, dA, lda, &beta, dC, N));
}