// ============================================================================
// Row-major wrapper over cuBLAS: the dedicated layout test.
//
// Charter risk R2 is "cuBLAS layout confusion", mitigated by "one dedicated
// test with the oracle before anything depends on it". This is that test.
//
// Covers all four transpose combinations, several shapes, and beta=1
// accumulation. A wrong transpose does not produce garbage, it produces a
// plausible matrix of wrong numbers, so the oracle comparison is the only way
// to know. Keeping it separate from test_linear means a failure here points at
// the layout and a failure there points at the calculus.
//
// Exit 0 = all pass.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "core/gemm_cublas.cuh"

// Row-major CPU oracle for C = alpha * opA(A) @ opB(B) + beta * C.
// A is (M,K) when !ta else (K,M);  B is (K,N) when !tb else (N,K).
static void cpu_gemm_rm(bool ta, bool tb, int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = ta ? A[k * M + m] : A[m * K + k];
                float b = tb ? B[n * K + k] : B[k * N + n];
                sum += a * b;
            }
            C[m * N + n] = alpha * sum + beta * C[m * N + n];
        }
}

static bool run(bool ta, bool tb, int M, int N, int K, float beta, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const float alpha = 1.0f;

    std::vector<float> A(M * K), B(K * N), C(M * N), ref(M * N);
    for (float& v : A) v = dist(rng);
    for (float& v : B) v = dist(rng);
    for (float& v : C) v = dist(rng);  // matters when beta != 0
    ref = C;

    Device_Buffer dA(M * K), dB(K * N), dC(M * N);
    dA.upload(A.data());
    dB.upload(B.data());
    dC.upload(C.data());
    sgemm_rm(ta, tb, M, N, K, alpha, dA.ptr, dB.ptr, beta, dC.ptr);
    dC.download(C.data());
    dA.free_it();
    dB.free_it();
    dC.free_it();

    cpu_gemm_rm(ta, tb, M, N, K, alpha, A.data(), B.data(), beta, ref.data());

    float worst = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float d = std::fabs(C[i] - ref[i]);
        if (d > worst) worst = d;
    }
    bool ok = worst < 1e-3f;
    std::printf("  A%s B%s  M=%-4d N=%-4d K=%-4d beta=%.0f  max|err|=%.2e  %s\n",
                ta ? "^T" : "  ", tb ? "^T" : "  ", M, N, K, beta, worst, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    std::mt19937 rng(777);
    int failures = 0;

    struct S { int M, N, K; };
    S shapes[] = {
        {2, 3, 4},       // tiny, easy to hand-check
        {64, 64, 64},    // square
        {128, 96, 64},   // rectangular
        {37, 53, 41},    // all prime-ish, no nice divisibility
        {1, 64, 32},     // single row
        {64, 1, 32},     // single column
        {256, 384, 128}, // transformer-ish
    };

    std::printf("[row-major sgemm via cuBLAS, all transpose combinations]\n");
    for (S s : shapes) {
        if (!run(false, false, s.M, s.N, s.K, 0.0f, rng)) ++failures;
        if (!run(true, false, s.M, s.N, s.K, 0.0f, rng)) ++failures;
        if (!run(false, true, s.M, s.N, s.K, 0.0f, rng)) ++failures;
        if (!run(true, true, s.M, s.N, s.K, 0.0f, rng)) ++failures;
    }

    std::printf("[beta = 1: accumulate into existing C]\n");
    for (S s : shapes) {
        if (!run(false, false, s.M, s.N, s.K, 1.0f, rng)) ++failures;
        if (!run(true, false, s.M, s.N, s.K, 1.0f, rng)) ++failures;
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
