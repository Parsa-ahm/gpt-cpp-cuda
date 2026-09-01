// ============================================================================
// Correctness harness for the naive SGEMM.
//
//   For several shapes (square, non-square, non-multiples of 16):
//     - fill A (MxK) and B (KxN) with random floats
//     - compute C on the GPU via launch_gemm
//     - compute C_ref on the CPU with a plain triple loop (the oracle)
//     - compare within a float tolerance (matmul accumulates rounding, so we
//       check "close", not bit-exact)
//
// The non-multiple-of-16 shapes deliberately make the grid overshoot, so they
// exercise the  if (row < M && col < N)  bounds check.
//
// Exit 0 = all shapes pass.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "core/gemm.cuh"

// CPU oracle: C = A * B, row-major. Plain and obviously-correct on purpose.
static void cpu_gemm(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = sum;
        }
}

using LaunchFn = void (*)(const float*, const float*, float*, int, int, int);

// run one shape on the GPU, compare to the oracle; returns true on pass
static bool test_shape(int M, int K, int N, LaunchFn launch, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(M * K), B(K * N), C(M * N), ref(M * N);
    for (float& x : A) x = dist(rng);
    for (float& x : B) x = dist(rng);

    // GPU
    Device_Buffer dA(M * K), dB(K * N), dC(M * N);
    dA.upload(A.data());
    dB.upload(B.data());
    launch(dA.ptr, dB.ptr, dC.ptr, M, K, N);
    dC.download(C.data());
    dA.free_it();
    dB.free_it();
    dC.free_it();

    // CPU oracle
    cpu_gemm(A.data(), B.data(), ref.data(), M, K, N);

    // compare
    float max_abs = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float d = std::fabs(C[i] - ref[i]);
        if (d > max_abs) max_abs = d;
    }
    bool ok = max_abs < 1e-3f;
    std::printf("  M=%-4d K=%-4d N=%-4d  max|err|=%.2e  %s\n", M, K, N, max_abs,
                ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    std::mt19937 rng(12345);
    int failures = 0;

    struct S { int M, K, N; };

    std::printf("[naive gemm correctness]\n");
    S shapes[] = {
        {2, 3, 2},      // tiny (exact-ish)
        {64, 64, 64},   // square, multiple of 16
        {128, 96, 64},  // rectangular, multiples of 16
        {37, 53, 41},   // none divisible by 16 -> exercises bounds check
        {1, 1, 1},      // degenerate
        {100, 1, 100},  // K=1
        {17, 200, 3},   // skinny output
    };
    for (S s : shapes)
        if (!test_shape(s.M, s.K, s.N, launch_gemm, rng)) ++failures;

    // Tiled kernel: now handles ragged (non-multiple-of-16) shapes too.
    std::printf("[tiled gemm correctness]  (incl. ragged edges)\n");
    S tiled_shapes[] = {
        {2, 3, 2},      {64, 64, 64},   {128, 96, 64}, {37, 53, 41},  {1, 1, 1},
        {100, 1, 100},  {17, 200, 3},   {256, 256, 256}, {513, 511, 257}, {500, 500, 500},
    };
    for (S s : tiled_shapes)
        if (!test_shape(s.M, s.K, s.N, launch_gemm_tiled, rng)) ++failures;

    // Register-blocked kernel: no bounds guards yet, so only divisible shapes
    // (M,N multiples of 64; K a multiple of 8).
    std::printf("[reg-blocked gemm correctness]  (divisible shapes only)\n");
    S reg_shapes[] = {
        {64, 8, 64},     {64, 64, 64},   {128, 128, 128},
        {256, 512, 128}, {256, 256, 256}, {512, 512, 512},
    };
    for (S s : reg_shapes)
        if (!test_shape(s.M, s.K, s.N, launch_gemm_reg, rng)) ++failures;

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
