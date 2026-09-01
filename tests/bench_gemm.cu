// ============================================================================
// SGEMM benchmark: your kernel vs cuBLAS.
//
// Times both on square problems, reports GFLOP/s and % of cuBLAS.
// FLOPs for an MxNxK matmul = 2*M*N*K (one multiply + one add per inner step).
//
// cuBLAS is the perf REFERENCE only (charter: it never touches the model path).
// It is column-major; we call it with A/B swapped so its column-major result
// equals our row-major C = A*B. For timing the layout is irrelevant (same FLOPs),
// but doing it right lets the numbers mean something.
// ============================================================================
#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "core/device_buffer.hpp"
#include "core/gemm.cuh"

static float time_ms(void (*fn)(), int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    fn();  // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

// globals so the timing thunks can see them
static int gM, gK, gN;
static float *gA, *gB, *gC;
static cublasHandle_t gHandle;

static void run_mine() { launch_gemm(gA, gB, gC, gM, gK, gN); }
static void run_tiled() { launch_gemm_tiled(gA, gB, gC, gM, gK, gN); }
static void run_reg() { launch_gemm_reg(gA, gB, gC, gM, gK, gN); }
static void run_cublas() {
    const float alpha = 1.0f, beta = 0.0f;
    // row-major C = A*B  computed as column-major  C^T = B^T * A^T
    cublasSgemm(gHandle, CUBLAS_OP_N, CUBLAS_OP_N, gN, gM, gK, &alpha, gB, gN, gA, gK, &beta, gC,
                gN);
}

static void bench(int n) {
    gM = gK = gN = n;
    std::vector<float> A(n * n, 1.0f), B(n * n, 1.0f);
    Device_Buffer dA(n * n), dB(n * n), dC(n * n);
    dA.upload(A.data());
    dB.upload(B.data());
    gA = dA.ptr;
    gB = dB.ptr;
    gC = dC.ptr;

    int iters = n <= 512 ? 50 : 20;
    double flops = 2.0 * n * n * n;

    float mine = time_ms(run_mine, iters);
    float tiled = time_ms(run_tiled, iters);
    float reg = time_ms(run_reg, iters);
    float cub = time_ms(run_cublas, iters);
    double g_mine = flops / (mine / 1e3) / 1e9;
    double g_tiled = flops / (tiled / 1e3) / 1e9;
    double g_reg = flops / (reg / 1e3) / 1e9;
    double g_cub = flops / (cub / 1e3) / 1e9;

    std::printf(
        "  N=%-5d  naive %6.0f (%.0f%%)  tiled %6.0f (%.0f%%)  reg %6.0f (%.0f%%)  cuBLAS %6.0f\n",
        n, g_mine, 100.0 * g_mine / g_cub, g_tiled, 100.0 * g_tiled / g_cub, g_reg,
        100.0 * g_reg / g_cub, g_cub);

    dA.free_it();
    dB.free_it();
    dC.free_it();
}

int main() {
    cublasCreate(&gHandle);
    std::printf("[sgemm benchmark]  (RTX 2070 SUPER, sm_75)\n");
    for (int n : {256, 512, 1024, 2048}) bench(n);
    cublasDestroy(gHandle);
    return 0;
}
