// ============================================================================
// Distribution harness — validates the converters statistically.
//
//   (1) uniformity : next_float() over ~2M draws is flat in [0,1).
//                    range check + per-bin count within tolerance (chi-ish).
//   (2) normality  : next_n() over ~2M draws is standard normal:
//                    mean ~= 0, var ~= 1, and the 1/2/3-sigma mass matches
//                    68.27 / 95.45 / 99.73 %.
//   (3) cached branch: split draws into even (cos) and odd (sin=cached).
//                    BOTH subsets must independently be N(0,1) — proves the
//                    cached value is a valid, independent normal, not a dup.
//   (4) device finite: next_n() on the GPU produces no NaN/Inf.
//
// Stats accumulate in double (test-side reference math only; the model stays
// float32). Exit 0 = pass.
// ============================================================================
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "rng/philox.hpp"

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            return 99;                                                                             \
        }                                                                                          \
    } while (0)

static PhiloxStream make_stream(uint32_t element, uint32_t iter, uint32_t stream) {
    PhiloxStream s;
    uint32_t ctr[4] = {element, iter, stream, 0};
    uint32_t seed[2] = {0xC0FFEE01, 0xDEADBEEF};
    s.init(ctr, seed);
    return s;
}

__global__ void normal_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    PhiloxStream s;
    uint32_t ctr[4] = {(uint32_t)idx, 0u, 9u, 0u};
    uint32_t seed[2] = {0xC0FFEE01, 0xDEADBEEF};
    s.init(ctr, seed);
    out[idx] = s.next_n();
}

int main() {
    int failures = 0;
    const int N = 2000000;

    // ---------- (1) uniformity ----------
    std::printf("[uniformity]  next_float over %d draws\n", N);
    {
        const int BINS = 64;
        long counts[BINS] = {0};
        bool in_range = true;
        PhiloxStream s = make_stream(1, 1, 0);
        for (int i = 0; i < N; ++i) {
            float u = s.next_float();
            if (u < 0.0f || u >= 1.0f)
                in_range = false;
            int b = (int)(u * BINS);
            if (b < 0)
                b = 0;
            if (b >= BINS)
                b = BINS - 1;
            counts[b]++;
        }
        double expect = (double)N / BINS;
        double max_dev = 0.0;
        for (int b = 0; b < BINS; ++b) {
            double dev = std::fabs(counts[b] - expect) / expect;
            if (dev > max_dev)
                max_dev = dev;
        }
        bool ok = in_range && max_dev < 0.05;  // ~5% band; per-bin noise ~0.6%
        std::printf("  range [0,1): %s   max bin deviation: %.3f%%  -> %s\n",
                    in_range ? "ok" : "OUT OF RANGE", max_dev * 100.0, ok ? "PASS" : "FAIL");
        if (!ok)
            ++failures;
    }

    // ---------- (2)+(3) normality, all + even(cos)/odd(sin cached) ----------
    std::printf("[normality]  next_n over %d draws\n", N);
    {
        double sum = 0, sumsq = 0;
        double sum_e = 0, sumsq_e = 0;  // even index = cos branch
        double sum_o = 0, sumsq_o = 0;  // odd index  = cached sin branch
        long n_e = 0, n_o = 0;
        long w1 = 0, w2 = 0, w3 = 0;  // counts within 1/2/3 sigma
        bool finite = true;
        PhiloxStream s = make_stream(2, 2, 0);
        for (int i = 0; i < N; ++i) {
            float z = s.next_n();
            if (!std::isfinite(z))
                finite = false;
            sum += z;
            sumsq += (double)z * z;
            double az = std::fabs(z);
            if (az < 1.0)
                ++w1;
            if (az < 2.0)
                ++w2;
            if (az < 3.0)
                ++w3;
            if ((i & 1) == 0) {
                sum_e += z;
                sumsq_e += (double)z * z;
                ++n_e;
            } else {
                sum_o += z;
                sumsq_o += (double)z * z;
                ++n_o;
            }
        }
        double mean = sum / N;
        double var = sumsq / N - mean * mean;
        double mean_e = sum_e / n_e, var_e = sumsq_e / n_e - mean_e * mean_e;
        double mean_o = sum_o / n_o, var_o = sumsq_o / n_o - mean_o * mean_o;
        double p1 = 100.0 * w1 / N, p2 = 100.0 * w2 / N, p3 = 100.0 * w3 / N;

        bool ok = finite && std::fabs(mean) < 0.01 && std::fabs(var - 1.0) < 0.02 &&
                  std::fabs(p1 - 68.27) < 0.5 && std::fabs(p2 - 95.45) < 0.5 &&
                  std::fabs(p3 - 99.73) < 0.3;
        std::printf("  finite: %s   mean %.4f (want 0)   var %.4f (want 1)\n",
                    finite ? "ok" : "NaN/Inf", mean, var);
        std::printf("  sigma mass: 1s %.2f%% (68.27)  2s %.2f%% (95.45)  3s %.2f%% (99.73)  -> %s\n",
                    p1, p2, p3, ok ? "PASS" : "FAIL");
        if (!ok)
            ++failures;

        bool ok_e = std::fabs(mean_e) < 0.02 && std::fabs(var_e - 1.0) < 0.03;
        bool ok_o = std::fabs(mean_o) < 0.02 && std::fabs(var_o - 1.0) < 0.03;
        std::printf("  cos branch: mean %.4f var %.4f -> %s\n", mean_e, var_e,
                    ok_e ? "PASS" : "FAIL");
        std::printf("  sin branch (cached): mean %.4f var %.4f -> %s\n", mean_o, var_o,
                    ok_o ? "PASS" : "FAIL");
        if (!ok_e)
            ++failures;
        if (!ok_o)
            ++failures;
    }

    // ---------- (4) device finite ----------
    std::printf("[device finite]  next_n on GPU\n");
    {
        const int M = 4096;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * M));
        normal_kernel<<<(M + 127) / 128, 128>>>(d_out, M);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float* h = new float[M];
        CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(float) * M, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < M; ++i)
            if (!std::isfinite(h[i]))
                ok = false;
        std::printf("  all %d device normals finite: %s\n", M, ok ? "PASS" : "FAIL");
        if (!ok)
            ++failures;
        delete[] h;
        CUDA_CHECK(cudaFree(d_out));
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
