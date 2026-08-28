// ============================================================================
// PhiloxStream harness — validates the per-thread stream wrapper.
//
//   (1) continuity : 8 next_u32() calls == philox(block=0) ++ philox(block=1),
//                    in word order. Proves the refill/cursor/block logic.
//   (2) determinism: same (seed, element, iter, stream) -> identical sequence.
//   (3) independence: distinct coordinate tuples never produce the same
//                     sequence (no counter collisions -> independent streams).
//   (4) host==device: the SAME stream run inside a kernel matches the host run.
//
// Exit 0 = all pass; nonzero = failure (CTest goes red).
// ============================================================================
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

// draw n words from a fresh host stream at the given coordinates
static void host_stream(uint32_t element, uint32_t iter, uint32_t stream, uint32_t s0, uint32_t s1,
                        uint32_t* out, int n) {
    PhiloxStream s;
    uint32_t ctr[4] = {element, iter, stream, 0};
    uint32_t seed[2] = {s0, s1};
    s.init(ctr, seed);
    for (int i = 0; i < n; ++i)
        out[i] = s.next_u32();
}

// one thread per coordinate tuple; each draws n words into out[idx*n ...]
__global__ void stream_kernel(const uint32_t* elem, const uint32_t* it, const uint32_t* strm,
                              uint32_t s0, uint32_t s1, uint32_t* out, int ntup, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ntup)
        return;
    PhiloxStream s;
    uint32_t ctr[4] = {elem[idx], it[idx], strm[idx], 0};
    uint32_t seed[2] = {s0, s1};
    s.init(ctr, seed);
    for (int i = 0; i < n; ++i)
        out[idx * n + i] = s.next_u32();
}

int main() {
    int failures = 0;
    const uint32_t S0 = 0x1234abcd, S1 = 0x0badf00d;
    uint32_t seed[2] = {S0, S1};

    // ---------- (1) continuity ----------
    std::printf("[continuity]\n");
    {
        const uint32_t E = 7, IT = 3, ST = 1;
        uint32_t got[8];
        host_stream(E, IT, ST, S0, S1, got, 8);

        uint32_t c0[4] = {E, IT, ST, 0};
        uint32_t c1[4] = {E, IT, ST, 1};
        Philox4 b0 = philox(c0, seed);
        Philox4 b1 = philox(c1, seed);
        uint32_t expect[8] = {b0.w[0], b0.w[1], b0.w[2], b0.w[3],
                              b1.w[0], b1.w[1], b1.w[2], b1.w[3]};
        bool ok = true;
        for (int i = 0; i < 8; ++i)
            if (got[i] != expect[i])
                ok = false;
        std::printf("  8 draws == 2 philox blocks: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) {
            ++failures;
            for (int i = 0; i < 8; ++i)
                std::printf("    [%d] got 0x%08x  expect 0x%08x\n", i, got[i], expect[i]);
        }
    }

    // ---------- (2) determinism ----------
    std::printf("[determinism]\n");
    {
        uint32_t a[16], b[16];
        host_stream(42, 99, 2, S0, S1, a, 16);
        host_stream(42, 99, 2, S0, S1, b, 16);
        bool ok = true;
        for (int i = 0; i < 16; ++i)
            if (a[i] != b[i])
                ok = false;
        std::printf("  same coords -> same sequence: %s\n", ok ? "PASS" : "FAIL");
        if (!ok)
            ++failures;
    }

    // ---------- (3) independence ----------
    std::printf("[independence]\n");
    const int NT = 8, N = 8;
    uint32_t elem[NT] = {0, 1, 0, 0, 5, 5, 123, 0xffff};
    uint32_t it[NT] = {0, 0, 1, 0, 7, 7, 456, 0xffff};
    uint32_t strm[NT] = {0, 0, 0, 1, 2, 3, 1, 0};
    uint32_t seqs[NT][N];
    for (int t = 0; t < NT; ++t)
        host_stream(elem[t], it[t], strm[t], S0, S1, seqs[t], N);
    {
        bool ok = true;
        for (int a = 0; a < NT && ok; ++a)
            for (int b = a + 1; b < NT && ok; ++b) {
                bool same = true;
                for (int i = 0; i < N; ++i)
                    if (seqs[a][i] != seqs[b][i])
                        same = false;
                if (same) {
                    ok = false;
                    std::printf("    COLLISION between tuple %d and %d\n", a, b);
                }
            }
        std::printf("  distinct coords -> distinct streams: %s\n", ok ? "PASS" : "FAIL");
        if (!ok)
            ++failures;
    }

    // ---------- (4) host == device ----------
    std::printf("[host == device]\n");
    {
        uint32_t *d_e, *d_i, *d_s, *d_out;
        CUDA_CHECK(cudaMalloc(&d_e, sizeof(elem)));
        CUDA_CHECK(cudaMalloc(&d_i, sizeof(it)));
        CUDA_CHECK(cudaMalloc(&d_s, sizeof(strm)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint32_t) * NT * N));
        CUDA_CHECK(cudaMemcpy(d_e, elem, sizeof(elem), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_i, it, sizeof(it), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s, strm, sizeof(strm), cudaMemcpyHostToDevice));

        stream_kernel<<<1, NT>>>(d_e, d_i, d_s, S0, S1, d_out, NT, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t dev[NT][N];
        CUDA_CHECK(cudaMemcpy(dev, d_out, sizeof(uint32_t) * NT * N, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int t = 0; t < NT; ++t)
            for (int i = 0; i < N; ++i)
                if (dev[t][i] != seqs[t][i])
                    ok = false;
        std::printf("  kernel stream == host stream: %s\n", ok ? "PASS" : "FAIL");
        if (!ok)
            ++failures;

        CUDA_CHECK(cudaFree(d_e));
        CUDA_CHECK(cudaFree(d_i));
        CUDA_CHECK(cudaFree(d_s));
        CUDA_CHECK(cudaFree(d_out));
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
