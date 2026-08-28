// ============================================================================
// KAT + host==device harness for philox().
//
// Checks three things:
//   (1) KAT: your philox() output matches the published Philox 4x32-10
//       known-answer vectors (bit-exact). If these pass, your round is correct.
//   (2) host==device: the SAME philox() run inside a CUDA kernel produces
//       identical bits to the host run. This is the __host__ __device__ contract.
//   (3) determinism: same (ctr,key) called twice gives the same words.
//
// The expected values below are the ground truth (independently generated from
// the canonical Philox definition). They are just numbers on purpose — the round
// derivation stays yours; this file only judges the result.
//
// Exit code 0 = all pass; nonzero = something failed (so CTest reports red).
// ============================================================================
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "rng/philox.hpp"

// ---- minimal CUDA error check -------------------------------------------------
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            return 99;                                                                             \
        }                                                                                          \
    } while (0)

// ---- test vectors -------------------------------------------------------------
struct Case {
    const char* name;
    uint32_t ctr[4];
    uint32_t key[2];
    uint32_t expect[4];
};

// Canonical Philox 4x32-10 known-answer vectors.
static const Case KAT[] = {
    {"all-zero",
     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
     {0x00000000, 0x00000000},
     {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
    {"all-ones",
     {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
     {0xffffffff, 0xffffffff},
     {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
    {"pi-digits",
     {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
     {0xa4093822, 0x299f31d0},
     {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
};
static const int NCASES = sizeof(KAT) / sizeof(KAT[0]);

// ---- device side: run philox() inside a kernel --------------------------------
// One thread per case. Copies its ctr/key into locals so philox()'s
// reference-to-array parameters can bind, then writes the 4 output words.
__global__ void philox_kernel(const uint32_t* ctr_in, const uint32_t* key_in, uint32_t* out,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    uint32_t c[4] = {ctr_in[idx * 4 + 0], ctr_in[idx * 4 + 1], ctr_in[idx * 4 + 2],
                     ctr_in[idx * 4 + 3]};
    uint32_t k[2] = {key_in[idx * 2 + 0], key_in[idx * 2 + 1]};
    Philox4 r = philox(c, k);
    for (int j = 0; j < 4; ++j)
        out[idx * 4 + j] = r.w[j];
}

static bool eq4(const uint32_t* a, const uint32_t* b) {
    for (int j = 0; j < 4; ++j)
        if (a[j] != b[j])
            return false;
    return true;
}

static void print4(const char* label, const uint32_t* w) {
    std::printf("    %-8s {0x%08x, 0x%08x, 0x%08x, 0x%08x}\n", label, w[0], w[1], w[2], w[3]);
}

int main() {
    int failures = 0;

    // ---------- (1) HOST KAT ----------
    std::printf("[host KAT]\n");
    uint32_t host_out[NCASES][4];
    for (int i = 0; i < NCASES; ++i) {
        Philox4 r = philox(KAT[i].ctr, KAT[i].key);
        for (int j = 0; j < 4; ++j)
            host_out[i][j] = r.w[j];
        bool ok = eq4(host_out[i], KAT[i].expect);
        std::printf("  %-10s %s\n", KAT[i].name, ok ? "PASS" : "FAIL");
        if (!ok) {
            print4("got", host_out[i]);
            print4("expect", KAT[i].expect);
            ++failures;
        }
    }

    // ---------- (3) determinism ----------
    std::printf("[determinism]\n");
    for (int i = 0; i < NCASES; ++i) {
        Philox4 a = philox(KAT[i].ctr, KAT[i].key);
        Philox4 b = philox(KAT[i].ctr, KAT[i].key);
        bool ok = eq4(a.w, b.w);
        std::printf("  %-10s %s\n", KAT[i].name, ok ? "PASS" : "FAIL");
        if (!ok)
            ++failures;
    }

    // ---------- (2) host == device ----------
    std::printf("[host == device]\n");
    uint32_t flat_ctr[NCASES * 4], flat_key[NCASES * 2];
    for (int i = 0; i < NCASES; ++i) {
        for (int j = 0; j < 4; ++j)
            flat_ctr[i * 4 + j] = KAT[i].ctr[j];
        for (int j = 0; j < 2; ++j)
            flat_key[i * 2 + j] = KAT[i].key[j];
    }
    uint32_t *d_ctr, *d_key, *d_out;
    CUDA_CHECK(cudaMalloc(&d_ctr, sizeof(flat_ctr)));
    CUDA_CHECK(cudaMalloc(&d_key, sizeof(flat_key)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint32_t) * NCASES * 4));
    CUDA_CHECK(cudaMemcpy(d_ctr, flat_ctr, sizeof(flat_ctr), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, flat_key, sizeof(flat_key), cudaMemcpyHostToDevice));

    philox_kernel<<<1, NCASES>>>(d_ctr, d_key, d_out, NCASES);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t dev_out[NCASES][4];
    CUDA_CHECK(cudaMemcpy(dev_out, d_out, sizeof(uint32_t) * NCASES * 4, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NCASES; ++i) {
        bool ok = eq4(dev_out[i], host_out[i]);
        std::printf("  %-10s %s\n", KAT[i].name, ok ? "PASS" : "FAIL");
        if (!ok) {
            print4("host", host_out[i]);
            print4("device", dev_out[i]);
            ++failures;
        }
    }

    CUDA_CHECK(cudaFree(d_ctr));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_out));

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
