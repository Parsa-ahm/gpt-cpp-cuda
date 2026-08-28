#pragma once
#include <cstdint>
#include <cmath>
constexpr float PI = 3.14159265358979323846f;
// REFERENCE CONSTANTS (published values from the paper

//   round multipliers:  M0 = 0xD2511F53   (mixes counter lane 0)
//                       M1 = 0xCD9E8D57   (mixes counter lane 2)
//   key Weyl bumps:     W0 = 0x9E3779B9   (golden ratio)
//                       W1 = 0xBB67AE85   (sqrt 3 fractional)
//   rounds: 10
//
// Reference: Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as
// 1, 2, 3", SC'11.
//
// ============================================================================

struct Philox4 {
    uint32_t w[4];
};

__host__ __device__ Philox4 philox(const uint32_t (&ctr)[4], const uint32_t (&key)[2]) {
    const uint8_t rounds = 10;
    uint32_t x0 = ctr[0];
    uint32_t x1 = ctr[1];
    uint32_t x2 = ctr[2];
    uint32_t x3 = ctr[3];
    uint32_t k0 = key[0];
    uint32_t k1 = key[1];

    constexpr uint32_t M0 = 0xD2511F53;
    constexpr uint32_t M1 = 0xCD9E8D57;

    constexpr uint32_t W0 = 0x9E3779B9;
    constexpr uint32_t W1 = 0xBB67AE85;

    for (int i = 0; i < rounds; ++i) {
        uint64_t l0 = (uint64_t)x0 * M0;
        uint64_t l1 = (uint64_t)x2 * M1;

        uint32_t v0 = (uint32_t)(l1 >> 32) ^ x1 ^ k0;
        uint32_t v1 = (uint32_t)l1;
        uint32_t v2 = (uint32_t)(l0 >> 32) ^ x3 ^ k1;
        uint32_t v3 = (uint32_t)l0;

        k0 += W0;
        k1 += W1;

        x0 = v0;
        x1 = v1;
        x2 = v2;
        x3 = v3;
    }

    return {x0, x1, x2, x3};
}

struct PhiloxStream {
    uint32_t key[2];
    uint32_t element, iter, stream, block;
    Philox4 buf;
    int cursor;
    float cached_n;
    bool has_cache;
    __host__ __device__ void init(uint32_t ctr[4], uint32_t seed[2]) {
        element = ctr[0];
        iter = ctr[1];
        stream = ctr[2];
        block = 0;
        key[0] = seed[0];
        key[1] = seed[1];
        cursor = 4;
        has_cache = false;
    }

    __host__ __device__ uint32_t next_u32() {
        if (cursor == 4) {
            uint32_t ctr[4] = {element, iter, stream, block};
            buf = philox(ctr, key);
            cursor = 0;
            ++block;
        }
        return buf.w[cursor++];
    }

    __host__ __device__ float next_float() {
        constexpr float denom = 2.3283064365386963e-10f;  // = 1 / 2^32
        return next_u32() * denom;
    }

    __host__ __device__ float next_n() {
        if (has_cache) {
            has_cache = false;
            return cached_n;
        }
        float u1 = next_float();
        float u2 = next_float();
        if (u1 < 1.0e-7f) {
            u1 += 1.0e-7f;
        }
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * PI * u2;
        cached_n = r * sinf(theta);
        has_cache = true;
        return r * cosf(theta);
    }
};
