#pragma once
//
// ============================================================================
// SPEC — Counter-based PRNG (Philox 4x32-10).   YOU implement all of this.
// (Per the collaboration model: this file states WHAT to build and the reference
//  constants; the interface design and the code are yours to write below.)
// ============================================================================
//
// WHY counter-based (the idea you must internalize before coding):
//   Every parallel GPU thread needs its own independent random stream with NO
//   shared state and NO locks. A counter-based generator makes the random value
//   a *stateless pure function* of (key, counter):   value = f(key, counter).
//   So thread i at step t just evaluates f(seed, {i, t, ...}). Fully
//   reproducible, embarrassingly parallel, zero coordination. This is what
//   cuRAND does internally; hand-writing it is the point.
//
// WHAT to build (semantics — not signatures; you choose the interface):
//   1. A stateless core: a bijection that maps a 128-bit counter (four uint32)
//      under a 64-bit key (two uint32) to four uniform-random uint32 words, via
//      10 Philox rounds. Same (counter, key) MUST always give the same output.
//      Per round: split the counter into two (hi, lo) pairs; each pair is mixed
//      by a 32x32->64 multiply with a fixed multiplier, then the high/low halves
//      are xor-combined with the key and the other lane. Between rounds, bump the
//      two key words by the Weyl constants below. (Derive the exact round from
//      the paper — that derivation is part of the learning.)
//   2. A per-thread wrapper: key = seed, counter = {tid, step, ...}; a "next"
//      operation that advances step and returns 4 fresh words.
//   3. Converters: uint32 -> uniform float in [0,1); and two uniforms -> a
//      standard normal via Box-Muller (you'll need Gaussian noise everywhere).
//   4. Must compile and give identical results on host and device (guard the
//      __host__ __device__ qualifiers so CPU tests can run).
//
// REFERENCE CONSTANTS (published values from the paper — provided so your output
// matches the known-answer test vectors the test harness will check):
//   round multipliers:  M0 = 0xD2511F53   (mixes counter lane 0)
//                       M1 = 0xCD9E8D57   (mixes counter lane 2)
//   key Weyl bumps:     W0 = 0x9E3779B9   (golden ratio)
//                       W1 = 0xBB67AE85   (sqrt 3 fractional)
//   rounds: 10
//
// Reference: Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as
// 1, 2, 3", SC'11.
//
// Once the bodies are in, the test harness (known-answer vectors
// + statistical uniformity/normality checks + host==device equality) comes next.
// ============================================================================

// ---- your implementation below ----
