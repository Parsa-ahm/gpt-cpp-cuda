#pragma once
//
// Counter-based PRNG (Philox 4x32-10), hand-written.
// ---------------------------------------------------------------------------
// WHY counter-based: every parallel GPU thread needs its own independent random
// stream with NO shared state and NO locks. A counter-based generator gives that
// for free: the random value is a stateless function f(key, counter). Thread i at
// step t just evaluates f(seed, {i, t}) -- fully reproducible, embarrassingly
// parallel. This is exactly what curand uses internally; hand-writing it is the
// legit systems flex, not a toy.
//
// Reference: Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as
// 1, 2, 3", SC'11. Philox-4x32-10 = 10 rounds over a 4x32-bit counter with a
// 2x32-bit key, using two fixed multipliers and the Weyl constants below.
//
// STATUS: STUB. Bodies come first; the test harness
// (statistical + known-answer-test vectors from the paper) comes separately.
// ---------------------------------------------------------------------------

#include <cstdint>

// Usable on both host and device once implemented. (__host__ __device__ is a
// no-op under a plain C++ compiler via this guard, so tests can run on CPU.)
#ifndef PHILOX_HD
#  if defined(__CUDACC__)
#    define PHILOX_HD __host__ __device__
#  else
#    define PHILOX_HD
#  endif
#endif

namespace rng {

// Philox constants (from the paper). Provided so the KAT vectors line up.
constexpr uint32_t kPhiloxM0 = 0xD2511F53u;  // multiplier for lane 0
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;  // multiplier for lane 2
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // key Weyl bump (golden ratio)
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;  // key Weyl bump (sqrt 3)

struct uint4x { uint32_t x, y, z, w; };
struct uint2x { uint32_t x, y; };

// Stateless core: 10-round Philox bijection of a 128-bit counter under a 64-bit key.
// Returns four uniform-random 32-bit words. Same (counter, key) -> same output.
//   TODO(parsa): implement the single-round function then loop it 10 times,
//   bumping the key by (kPhiloxW0, kPhiloxW1) between rounds.
PHILOX_HD inline uint4x philox4x32_10(uint4x counter, uint2x key);

// Convenience wrapper: a thread's stream. key = seed; counter = {tid, step, 0, 0}.
// next() advances the step and returns 4 fresh words.
struct Philox {
  uint2x key;       // seed
  uint4x counter;   // {tid, step, hi, lo}

  PHILOX_HD Philox(uint64_t seed, uint64_t tid);  // TODO(parsa)
  PHILOX_HD uint4x next_uint4();                    // TODO(parsa): calls core, ++step
  PHILOX_HD float  next_uniform();                  // TODO(parsa): one u32 -> [0,1)
  PHILOX_HD float  next_normal();                   // TODO(parsa): Box-Muller from two uniforms
};

}  // namespace rng
