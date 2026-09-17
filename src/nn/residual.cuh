#pragma once
// ============================================================================
// SPEC — residual add, forward + backward.   YOU implement all of this.
//
// Second op of Rung 2. The forward is one line. The BACKWARD is the reason this
// op comes before linear and long before attention, because it teaches the two
// rules that govern how gradient moves through a graph that is not a straight
// line. GPT-2 is not a straight line: that is what "residual stream" means.
//
// ---------------------------------------------------------------------------
// WHERE IT LIVES
//
// Twice per transformer layer:
//
//     x -> LayerNorm -> Attention -----> (+) -> ...
//      \_______________________________/          <- this op
//
//     x -> LayerNorm -> MLP -----------> (+) -> ...
//      \_______________________________/          <- and this one
//
// The skip path is why 12-layer (and 96-layer) transformers train at all: it
// gives gradient a route from the loss to the early layers that does not pass
// through every intervening weight matrix.
//
// ---------------------------------------------------------------------------
// FORWARD
//
//     out[i] = a[i] + b[i]
//
// That is the whole thing.
//
// ---------------------------------------------------------------------------
// BACKWARD — two rules, and they are duals of each other
//
// RULE 1: an ADD in forward COPIES gradient to both inputs, unchanged.
//
//   out = a + b, so d(out)/da = 1 and d(out)/db = 1. Chain rule:
//
//       d_a[i] = dy[i] * 1 = dy[i]
//       d_b[i] = dy[i] * 1 = dy[i]
//
//   No scaling, no transform. An add is a gradient tee. (Multiply is the
//   contrast: for out = a*b you get d_a = dy*b, d_b = dy*a. Each input's
//   gradient is scaled by the OTHER input. Add is the easy case.)
//
// RULE 2: a FAN-OUT in forward SUMS gradient in backward.
//
//   Look at the diagram. `x` is used TWICE: once into LayerNorm, once straight
//   into the add. Two consumers means x affects the loss through two routes, so
//
//       d_x = (gradient arriving from the LayerNorm route)
//           + (gradient arriving from the skip route)
//
//   Multivariable chain rule. Contributions add.
//
// These two rules are the same fact seen from both ends, and together they are
// all the "autodiff" this project needs.
//
// ---------------------------------------------------------------------------
// THE CONSEQUENCE FOR YOUR KERNEL:  +=  NOT  =
//
// Because of Rule 2, this op must ACCUMULATE:
//
//       d_a[i] += dy[i];
//       d_b[i] += dy[i];
//
// Note this is the OPPOSITE of what gelu_bwd does, and the difference is not
// arbitrary. GELU sits in a chain: it is the only writer of its dx, so it owns
// that buffer and assigns. Residual writes into buffers that other ops also
// write into, so it must add to what is already there.
//
// The contract that makes `+=` safe: the CALLER zeroes every gradient buffer at
// the start of the step. That is one line in the Rung 3 training loop and it is
// also why PyTorch makes you call `optimizer.zero_grad()`. Now you know what
// that call is actually for.
//
// A missing zero-out is one of the nastiest bugs in this project: nothing
// crashes, gradients just quietly grow every step and training diverges a few
// hundred steps in. Test 3 below exists to pin this behaviour down now.
//
// ---------------------------------------------------------------------------
// WHAT TO WRITE
//
//  (1) __global__ void residual_fwd(const float* a, const float* b, float* out, int n)
//        - index, guard, out[i] = a[i] + b[i]
//
//  (2) __global__ void residual_bwd(const float* dy, float* d_a, float* d_b, int n)
//        - index, guard, d_a[i] += dy[i];  d_b[i] += dy[i];
//        - note what is NOT in the parameter list: `a`, `b`, and `out`. The
//          derivative of an add is the constant 1, so backward needs neither
//          the inputs nor the output. This op caches NOTHING. Compare GELU,
//          which had to keep `x` alive. Per-op cache decisions, again.
//        - d_a and d_b are NOT const. You are writing through both.
//
//  (3) inline host launchers, same shape as the GELU ones:
//        void launch_residual_fwd(const float* d_a_in, const float* d_b_in,
//                                 float* d_out, int n)
//        void launch_residual_bwd(const float* d_dy, float* d_da, float* d_db, int n)
//      - 256 threads, ceil-divided blocks
//      - use CUDA_CHECK_KERNEL() from "cuda_check.hpp" this time instead of the
//        bare cudaGetLastError(); cudaDeviceSynchronize(); pair. Go back and
//        switch gelu.cuh over to it too.
//
// ---------------------------------------------------------------------------
// PERFORMANCE NOTE (for the Rung 5 writeup)
//
// Forward: 2 loads + 1 store = 12 bytes per element, ~1 flop. Backward: 1 load
// + 2 read-modify-writes = 20 bytes per element, 2 flops. Even more bandwidth-
// starved than GELU. There is nothing to optimize here in isolation, which is
// exactly why real engines FUSE this op into its neighbour (gelu+residual in
// one kernel, so the intermediate never makes the round trip to global memory).
// Do not fuse anything yet. Note it as a Rung 5 candidate and move on.
//
// ---------------------------------------------------------------------------
// WHAT THE TEST DOES (tests/test_residual.cu — already written)
//   1. forward vs CPU oracle, incl. non-multiples of 256.
//   2. backward: d_a and d_b both equal dy, starting from zeroed buffers.
//   3. ACCUMULATION: pre-fills d_a and d_b with known non-zero junk, then runs
//      backward and checks the result is junk + dy. An `=` kernel passes tests
//      1 and 2 and fails only this one. This is the test that matters.
//   4. aliasing: d_a and d_b pointing at the SAME buffer, which is what happens
//      when a tensor feeds both branches of the add. Result must be junk + 2*dy.
// ============================================================================

#include <cuda_runtime.h>

#include "cuda_check.hpp"

__global__ void residual_fwd(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    out[i] = a[i] + b[i];
}

inline void launch_residual_fwd(const float* d_a, const float* d_b, float* d_out, int d_n) {
    const int threads = 256;
    const int blocks = (d_n + threads - 1) / threads;
    residual_fwd<<<blocks, threads>>>(d_a, d_b, d_out, d_n);
    CUDA_CHECK_KERNEL();
}

__global__ void residual_bwd(const float* dy, float* d_a, float* d_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    d_a[i] += dy[i];
    d_b[i] += dy[i];
}

inline void launch_residual_bwd(const float* d_dy, float* d_a, float* d_b, int d_n) {
    const int threads = 256;
    const int blocks = (d_n + threads - 1) / threads;
    residual_bwd<<<blocks, threads>>>(d_dy, d_a, d_b, d_n);
    CUDA_CHECK_KERNEL();
}