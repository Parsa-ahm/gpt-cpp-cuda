# Step 2 of linear: the layer itself

File to write: `src/nn/linear.cuh`
Test: `tests/test_linear.cu` (target `test_linear`)
Prerequisite: `sgemm_rm` from `specs/cublas_layout.md` must be green first.

This is the first op with **weights**, so it is the first backward that produces
more than one gradient. Do it in three sub-steps and build between each.

## Shapes and the convention

    x    (N, C)      N = batch * tokens, flattened. C = channels in.
    W    (C, OC)     OC = channels out.
    b    (OC)
    out  (N, OC)

`W` is stored `(C, OC)`, input-dim first. That is deliberate: forward becomes a
plain `x @ W` with no transpose, and it is also exactly how HuggingFace GPT-2
stores these weights (its `Conv1D`), so the Rung 4 weight loader is a straight
copy with no fixups. PyTorch's `nn.Linear` stores the other way round, `(OC, C)`;
we are matching GPT-2, not `nn.Linear`.

`N` collapses batch and tokens into one dimension. A linear layer treats every
token independently, so it does not care which batch row a token came from.
Folding them gives one big matmul instead of a loop, which is most of why
transformers run fast on GPUs.

## Sub-step A: forward

    out = x @ W        then      out[n][o] += b[o]

The matmul is one `sgemm_rm` call, no transposes, `alpha = 1`, `beta = 0`.

The bias is your own kernel. It adds a vector of length `OC` to every one of the
`N` rows. Note the index arithmetic: a thread owning flat position `i` of a
`(N, OC)` row-major array is at row `i / OC` and column `i % OC`, and the bias
element it needs is `b[i % OC]`. Row-major means the column index is the fast
one, so `i % OC` is the bias index.

Write:

    __global__ void bias_fwd(float* out, const float* b, int N, int OC)
    inline void launch_linear_fwd(const float* x, const float* W, const float* b,
                                  float* out, int N, int C, int OC)

`bias_fwd` reads and writes `out` in place, right after the matmul. `out` is not
`const` there.

## Sub-step B: backward, the two matmuls

Backward receives `dy`, shape `(N, OC)`, and must produce three things.

**Gradient into the input:**

    dx = dy @ W^T          (N,OC) @ (OC,C) = (N,C)

**Gradient into the weights:**

    dW = x^T @ dy          (C,N) @ (N,OC) = (C,OC)

Both are one `sgemm_rm` call. `W` is stored `(C,OC)` so `W^T` means
`trans_b = true`; `x` is stored `(N,C)` so `x^T` means `trans_a = true`.

How to be sure you have not swapped them: **a gradient always has the same shape
as the thing it is the gradient of.** `dx` must come out `(N,C)` like `x`, `dW`
must come out `(C,OC)` like `W`. Only one arrangement of transposes gives each,
so the shapes are a complete check. Write the shapes down before you write the
call, every time. This is the single highest-value habit in the whole rung.

Where the two come from, if you want the intuition rather than the rule:
`out[n][o] = sum_c x[n][c] * W[c][o]`. Differentiate w.r.t. `x[n][c]` and the
only surviving term is `W[c][o]`, summed over `o`, which is `dy @ W^T`.
Differentiate w.r.t. `W[c][o]` and you get `x[n][c]` summed over `n`, which is
`x^T @ dy`. The dimension each one sums over is the dimension that is *not*
shared with the answer.

Notice `dW` sums over `N`, the token dimension. Every token contributes to every
weight and the matmul adds those contributions up. That is why `dW` has `W`'s
shape no matter how big the batch is, and why bigger batches give less noisy
gradients.

**beta, per output:**

    dx:  beta = 0.0f    assign
    dW:  beta = 1.0f    accumulate

`dW` accumulates because parameter gradients sum across a step: gradient
accumulation over micro-batches, and tied weights (GPT-2's output projection
shares the token embedding matrix, so it receives gradient twice). `dx` assigns
because this op is the only writer of that buffer. Free either way, `beta` costs
nothing.

Caching: backward needs `x` (for `dW`) and `W` (for `dx`). It does not need
`out`. So forward must keep `x` alive, same as GELU.

## Sub-step C: backward, the bias

    db[o] += sum over all n of dy[n][o]

A column sum. Every one of the `N` rows contributes to the same `OC` numbers.

This is your first **reduction**: an output that depends on many inputs rather
than one. Write the simple version now, one thread per output column, looping
over `N` internally:

    __global__ void bias_bwd(const float* dy, float* db, int N, int OC)

with `o = blockIdx.x * blockDim.x + threadIdx.x`, guard `o < OC`, then a
`for` loop over `n` accumulating `dy[n * OC + o]` into a local, and finally
`db[o] += local`.

Accumulate into a local float inside the loop, then do one `+=` into `db[o]` at
the end. Not `db[o] +=` inside the loop, which would be `N` round trips to
global memory instead of one.

Two things are wrong with this kernel and both are fine for now:

- Only `OC` threads do any work. At `OC = 1536` that is 6 blocks on a 40-SM GPU,
  so most of the card is idle.
- Consecutive loop iterations read addresses `OC` floats apart, so the reads do
  not coalesce.

Rung 5 fixes it with a proper parallel reduction. Write the slow correct one,
note it, move on. Correct then fast, never the other way.

## Launchers

    launch_linear_fwd (const float* x, const float* W, const float* b,
                       float* out, int N, int C, int OC)
    launch_linear_bwd (const float* x, const float* W, const float* dy,
                       float* dx, float* dW, float* db, int N, int C, int OC)

## What the test does

1. forward vs CPU oracle, including the bias
2. `dx` vs oracle
3. `dW` vs oracle, and that it accumulates into a pre-filled buffer
4. `db` vs oracle, and that it accumulates
5. a finite-difference gradient check on a scalar loss `L = sum(dy * out)` with
   `dy` fixed random. It perturbs each element of `x`, `W` and `b` one at a time,
   remeasures `L` through your forward kernels, and compares against what your
   backward claimed. This is the check that catches a swapped transpose: tests 2
   through 4 compare against an oracle I wrote from the same formulas you are
   reading, so we could both be wrong the same way. The finite difference only
   uses your forward pass, so it cannot share the mistake.
