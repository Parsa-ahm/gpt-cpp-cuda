# Embedding: tokens + positions

File to write: `src/nn/embedding.cuh`
Test: `tests/test_embedding.cu` (target `test_embedding`)

The first layer of the model. Turns integer token ids into vectors and adds a learned
position vector. Your first op whose input is **not** a float tensor, and your first
backward that needs **atomics**.

## Shapes

    ids   (B, T)     int32 token ids, values in [0, V)
    wte   (V, C)     token embedding table, V = vocab size
    wpe   (T, C)     position embedding table
    out   (B, T, C)

## Sub-step A: forward

    out[b][t][c] = wte[ids[b][t]][c] + wpe[t][c]

That is the whole op. It is a gather: no arithmetic, just two table lookups and an add.

Write:

    __global__ void embedding_fwd(const int* ids, const float* wte, const float* wpe,
                                  float* out, int B, int T, int C)

One thread per **output element**, `B*T*C` of them. Thread `i` is at `b = i / (T*C)`,
`t = (i / C) % T`, `c = i % C`. Then `id = ids[b*T + t]` and you read `wte[id*C + c]` and
`wpe[t*C + c]`.

This one actually coalesces well: consecutive threads have consecutive `c`, so they read
consecutive addresses in both tables and write consecutive addresses in `out`. Note that and
move on. Not every kernel needs a Rung 5 apology.

## Sub-step B: backward

There is **no `dx`**. The input is integer token ids. Gradients do not flow into them, there
is nothing to flow into. The only gradients are into the two tables.

    dwte[ids[b][t]][c] += dout[b][t][c]
    dwpe[t][c]         += sum over b of dout[b][t][c]

Look carefully at `dwte`. The row you write to is chosen by the **data**, and the same token
appears many times in a batch. Every occurrence of the word "the" writes to the same row of
`dwte`. If two threads do that at once with a plain `+=` you lose one of the updates. That is
a race, and it will not show up as a crash, it will show up as a gradient that is quietly too
small.

So: `atomicAdd(&dwte[id*C + c], dout[...])`.

`dwpe` has the same problem across the batch dimension: every one of the `B` sequences writes
to the same `wpe[t]`. Two options, both acceptable:

- one thread per output element with `atomicAdd`, same as `dwte`
- one thread per `(t, c)` pair looping over `b`, accumulating a local, one plain `+=` at the
  end. No atomic needed because each `(t,c)` now has exactly one owner.

The second is faster and shows you understand why the first needed the atomic. Do the second
if you see it; do the first if you want them symmetric.

Write:

    __global__ void embedding_bwd(const int* ids, const float* dout,
                                  float* dwte, float* dwpe, int B, int T, int C)

(or split into two kernels if you take the second `dwpe` option).

**Determinism warning, and it matters later.** `atomicAdd` on floats accumulates in whatever
order the hardware happens to schedule. Float addition is not associative, so two runs of the
same backward give bitwise-different answers in the last few bits. This is normal and every
framework does it. It is also why the test compares with a tolerance rather than exact
equality, and why "my loss curve is not bit-reproducible" is not a bug.

Both tables **accumulate**. Never assign.

## Launchers

    launch_embedding_fwd(const int* ids, const float* wte, const float* wpe,
                         float* out, int B, int T, int C)
    launch_embedding_bwd(const int* ids, const float* dout,
                         float* dwte, float* dwpe, int B, int T, int C)

## What the test does

1. forward vs CPU oracle, including a batch where the same token id repeats many times
2. that `ids` out of range are never dereferenced (the test only passes valid ids; this is a
   note for you, add the guard anyway)
3. `dwte` vs oracle, with a deliberately collision-heavy id distribution, accumulating into a
   pre-filled buffer
4. `dwpe` vs oracle, accumulating
5. a run-to-run consistency check: backward twice on identical inputs, assert the two results
   agree to `1e-4` relative, not bitwise. This documents the atomics non-determinism rather
   than pretending it does not exist.

There is no finite-difference check on `ids` because there is no gradient there. The check on
the tables is the oracle comparison plus a finite difference on `wte` and `wpe` through your
forward kernel.
