# LayerNorm

File to write: `src/nn/layernorm.cuh`
Test: `tests/test_layernorm.cu` (target `test_layernorm`)

Normalises each token's channel vector to zero mean and unit variance, then applies a
learned per-channel scale and shift. GPT-2 puts one before attention and one before the
MLP, plus a final one before the output head.

## Shapes

    x     (N, C)     N = batch * tokens flattened, same convention as linear
    w     (C)        gain, initialised to 1
    b     (C)        bias, initialised to 0
    out   (N, C)
    mean  (N)        cached for backward
    rstd  (N)        cached for backward

Each row is normalised **independently**. Nothing crosses the row boundary. That is the
whole difference from BatchNorm, and it is why LayerNorm needs no running statistics and
behaves identically at train and eval time.

## Sub-step A: forward

Per row `n`:

    mean = (1/C) * sum_c x[n][c]
    var  = (1/C) * sum_c (x[n][c] - mean)^2
    rstd = 1 / sqrt(var + eps)
    xhat = (x[n][c] - mean) * rstd
    out[n][c] = xhat * w[c] + b[c]

`eps = 1e-5f`. Use the **biased** variance, divide by `C`, not `C - 1`. GPT-2 does, and the
test oracle does.

Store `mean[n]` and `rstd[n]`. Backward needs both and recomputing them costs a second pass
over the row for nothing. Store `rstd`, not `var`: backward wants the reciprocal square root
and you already paid for it.

Write:

    __global__ void layernorm_fwd(const float* x, const float* w, const float* b,
                                  float* out, float* mean, float* rstd, int N, int C)

**One thread per row.** Thread `n` loops over all `C` channels: once to accumulate the sum,
once to accumulate the squared deviation, once to write the output. Guard `n < N`.

Two things are wrong with that and both are fine for now:

- Consecutive threads read addresses `C` floats apart, so nothing coalesces.
- At `N = 16384` you have plenty of threads, but each does `3C` serial work.

The fast version is one **block** per row with a shared-memory reduction, which is Rung 5,
after the reductions lecture. Correct first.

Accumulate the sums in a local `float` and write once at the end. Do not accumulate into
global memory inside the loop.

## Sub-step B: backward

Backward receives `dout` (N, C) and produces `dx` (N, C), `dw` (C), `db` (C).

**The two parameter gradients** are column sums, the same shape of problem as `bias_bwd`:

    db[c] += sum over n of dout[n][c]
    dw[c] += sum over n of dout[n][c] * xhat[n][c]

You do not have `xhat` stored. Recompute it from `x`, `mean[n]` and `rstd[n]`, which is why
you cached them.

**The input gradient** is the interesting one. Every output in a row depends on every input
in that row, through the mean and the variance, so `dx` is not elementwise. The result:

    dxhat[c] = dout[n][c] * w[c]
    s1 = (1/C) * sum_c dxhat[c]
    s2 = (1/C) * sum_c dxhat[c] * xhat[c]
    dx[n][c] = rstd[n] * (dxhat[c] - s1 - xhat[c] * s2)

Read that as three corrections. The first term is the direct path. `s1` removes the mean
shift: if you push every channel up equally, the mean moves with them and nothing changes,
so that component of the gradient is projected out. `s2` removes the scale component: if you
push every channel out proportionally to how far it already is, the variance moves with it
and the normalised values do not change. LayerNorm is invariant to those two directions, so
its gradient must be zero along them. That is the whole derivation in words, and it is worth
being able to say out loud.

Write two kernels:

    __global__ void layernorm_bwd_dx(const float* dout, const float* x, const float* w,
                                     const float* mean, const float* rstd,
                                     float* dx, int N, int C)
    __global__ void layernorm_bwd_dwdb(const float* dout, const float* x,
                                       const float* mean, const float* rstd,
                                       float* dw, float* db, int N, int C)

`layernorm_bwd_dx`: one thread per row. Two passes over the row, first to build `s1` and
`s2`, then to write. Nothing else in the row is needed.

`layernorm_bwd_dwdb`: one thread per **channel**, looping over `n`. Accumulate two locals,
one `+=` each into `dw[c]` and `db[c]` at the end. Same slow-but-correct shape as `bias_bwd`,
same Rung 5 fix.

**beta semantics:** `dx` assigns, `dw` and `db` accumulate, exactly as in linear and for the
same reasons.

## Launchers

    launch_layernorm_fwd(const float* x, const float* w, const float* b,
                         float* out, float* mean, float* rstd, int N, int C)
    launch_layernorm_bwd(const float* dout, const float* x, const float* w,
                         const float* mean, const float* rstd,
                         float* dx, float* dw, float* db, int N, int C)

256 threads per block, `ceil(n/256)` blocks, bounds guard, as always.

## What the test does

1. forward vs CPU oracle, and that `mean` and `rstd` match
2. that each output row really has zero mean and unit variance when `w = 1`, `b = 0`
3. `dx` vs oracle
4. `dw` and `db` vs oracle, accumulating into pre-filled buffers
5. finite-difference gradient check on `L = sum(dout * out)` over every element of `x`, `w`
   and `b`, through your forward kernels only

Test 5 is the one that catches a dropped `s1` or `s2` term. A backward missing those still
looks plausible and is wrong by a few percent everywhere.
