# AdamW

File to write: `src/optim/adamw.cuh`  (new directory, alongside `core/`, `nn/`, `rng/`)
Test: `tests/test_adamw.cu` (target `test_adamw`)

The optimiser. Elementwise, stateless across parameters, and the easiest kernel in Rung 2.
The only hard part is one detail that almost everyone gets wrong.

## State

For every parameter tensor you keep two buffers the same size:

    p   (n)   the parameters
    g   (n)   the gradients, as produced by your backward passes
    m   (n)   first moment,  zero-initialised
    v   (n)   second moment, zero-initialised

Plus a step counter `t`, starting at **1** on the first update, not 0.

## The update

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g

    mhat = m / (1 - beta1^t)
    vhat = v / (1 - beta2^t)

    p -= lr * ( mhat / (sqrt(vhat) + eps) + weight_decay * p )

GPT-2 values: `beta1 = 0.9`, `beta2 = 0.95`, `eps = 1e-8`, `weight_decay = 0.1`.

Note `beta2 = 0.95`, not the `0.999` you see in most Adam code. Karpathy uses 0.95 for GPT-2
and so does nanoGPT. Match the video.

### The bias correction, and why it exists

`m` and `v` start at zero, so on step 1 the moving average is pulled hard toward zero and
underestimates the true gradient magnitude by a factor of `(1 - beta)`. Dividing by
`1 - beta^t` undoes exactly that. At `t = 1` with `beta1 = 0.9` the correction is a factor of
10. By `t = 100` it is 1.0000. It matters only for the first few dozen steps, and without it
your first steps are far too small and the loss curve has a flat spot at the start that looks
like a bug in something else.

Compute `beta1^t` and `beta2^t` **on the host** and pass them in. One `powf` per step instead
of one per element.

### The W in AdamW, which is the whole point

Weight decay is applied **directly to the parameter**, as `lr * weight_decay * p` in the
update, and is **not** added into the gradient.

The wrong version, which is plain Adam with L2 regularisation:

    g = g + weight_decay * p        then run Adam on that g

Those are not the same. In the wrong version the decay term goes through `m` and `v` and gets
divided by `sqrt(vhat)`, so parameters with large gradients get *less* decay than parameters
with small ones. The regularisation strength ends up coupled to the gradient history, which is
not what anybody wants. Decoupling it is the entire contribution of the AdamW paper
(Loshchilov and Hutter 2019) and it is why the optimiser has a different name.

If you take one thing from this spec, take that. It is a common interview question and the
answer is "decoupled weight decay, applied to the parameter not the gradient, so the decay is
not scaled by the second moment."

## The kernel

    __global__ void adamw_step(float* p, const float* g, float* m, float* v, int n,
                               float lr, float beta1, float beta2, float eps,
                               float weight_decay, float beta1_pow_t, float beta2_pow_t)

One thread per element, bounds guard. Read `g[i]`, update `m[i]` and `v[i]` in place, write
`p[i]`. No reductions, no atomics, no inter-thread anything. It is memory bound: four buffers
read or written per element and about ten flops. Note that in the README, it is a good
roofline example for Rung 5.

    launch_adamw_step(float* p, const float* g, float* m, float* v, int n,
                      float lr, float beta1, float beta2, float eps,
                      float weight_decay, int t)

The launcher computes `powf(beta1, t)` and `powf(beta2, t)` on the host and calls the kernel.

## Zeroing gradients

Not this op's job, but say it out loud now because it bites everyone: your `dW`, `db`, `dwte`
and `dwpe` kernels all **accumulate**. Something has to zero them between steps. Add a trivial
`launch_zero(float* p, int n)` to this file and call it at the top of each training step. When
Rung 3 produces a loss curve that plateaus immediately, this is the first thing to check.

## What the test does

1. one step against a CPU oracle from zeroed state, several `lr` and `weight_decay`
2. fifty steps against the oracle, checking drift does not accumulate
3. **a bias-correction check**: with `g` constant, step 1 must move `p` by very close to
   `lr * (1 + weight_decay * p)`. Without bias correction it moves by about a tenth of that.
   This test fails loudly if you drop the correction.
4. **a decoupling check**: run with `weight_decay = 0.1` and a large constant gradient, and
   separately with `weight_decay = 0` on a gradient pre-loaded with `0.1 * p`. The two must
   give **different** answers. If they match, you implemented Adam with L2, not AdamW.
5. a convergence check: minimise `f(x) = sum(x^2)` from a random start, assert the loss falls
   monotonically and reaches near zero in 200 steps
6. that `m` and `v` match the oracle after the run, not just `p`
