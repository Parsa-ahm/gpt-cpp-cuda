# Softmax cross-entropy loss

File to write: `src/nn/cross_entropy.cuh`
Test: `tests/test_cross_entropy.cu` (target `test_cross_entropy`)

The last op in the forward pass and the first in the backward pass. It turns logits into a
single number, and its gradient is where backprop starts.

## Shapes

    logits    (N, V)    N = batch * tokens, V = vocab size
    targets   (N)       int32, the correct token id for each position, in [0, V)
    losses    (N)       per-token loss
    lse       (N)       log-sum-exp per row, cached for backward
    dlogits   (N, V)

## Sub-step A: forward

Per row `n`, with `y = targets[n]`:

    m   = max over v of logits[n][v]
    lse = m + log( sum over v of exp(logits[n][v] - m) )
    losses[n] = lse - logits[n][y]

That is it. Three passes over the row: max, sum, then one lookup.

Why that is the cross-entropy you know: `loss = -log(softmax(logits)[y])`, and
`softmax(logits)[y] = exp(logits[y] - lse)`, so `-log(...)` is `lse - logits[y]`. Written this
way you never form the softmax probabilities at all in the forward pass, so nothing can
underflow. Computing `p = exp(l - lse)` and then `-log(p)` would round a small probability to
zero and hand you `inf`.

The max subtraction is the same trick as attention, for the same reason. `V` is 50257 for
GPT-2 and logits routinely reach 20 or 30.

Cache `lse[n]`. Backward needs the softmax, and `softmax[n][v] = exp(logits[n][v] - lse[n])`
is one `exp` per element with no second reduction. Caching `lse` instead of the full
`(N, V)` probability matrix saves you a buffer the size of the logits, which at
`N = 16384, V = 50257` is 3.3 GB. You have 8 GB. This is not an optimisation, it is the
difference between running and not running.

The scalar loss the model reports is the mean over `N`. Do that reduction on the host for now:
download `losses`, sum, divide. It is `N` floats once per step and Rung 5 can fix it.

Write:

    __global__ void crossentropy_fwd(const float* logits, const int* targets,
                                     float* losses, float* lse, int N, int V)

One thread per row, looping over `V`. Same coalescing sin as layernorm, same Rung 5 fix.

## Sub-step B: backward

    dlogits[n][v] = ( softmax[n][v] - (v == targets[n] ? 1 : 0) ) * dloss

where `softmax[n][v] = exp(logits[n][v] - lse[n])` and `dloss` is the gradient flowing into
the mean loss, which is `1/N` when the reported loss is the mean.

This is the cleanest gradient in the whole model: **predicted minus actual**. The gradient on
the correct token is `p - 1`, always negative, pushing that logit up. The gradient on every
other token is `p`, always positive, pushing them down. Its magnitude is exactly how wrong the
model was.

It is that clean because softmax and cross-entropy are fused. Backprop through a softmax layer
and then a separate log-loss and you get the `diag(p) - p p^T` Jacobian you met in attention,
and the terms cancel to this. Never implement them separately. This is the third time the same
row-coupling structure has shown up (layernorm `s1`, attention softmax, here) and the third
time it collapses.

Write:

    __global__ void crossentropy_bwd(const float* logits, const float* lse, const int* targets,
                                     float* dlogits, float dloss, int N, int V)

One thread per **element**, `N*V` of them. Thread `i` is at `n = i / V`, `v = i % V`. This one
coalesces: consecutive threads read and write consecutive addresses.

`dlogits` **assigns**. This op is the only writer of that buffer, and it is the start of the
chain, so there is nothing to accumulate onto.

## Launchers

    launch_crossentropy_fwd(const float* logits, const int* targets,
                            float* losses, float* lse, int N, int V)
    launch_crossentropy_bwd(const float* logits, const float* lse, const int* targets,
                            float* dlogits, float dloss, int N, int V)

## What the test does

1. forward vs CPU oracle in double precision, including a row with logits around 100 to prove
   the max subtraction works. Without it that row is `inf` and the test fails loudly.
2. a known-answer case: uniform logits give exactly `log(V)`
3. that a confidently correct row gives a loss near 0 and a confidently wrong row gives a large
   finite loss, never `inf` or `NaN`
4. `dlogits` vs oracle
5. that each row of `dlogits` sums to `dloss * 0` within tolerance. The softmax row sums to 1
   and the one-hot sums to 1, so the difference sums to zero. Cheap, and it catches a missing
   `-1` on the target index.
6. finite-difference gradient check on the mean loss over every element of `logits`, through
   your forward kernel only, on a small `(N, V)`
