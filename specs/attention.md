# Causal multi-head self-attention

File to write: `src/nn/attention.cuh`
Test: `tests/test_attention.cu` (target `test_attention`)
Watch first: Karpathy "Let's build GPT" (done 2026-09-20). Have your shape notes next to you.

The only op where tokens talk to each other. Everything else in the model treats each token
independently. Budget two sessions for this one, not one.

## What this op does and does not do

It does **not** do the QKV projection or the output projection. Those are `linear` calls you
already have. This op takes the already-projected `qkv` and returns the attention output,
which you then feed through another `linear`. Keeping the matmuls outside is how llm.c splits
it too, and it means this file contains only the part that is actually attention.

## Shapes

    qkv   (B, T, 3C)     q, k, v concatenated along the last dim, in that order
    att   (B, NH, T, T)  the softmax weights, cached for backward
    out   (B, T, C)

    NH = number of heads,  HS = C / NH  (head size).  C = NH * HS exactly.

Indexing, which is where every bug lives. For batch `b`, time `t`, head `h`, element `i`:

    q[b][t][h][i]  is  qkv[b*T*3C + t*3C + 0*C + h*HS + i]
    k[b][t][h][i]  is  qkv[b*T*3C + t*3C + 1*C + h*HS + i]
    v[b][t][h][i]  is  qkv[b*T*3C + t*3C + 2*C + h*HS + i]

Write that down before you write the kernel. The `0*C / 1*C / 2*C` offsets are the whole
trick: one tensor, three logical views.

Heads are independent. A head never reads another head's `q`, `k` or `v`. Multi-head is not
a new mechanism, it is the same mechanism run `NH` times on `HS`-sized slices in parallel.

## Sub-step A: forward

For each `(b, h, t)`:

    scale = 1 / sqrt(HS)

    for t2 in [0, t]:                          # causal: never past t
        preatt[t2] = scale * dot(q[b][t][h][:], k[b][t2][h][:])

    m = max over t2 of preatt[t2]              # numerical stability
    for t2 in [0, t]:
        e[t2] = exp(preatt[t2] - m)
    s = sum of e
    att[b][h][t][t2] = e[t2] / s

    out[b][t][h][i] = sum over t2 in [0,t] of att[b][h][t][t2] * v[b][t2][h][i]

Three things to be deliberate about:

**Causality by loop bound, not by masking.** Do not fill a `-inf` into positions `t2 > t` and
softmax over the whole row. Just stop the loop at `t`. Same answer, half the work, and no
`-inf` arithmetic to get wrong. Leave `att[b][h][t][t2]` for `t2 > t` as zero so backward and
the oracle agree on the buffer's contents.

**Subtract the max before exp.** `exp(preatt)` with raw logits overflows to `inf` the moment
a dot product exceeds about 88 in fp32. Subtracting the row max makes the largest term exactly
`exp(0) = 1` and changes nothing mathematically, since the constant cancels in the ratio. This
is not a nicety, it is the difference between a model that trains and one that produces NaN at
step 200.

**The scale.** `1/sqrt(HS)`, not `1/sqrt(C)`. The dot product runs over `HS` terms, so its
variance grows with `HS`. Without the scale, softmax saturates as the model gets wider and the
gradient dies.

Write:

    __global__ void attention_fwd(const float* qkv, float* att, float* out,
                                  int B, int T, int C, int NH)

**One thread per `(b, h, t)`.** `B*NH*T` threads. Each one owns one query position: it
computes that row of `att` and that slice of `out`. No thread touches another thread's output,
so no atomics in the forward pass.

Note what this costs: `att` is `B * NH * T * T` floats. At `B=4, NH=6, T=256` that is 6 MB,
fine. At GPT-2 124M with `T=1024, NH=12` it is `B * 48 MB`. That is the memory wall
FlashAttention exists to remove, by never materialising `att` at all. You are building the
version that stores it so that in Rung 5 you can build the version that does not, and have a
number to compare against. Write it down in the README as a known cost.

## Sub-step B: backward

Three kernels, in order. Resist the urge to fuse them. Each has one job, each can be tested,
and fusing is a Rung 5 move.

### B1: `datt` and `dv`

    datt[b][h][t][t2] = dot(dout[b][t][h][:], v[b][t2][h][:])        for t2 <= t
    dv[b][t2][h][i]  += sum over t >= t2 of att[b][h][t][t2] * dout[b][t][h][i]

`datt` has one owner per element, no atomics. `dv` is summed over `t`, and if you thread over
`(b,h,t)` then many threads hit the same `t2`, so `dv` needs `atomicAdd`. If you instead
thread over `(b,h,t2)` and loop `t` from `t2` to `T-1`, each `dv` element has one owner and
you need no atomic. Either is acceptable; the second is better and you should see why.

### B2: softmax backward

    dpreatt[t][t2] = att[t][t2] * (datt[t][t2] - sum over t3 of att[t][t3] * datt[t][t3])

for `t2 <= t`, and zero otherwise. One thread per `(b,h,t)`: compute the inner sum over the
row once into a local, then write the row.

That formula is the Jacobian of softmax collapsed into a form with no matrix in it. Softmax's
Jacobian is `diag(p) - p p^T`, so `dpre = p * (dy - dot(p, dy))`. The `dot(p, dy)` term is the
row-wide coupling: pushing one attention weight up necessarily pushes the others down, because
they sum to one. Same shape of argument as the `s1` term in layernorm. You will meet it a
third time in cross-entropy.

### B3: `dq` and `dk`

    dq[b][t][h][i]  += scale * sum over t2 <= t of dpreatt[t][t2] * k[b][t2][h][i]
    dk[b][t2][h][i] += scale * sum over t >= t2 of dpreatt[t][t2] * q[b][t][h][i]

Note the asymmetric loop bounds. A query at position `t` attends to keys at `t2 <= t`, so a
key at position `t2` is attended to by queries at `t >= t2`. The causal mask transposes when
you go backwards. Getting this wrong gives a `dk` that is wrong only for later positions,
which passes a small-`T` test and fails a large one. The test uses a ragged `T` on purpose.

Both write into `dqkv`, the same `(B, T, 3C)` layout as the input, at offsets `0*C` and `1*C`.

Signatures:

    __global__ void attention_bwd_datt_dv(const float* qkv, const float* att, const float* dout,
                                          float* datt, float* dqkv, int B, int T, int C, int NH)
    __global__ void attention_bwd_softmax(const float* att, const float* datt,
                                          float* dpreatt, int B, int T, int C, int NH)
    __global__ void attention_bwd_dq_dk(const float* qkv, const float* dpreatt,
                                        float* dqkv, int B, int T, int C, int NH)

`datt` and `dpreatt` are scratch buffers, both `(B, NH, T, T)`, allocated by the caller. All
three gradients accumulate into `dqkv`, so the caller zeroes it first.

## Launchers

    launch_attention_fwd(const float* qkv, float* att, float* out,
                         int B, int T, int C, int NH)
    launch_attention_bwd(const float* qkv, const float* att, const float* dout,
                         float* datt, float* dpreatt, float* dqkv,
                         int B, int T, int C, int NH)

## What the test does

1. forward vs CPU oracle, several `(B, T, C, NH)` including a ragged non-power-of-two `T`
2. that every row of `att` sums to 1 over `t2 <= t`, and is exactly 0 for `t2 > t`
3. **a causality probe**: perturb `qkv` at position `t2`, assert `out` at every position
   `t < t2` is bitwise unchanged. This is the test that catches an off-by-one in the mask, and
   it is the single most valuable test in this file. A model that peeks one token into the
   future trains beautifully and generates garbage, and you would not find it any other way.
4. `dqkv` vs oracle
5. finite-difference gradient check on `L = sum(dout * out)` over every element of `qkv`,
   through your forward kernels only, on a small shape

Run test 3 before you trust anything else.
