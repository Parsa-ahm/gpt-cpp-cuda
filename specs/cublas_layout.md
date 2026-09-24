# Step 1 of linear: a row-major cuBLAS wrapper

File to write: `src/core/gemm_cublas.cuh`
Test: `tests/test_cublas_layout.cu` (target `test_cublas_layout`)

This step has no machine learning in it. It exists because cuBLAS stores matrices
the opposite way round from you, and the charter lists that confusion as risk R2:
a classic multi-day time sink. So we pin it down with its own test before linear,
attention, or anything else depends on it.

## The problem

You store matrices **row-major**: element `[r][c]` of an `r x c` matrix lives at
`base[r*cols + c]`. Consecutive elements of a row are next to each other in memory.
That is what your Rung 1 SGEMM assumes and what C arrays do.

cuBLAS is a Fortran API underneath. It stores matrices **column-major**: element
`[r][c]` lives at `base[c*rows + r]`. Consecutive elements of a *column* are next
to each other.

Nobody converts anything. The bytes stay put. The two libraries just disagree
about how to read them.

## The one fact that solves it

> A matrix stored row-major with shape `(r, c)`, read as column-major, **is its
> own transpose**: shape `(c, r)`, with leading dimension `c`.

Nothing is copied or moved. Same bytes, different interpretation.

Example. Row-major 2x3 holding `[[1,2,3],[4,5,6]]` is laid out `1 2 3 4 5 6`.
Read that column-major as a 3x2 and you get `[[1,4],[2,5],[3,6]]`, which is the
transpose. Confirm it on paper before you write code.

## Deriving the call

You want, in row-major terms:

    C = opA(A) @ opB(B)          C is (M, N)

where `opA` is either nothing or a transpose, same for `opB`.

cuBLAS will hand you a column-major result. The bytes it writes, read back as
row-major `(M,N)`, are what you want. And column-major `C` is row-major `C`
transposed, so what cuBLAS must actually compute is:

    C^T = (opA(A) @ opB(B))^T = opB(B)^T @ opA(A)^T

Read that right-hand side. **B comes first now.** That is the whole trick: you
hand cuBLAS your operands in the opposite order.

Work each operand through. Take `op_b == N`, so `B` is row-major `(K,N)`. Its
column-major view is `B^T`, shape `(N,K)`. What the formula needs is `opB(B)^T`,
which for `op_b == N` is just `B^T`. Those are the same thing, so the flag you
pass cuBLAS for that operand is `OP_N`, unchanged.

Do the same for `op_b == T` and for both cases of `A` and you find the same
result every time: **the transpose flags do not change.** They pass straight
through. Only the operand order and the dimensions move.

## The recipe

    cublasSgemm(cublas_handle(),
                op_b_flag,          // B's flag, unchanged
                op_a_flag,          // A's flag, unchanged
                N, M, K,            // note: N and M swapped, K stays
                &alpha,
                B, ldb,             // B FIRST
                A, lda,             // A second
                &beta,
                C, N);              // ldc = N

Leading dimensions are always **the number of columns the matrix has as you
stored it**, which for row-major is its row length:

    lda = (op_a == N) ? K : M
    ldb = (op_b == N) ? N : K
    ldc = N

Four rules total: swap the operands, swap M and N, keep the flags, leading
dimension is the stored row length.

## alpha and beta are free accumulation

cuBLAS computes `C = alpha * (A@B) + beta * C`.

So `beta = 0.0f` overwrites `C`, and `beta = 1.0f` **adds into** `C`. That is
`+=` at no cost, no extra kernel, no extra pass over memory. The backward pass
needs exactly this for weight gradients, which is why the wrapper exposes it.

## What to write

One function in `src/core/gemm_cublas.cuh`:

    inline void sgemm_rm(bool trans_a, bool trans_b,
                         int M, int N, int K,
                         float alpha,
                         const float* dA, const float* dB,
                         float beta, float* dC)

Computes row-major `C(M,N) = alpha * opA(A) @ opB(B) + beta * C`, where
`A` is `(M,K)` if `!trans_a` else `(K,M)`, and `B` is `(K,N)` if `!trans_b`
else `(N,K)`.

Notes:

- `rm` is for row-major. `inline` because it lives in a header.
- The flag values are `CUBLAS_OP_N` and `CUBLAS_OP_T`. Pick with a ternary:
  `trans_a ? CUBLAS_OP_T : CUBLAS_OP_N`.
- `alpha` and `beta` are passed **by address** (`&alpha`), not by value. cuBLAS
  accepts host or device pointers there, so the API takes a pointer. Forgetting
  the `&` is a compile error, which is the good outcome.
- Wrap the call in `CUBLAS_CHECK(...)` from `cublas_ctx.hpp`.
- Get the handle from `cublas_handle()`. Do not create one yourself.
- No `CUDA_CHECK_KERNEL()` here. cuBLAS is async and the test synchronizes when
  it copies results back. `CUBLAS_CHECK` on the return value is the check you need.

## What the test does

All four transpose combinations, several shapes including non-square and
non-multiples of anything, against a CPU oracle. Then `beta = 1.0f` against a
pre-filled `C` to confirm accumulation.

If you get a transpose wrong the result is usually not garbage, it is a plausible
matrix of the wrong values, and at 1e-3 tolerance the test will still catch it.
That is why this test exists separately from linear: when it fails you know the
bug is in the layout, not in the calculus.
