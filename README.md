# gpt-cpp-cuda

A GPT training and inference engine written from scratch in C++ and CUDA. No PyTorch, no JAX.
Forward and backward kernels for every op are hand-written; matmuls go through cuBLAS, and the
same matmul is also implemented by hand and benchmarked against it.

What it does when finished:
- trains a ~10M-parameter GPT on a small corpus on a single consumer GPU
- loads OpenAI's GPT-2 124M weights and matches HuggingFace logits, then generates text
- ships a fused attention kernel and an SGEMM ladder with benchmarks and a roofline

The from-scratch line is *no ML framework*, not *no CUDA libraries*: cuBLAS is allowed for matmuls,
any autograd or tensor library is not. FP32, single GPU.

## Status

| Rung | State |
|---|---|
| 0 Scaffold + Philox PRNG | done |
| 1 SGEMM ladder (naive / tiled / register-blocked, ~40% cuBLAS) | done |
| 2 Ops fwd+bwd, gradient-checked | in progress |
| 3 Train small GPT | |
| 4 GPT-2 124M weights, match HF logits | |
| 5 Fused attention + benchmarks + roofline | |
| 6 Writeup | |

## Known cost

Attention materialises `att`, `B * NH * T * T` floats. At `B=4, NH=6, T=256` that is 6 MB; at
GPT-2 124M with `T=1024, NH=12` it is 48 MB per batch element. That is the memory wall
FlashAttention removes by never materialising it. Rung 5 builds the fused version and benchmarks
it against this one.

## Layout

    src/core/    allocation, error checking, cuBLAS handle, GEMM (hand-written + cuBLAS)
    src/nn/      transformer ops, forward and backward
    src/rng/     Philox counter-based PRNG
    tests/       one test_<op>.cu per op, gradient checks against finite differences
    data/        training corpus and GPT-2 weights (downloaded, not committed)

## Requirements

- NVIDIA GPU (developed on an RTX 2070 SUPER, `sm_75`), CUDA 12.x with cuBLAS
- CMake >= 3.24, C++17 host compiler (CUDA 12.0 needs `g++-12`)

## Build and test

```sh
cmake -S . -B build -DCMAKE_CUDA_HOST_COMPILER=$(which g++-12)
cmake --build build -j
ctest --test-dir build
./build/bench_gemm      # hand-written SGEMM vs cuBLAS
```
