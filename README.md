# gpt-cpp-cuda

A **GPT training and inference engine written from scratch in C++ and CUDA**. No PyTorch, no JAX.
Forward and backward kernels for every op are hand-written; matmuls go through cuBLAS, and the
same matmul is also implemented by hand and benchmarked against it.

What it does when finished:
- trains a ~10M-parameter GPT on a small corpus on a single consumer GPU
- loads OpenAI's GPT-2 124M weights and matches HuggingFace logits, then generates text
- ships a fused attention kernel and an SGEMM ladder with benchmarks and a roofline

See [`CHARTER.md`](./CHARTER.md) for purpose, thesis, boundaries, and the build ladder. It is the
single source of truth.

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
