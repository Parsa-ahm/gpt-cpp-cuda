# image-inpainting-cpp-cuda

A **score-based generative image model built from first principles in C++/CUDA** — hand-written
reverse-mode autodiff, a trained score network (denoising score matching), annealed Langevin
sampling, and hand-written CUDA sampling kernels. Demonstrated on **image inpainting**: give it an
image with a region removed, it fills the region back in.

**Zero ML frameworks.** No PyTorch, JAX, TensorFlow, cuDNN, or cuBLAS in the learning core. The
autodiff, model, training loop, PRNG, and sampling kernels are all hand-written.

See [`CHARTER.md`](./CHARTER.md) for the full purpose, thesis, non-goals, and build ladder — read
it before contributing (it is the single source of truth).

## Status

Rung 0 (scaffold). Build system + GPU sanity check in place; PRNG in progress.

## Requirements

- NVIDIA GPU (developed on an RTX 2070 SUPER, `sm_75`)
- CUDA toolkit (`nvcc`) + CMake ≥ 3.24 + a C++17 host compiler
- Note: CUDA 12.0's `nvcc` rejects gcc/g++ 13 — install `g++-12` and configure with
  `-DCMAKE_CUDA_HOST_COMPILER=$(which g++-12)` if you hit a host-compiler version error.

## Build

```sh
cmake -S . -B build            # add -DCMAKE_CUDA_HOST_COMPILER=$(which g++-12) if needed
cmake --build build -j
./build/inpaint                # Rung 0: prints banner + reports the GPU
```
