#pragma once
#include <cstddef>
#include <cuda_runtime.h>
// ============================================================================
// SPEC — device memory buffer.   YOU implement all of this.
//
// A thin owner of a chunk of GPU global memory (float). Host-side helper — it
// launches no kernels, it just wraps cudaMalloc / cudaMemcpy / cudaFree so the
// rest of the code (and the tests) can move float arrays on and off the device
// without repeating the raw calls.
//
// WHAT it must be able to do (semantics — you choose the interface):
//   1. allocate  : reserve `n` floats on the GPU. store the device pointer + n.
//   2. upload    : copy n floats host -> device (cudaMemcpyHostToDevice).
//   3. download  : copy n floats device -> host (cudaMemcpyDeviceToHost).
//   4. free      : release the GPU memory. every alloc pairs with exactly one
//                  free (no GC — you own the lifetime).
//   5. expose the raw `float*` device pointer so kernels can be launched on it.
//
// DESIGN CHOICE (yours): a struct that frees in its destructor (RAII), or plain
// functions you call by hand. RAII is safer (no leaks on early return); plain is
// simpler. Either is fine — say which so the test matches.
//
// RULES:
//   - bytes = n * sizeof(float). cudaMalloc's first arg is the ADDRESS of your
//     pointer (float** via &ptr) — it writes the allocation into your pointer.
//   - check every CUDA call for errors (see cuda_check in src/).
//   - if RAII: forbid copying (two owners -> double free) or you'll get bugs.
// ============================================================================

// ---- your implementation below ----

struct Device_Buffer {
    float* ptr = nullptr;
    int n = 0;

    Device_Buffer(int count) {
        make(count);
    }

    void make(int count) {
        n = count;
        cudaMalloc(&ptr, n * sizeof(float));
    }
    void upload(float* cpu) {
        cudaMemcpy(ptr, cpu, n * sizeof(float), cudaMemcpyHostToDevice);
    }
    void download(float* cpu) {
        cudaMemcpy(cpu, ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void free_it() {
        cudaFree(ptr);
    }
};
