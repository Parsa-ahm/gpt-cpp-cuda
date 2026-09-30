#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include "cuda_check.hpp"
struct Device_Buffer {
    float* ptr = nullptr;
    size_t n = 0;
    Device_Buffer() = default;

    Device_Buffer(size_t count) {
        make(count);
    }

    void make(size_t count) {
        n = count;
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float)));
    }
    void upload(float* cpu) {
        cudaMemcpy(ptr, cpu, n * sizeof(float), cudaMemcpyHostToDevice);
    }
    void download(float* cpu) {
        cudaMemcpy(cpu, ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void zero() {
        CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float)));
    }
    void free_it() {
        CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
        n = 0;
    }
};
