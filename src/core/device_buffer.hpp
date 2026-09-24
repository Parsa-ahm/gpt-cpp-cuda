#pragma once
#include <cstddef>
#include <cuda_runtime.h>

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
