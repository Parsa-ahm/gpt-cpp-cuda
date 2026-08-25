#include "cuda_check.hpp"

#include <cstdio>
#include <cuda_runtime.h>

int report_cuda_devices() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    std::printf("[cuda] cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    return 0;
  }
  std::printf("[cuda] %d device(s) visible\n", count);
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, i) == cudaSuccess) {
      std::printf("[cuda]   dev %d: %s  sm_%d%d  %.1f GB  %d SMs\n",
                  i, p.name, p.major, p.minor,
                  static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                  p.multiProcessorCount);
    }
  }
  return count;
}
