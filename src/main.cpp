#include "cuda_check.hpp"

#include <cstdio>

// Entry point. Grows into a small CLI dispatcher as rungs land
// (train / sample / inpaint / metrics). Rung 0: prove the build works
// and the GPU is visible.
int main() {
  std::printf("image-inpainting-cpp-cuda -- from-scratch score-based generative model\n");
  int devices = report_cuda_devices();
  if (devices == 0) {
    std::printf("no CUDA device -- CPU-only paths still run, GPU sampler will not.\n");
  }
  return 0;
}
