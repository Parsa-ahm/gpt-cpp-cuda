#include "cuda_check.hpp"

#include <cstdio>

// Entry point. Grows into a small CLI dispatcher as rungs land
// (train / generate / bench). Rung 0: prove the build works
// and the GPU is visible.
int main() {
  std::printf("gpt-cpp-cuda -- GPT training + inference engine, C++/CUDA, no ML framework\n");
  int devices = report_cuda_devices();
  if (devices == 0) {
    std::printf("no CUDA device -- CPU-only paths still run, GPU engine will not.\n");
  }
  return 0;
}
