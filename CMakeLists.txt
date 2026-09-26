cmake_minimum_required(VERSION 3.24)
project(gpt_cpp_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RTX 2070 SUPER = Turing = sm_75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# CUDA 12.0's nvcc rejects g++-13. If configure fails on the host compiler:
#   cmake -S . -B build -DCMAKE_CUDA_HOST_COMPILER=$(which g++-12)

find_package(CUDAToolkit REQUIRED)
enable_testing()

add_executable(gpt
  src/main.cpp
  src/core/cuda_check.cu
)
set_target_properties(gpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(gpt PRIVATE src)
target_link_libraries(gpt PRIVATE CUDA::cublas)
target_compile_options(gpt PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

# One target per tests/test_<name>.cu, registered with ctest as <name>.
# Uncomment an op once its implementation lands; the harnesses are already written
# for attention, cross_entropy and adamw.
set(TESTS
  philox philox_stream philox_dist
  gemm cublas_layout
  residual gelu linear layernorm embedding
  # attention
  # cross_entropy adamw
)

foreach(name IN LISTS TESTS)
  add_executable(test_${name} tests/test_${name}.cu)
  set_target_properties(test_${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(test_${name} PRIVATE src)
  target_link_libraries(test_${name} PRIVATE CUDA::cublas)
  add_test(NAME ${name} COMMAND test_${name})
endforeach()

# Hand-written SGEMM vs cuBLAS. Not a test; run it directly.
add_executable(bench_gemm tests/bench_gemm.cu)
set_target_properties(bench_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(bench_gemm PRIVATE src)
target_link_libraries(bench_gemm PRIVATE CUDA::cublas)
