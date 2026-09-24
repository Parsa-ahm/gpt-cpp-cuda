cmake_minimum_required(VERSION 3.24)
project(gpt_cpp_cuda LANGUAGES CXX CUDA)

# --- Standards ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Target GPU: RTX 2070 SUPER = Turing = sm_75 ---
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# --- Host compiler note ---
# CUDA 12.0's nvcc rejects gcc/g++ 13. If configure fails with a host-compiler
# version error, install g++-12 and configure with:
#   cmake -S . -B build -DCMAKE_CUDA_HOST_COMPILER=$(which g++-12)
# (Left unset here so a newer CUDA toolkit can use the default g++-13.)

# --- Main executable ---
# Add source files here as rungs land. Keep files small and focused.
add_executable(gpt
  src/main.cpp
  src/cuda_check.cu
)
set_target_properties(gpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(gpt PRIVATE src third_party)

# Warnings on for host code (keep the bar high).
target_compile_options(gpt PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# --- Tests ---
enable_testing()
find_package(CUDAToolkit REQUIRED)

add_executable(test_philox tests/test_philox.cu)
set_target_properties(test_philox PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_philox PRIVATE src third_party)
add_test(NAME philox COMMAND test_philox)

add_executable(test_philox_stream tests/test_philox_stream.cu)
set_target_properties(test_philox_stream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_philox_stream PRIVATE src third_party)
add_test(NAME philox_stream COMMAND test_philox_stream)

add_executable(test_philox_dist tests/test_philox_dist.cu)
set_target_properties(test_philox_dist PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_philox_dist PRIVATE src third_party)
add_test(NAME philox_dist COMMAND test_philox_dist)

add_executable(test_residual tests/test_residual.cu)
set_target_properties(test_residual PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_residual PRIVATE src third_party)
add_test(NAME residual COMMAND test_residual)

add_executable(test_cublas_layout tests/test_cublas_layout.cu)
set_target_properties(test_cublas_layout PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_cublas_layout PRIVATE src third_party)
target_link_libraries(test_cublas_layout PRIVATE CUDA::cublas)
add_test(NAME cublas_layout COMMAND test_cublas_layout)

add_executable(test_linear tests/test_linear.cu)
set_target_properties(test_linear PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_linear PRIVATE src third_party)
target_link_libraries(test_linear PRIVATE CUDA::cublas)
add_test(NAME linear COMMAND test_linear)

add_executable(test_gelu tests/test_gelu.cu)
set_target_properties(test_gelu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_gelu PRIVATE src third_party)
add_test(NAME gelu COMMAND test_gelu)

add_executable(test_gemm tests/test_gemm.cu)
set_target_properties(test_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_gemm PRIVATE src third_party)
add_test(NAME gemm COMMAND test_gemm)

# Benchmark: links cuBLAS as a PERF REFERENCE only (never in the model path).
add_executable(bench_gemm tests/bench_gemm.cu)
set_target_properties(bench_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(bench_gemm PRIVATE src third_party)
target_link_libraries(bench_gemm PRIVATE CUDA::cublas)

# --- Rung 2 remaining ops ---
add_executable(test_layernorm tests/test_layernorm.cu)
set_target_properties(test_layernorm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_layernorm PRIVATE src third_party)
add_test(NAME layernorm COMMAND test_layernorm)

add_executable(test_embedding tests/test_embedding.cu)
set_target_properties(test_embedding PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_embedding PRIVATE src third_party)
add_test(NAME embedding COMMAND test_embedding)

add_executable(test_attention tests/test_attention.cu)
set_target_properties(test_attention PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_attention PRIVATE src third_party)
add_test(NAME attention COMMAND test_attention)

add_executable(test_cross_entropy tests/test_cross_entropy.cu)
set_target_properties(test_cross_entropy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_cross_entropy PRIVATE src third_party)
add_test(NAME cross_entropy COMMAND test_cross_entropy)

add_executable(test_adamw tests/test_adamw.cu)
set_target_properties(test_adamw PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_adamw PRIVATE src third_party)
add_test(NAME adamw COMMAND test_adamw)
