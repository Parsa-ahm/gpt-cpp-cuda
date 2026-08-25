cmake_minimum_required(VERSION 3.24)
project(image_inpainting LANGUAGES CXX CUDA)

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
add_executable(inpaint
  src/main.cpp
  src/cuda_check.cu
)
set_target_properties(inpaint PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(inpaint PRIVATE src third_party)

# Warnings on for host code (keep the bar high).
target_compile_options(inpaint PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)
