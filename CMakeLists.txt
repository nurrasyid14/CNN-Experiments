cmake_minimum_required(VERSION 3.21)
project(CNNClassifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CNN_BUILD_TESTS "Build C++ tests" ON)
option(CNN_ENABLE_CUDA "Enable optional native CUDA preprocessing" OFF)
option(ENABLE_CUDA "Enable CUDA preprocessing backend" OFF)
if(ENABLE_CUDA)
  set(CNN_ENABLE_CUDA ON CACHE BOOL "Enable optional native CUDA preprocessing" FORCE)
endif()

if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/std:c++20>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
else()
  add_compile_options(-Wall -Wextra)
endif()

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgcodecs)
find_package(nlohmann_json CONFIG REQUIRED)

if(CNN_ENABLE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures" FORCE)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(cnn_preprocessing_cuda STATIC
      src/preprocessing/cuda/cuda_backend.cpp
      src/preprocessing/cuda/normalization.cu)
    target_include_directories(cnn_preprocessing_cuda PUBLIC
      src/preprocessing/include src/preprocessing/cuda)
    target_link_libraries(cnn_preprocessing_cuda PUBLIC CUDA::cudart)
    add_library(cnn_cuda_backend ALIAS cnn_preprocessing_cuda)
  else()
    message(WARNING "CNN_ENABLE_CUDA=ON but a CUDA compiler was not found; CPU preprocessing will be used.")
  endif()
endif()

# Block 0: toolchain sanity check
add_executable(env_check scripts/env_check.cpp)

# Block 1
add_subdirectory(src/preprocessing)

# Enabled as each block is implemented:
# add_subdirectory(processing)   # Block 2
# add_subdirectory(dataset)      # Block 3

if(CNN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(src/preprocessing/tests/ingestion)
endif()
