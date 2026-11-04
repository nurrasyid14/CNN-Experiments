cmake_minimum_required(VERSION 3.21)
project(CNNClassifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CNN_BUILD_TESTS "Build C++ tests" ON)

if(MSVC)
  add_compile_options(/W4 /utf-8 /std:c++20 /Zc:__cplusplus)
else()
  add_compile_options(-Wall -Wextra)
endif()

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgcodecs)
find_package(nlohmann_json CONFIG REQUIRED)

# Block 0: toolchain sanity check
add_executable(env_check scripts/env_check.cpp)

# Block 1
add_subdirectory(preprocessing)

# Enabled as each block is implemented:
# add_subdirectory(processing)   # Block 2
# add_subdirectory(dataset)      # Block 3

if(CNN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(preprocessing/tests/ingestion)
endif()
