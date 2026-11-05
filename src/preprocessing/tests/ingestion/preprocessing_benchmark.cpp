#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda_backend.hpp"

int main() {
    constexpr std::size_t spatial_pixels = 224 * 224;
    constexpr std::size_t count = 3 * spatial_pixels;
    constexpr int iterations = 100;
    std::vector<float> input(count, 128.0f / 255.0f);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stddev[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> cpu_output(count);

    if (!cnn::preprocessing::cuda_available()) {
        std::cerr << "CUDA preprocessing benchmark unavailable\n";
        return 2;
    }

    const auto cpu_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t index = 0; index < count; ++index) {
            const std::size_t channel = index / spatial_pixels;
            cpu_output[index] = (input[index] - mean[channel]) / stddev[channel];
        }
    }
    const auto cpu_end = std::chrono::steady_clock::now();

    std::vector<float> gpu_output;
    const auto result = cnn::preprocessing::normalize_nchw_float(
        input, count, spatial_pixels, mean, stddev, gpu_output);
    if (!result.available) {
        std::cerr << "CUDA preprocessing benchmark failed: " << result.error << "\n";
        return 1;
    }

    const auto gpu_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto iteration_result = cnn::preprocessing::normalize_nchw_float(
            input, count, spatial_pixels, mean, stddev, gpu_output);
        if (!iteration_result.available) {
            std::cerr << "CUDA preprocessing benchmark failed: "
                      << iteration_result.error << "\n";
            return 1;
        }
    }
    const auto gpu_end = std::chrono::steady_clock::now();

    const double cpu_ms = std::chrono::duration<double, std::milli>(
        cpu_end - cpu_start).count();
    const double gpu_ms = std::chrono::duration<double, std::milli>(
        gpu_end - gpu_start).count();
    std::cout << "iterations=" << iterations << "\n"
              << "cpu_ms=" << cpu_ms << "\n"
              << "cuda_end_to_end_ms=" << gpu_ms << "\n"
              << "cpu_images_per_second=" << iterations * 1000.0 / cpu_ms << "\n"
              << "cuda_images_per_second=" << iterations * 1000.0 / gpu_ms << "\n";
    return 0;
}
