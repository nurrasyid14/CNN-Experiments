#include "cuda_backend.hpp"

#include "kernels.cuh"

#include <cuda_runtime.h>

namespace cnn::preprocessing {

namespace {
#define CNN_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        const cudaError_t status = (call);                                     \
        if (status != cudaSuccess) {                                           \
            return false;                                                      \
        }                                                                      \
    } while (false)
}

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaNormalizationResult normalize_nchw_float(
    const std::vector<float>& input, std::size_t count,
    std::size_t spatial_size, const float mean[3],
    const float stddev[3], std::vector<float>& output) {
    CudaNormalizationResult result;
    result.available = cuda_available();
    if (!result.available) {
        result.error = "CUDA device is unavailable";
        return result;
    }
    if (count != input.size() || spatial_size == 0 ||
        count % spatial_size != 0 || count / spatial_size != 3) {
        result.error = "invalid NCHW normalization dimensions";
        return result;
    }
    float *device_input = nullptr;
    float *device_output = nullptr;
    float *device_mean = nullptr;
    float *device_stddev = nullptr;
    const std::size_t bytes = count * sizeof(float);
    if (cudaMalloc(&device_input, bytes) != cudaSuccess ||
        cudaMalloc(&device_output, bytes) != cudaSuccess ||
        cudaMalloc(&device_mean, 3 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_stddev, 3 * sizeof(float)) != cudaSuccess) {
        result.error = cudaGetErrorString(cudaGetLastError());
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(device_mean);
        cudaFree(device_stddev);
        return result;
    }
    if (cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(device_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(device_stddev, stddev, 3 * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        result.error = cudaGetErrorString(cudaGetLastError());
    } else {
        cuda::normalize_nchw_float_cuda(device_input, device_output, spatial_size,
                                        count, device_mean, device_stddev);
        const cudaError_t kernel_status = cudaGetLastError();
        if (kernel_status != cudaSuccess) {
            result.error = cudaGetErrorString(kernel_status);
            cudaFree(device_input);
            cudaFree(device_output);
            cudaFree(device_mean);
            cudaFree(device_stddev);
            return result;
        }
        output.resize(count);
        if (cudaMemcpy(output.data(), device_output, bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            result.error = cudaGetErrorString(cudaGetLastError());
        } else {
            result.available = true;
        }
    }
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mean);
    cudaFree(device_stddev);
    return result;
}

}  // namespace cnn::preprocessing
