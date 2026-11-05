#include "kernels.cuh"

#include <cuda_runtime.h>

namespace {
__global__ void normalize_kernel(const float* input, float* output,
                                 std::size_t spatial_pixels,
                                 std::size_t count, const float* mean,
                                 const float* stddev) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                            + threadIdx.x;
    if (index >= count) return;
    const std::size_t channel = index / spatial_pixels;
    output[index] = (input[index] - mean[channel % 3]) / stddev[channel % 3];
}
}

namespace cnn::preprocessing::cuda {
void normalize_nchw_float_cuda(const float* input, float* output,
                               std::size_t spatial_pixels,
                               std::size_t count,
                               const float* mean,
                               const float* stddev) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    normalize_kernel<<<blocks, threads>>>(input, output, spatial_pixels, count,
                                          mean, stddev);
    cudaDeviceSynchronize();
}
}  // namespace cnn::preprocessing::cuda
