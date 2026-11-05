#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cnn::preprocessing {

enum class ComputeBackend {
    CPU,
    CUDA,
    AUTO
};

ComputeBackend resolve_backend(ComputeBackend requested,
                               bool cuda_compiled_and_available);

struct CudaNormalizationResult {
    bool available = false;
    std::string error;
};

bool cuda_available();

// Normalizes NCHW planar float data already scaled to [0, 1].
CudaNormalizationResult normalize_nchw_float(
    const std::vector<float>& input, std::size_t count,
    std::size_t spatial_size, const float mean[3],
    const float stddev[3], std::vector<float>& output);

}  // namespace cnn::preprocessing
