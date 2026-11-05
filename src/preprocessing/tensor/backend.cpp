#include "cuda_backend.hpp"

namespace cnn::preprocessing {

ComputeBackend resolve_backend(ComputeBackend requested,
                               bool cuda_compiled_and_available) {
    if (requested == ComputeBackend::CPU) return ComputeBackend::CPU;
    if (cuda_compiled_and_available) return ComputeBackend::CUDA;
    if (requested == ComputeBackend::CUDA) return ComputeBackend::CUDA;
    return ComputeBackend::CPU;
}

}  // namespace cnn::preprocessing
