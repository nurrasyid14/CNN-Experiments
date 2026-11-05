#pragma once

#include <cstddef>

namespace cnn::preprocessing::cuda {

void normalize_nchw_float_cuda(const float* input, float* output,
                               std::size_t spatial_pixels,
                               std::size_t count,
                               const float* mean,
                               const float* stddev);

}  // namespace cnn::preprocessing::cuda
