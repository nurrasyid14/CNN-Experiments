#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "cuda_backend.hpp"

TEST(CudaBackend, NormalizationMatchesCpuWhenAvailable) {
    if (!cnn::preprocessing::cuda_available()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    const std::vector<float> input(6, 128.0f / 255.0f);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stddev[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> output;
    const auto result = cnn::preprocessing::normalize_nchw_float(
        input, input.size(), 2, mean, stddev, output);

    ASSERT_TRUE(result.available) << result.error;
    ASSERT_EQ(output.size(), input.size());
    for (std::size_t index = 0; index < output.size(); ++index) {
        const std::size_t channel = index / 2;
        const float expected = (input[index] - mean[channel]) / stddev[channel];
        EXPECT_NEAR(output[index], expected, 1.0e-5f);
    }
}
