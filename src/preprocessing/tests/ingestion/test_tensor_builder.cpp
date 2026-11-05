#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "tensor_builder.hpp"

using cnn::preprocessing::TensorSpec;
using cnn::preprocessing::build_normalized_tensor;

TEST(TensorBuilder, ProducesRgbChwFloatTensor) {
    cv::Mat image(2, 2, CV_8UC3,
                  cv::Scalar(255, 128, 0));
    TensorSpec spec;
    spec.width = 2;
    spec.height = 2;
    spec.mean = {0.0f, 0.0f, 0.0f};
    spec.stddev = {1.0f, 1.0f, 1.0f};

    std::vector<float> tensor;
    std::string error;
    ASSERT_TRUE(build_normalized_tensor(image, spec, tensor, error)) << error;
    ASSERT_EQ(tensor.size(), 12U);

    EXPECT_FLOAT_EQ(tensor[0], 0.0f);
    EXPECT_FLOAT_EQ(tensor[4], 128.0f / 255.0f);
    EXPECT_FLOAT_EQ(tensor[8], 1.0f);
}

TEST(TensorBuilder, RejectsZeroStandardDeviation) {
    cv::Mat image(1, 1, CV_8UC3, cv::Scalar(0, 0, 0));
    TensorSpec spec;
    spec.width = 1;
    spec.height = 1;
    spec.stddev = {0.0f, 1.0f, 1.0f};

    std::vector<float> tensor;
    std::string error;
    EXPECT_FALSE(build_normalized_tensor(image, spec, tensor, error));
    EXPECT_FALSE(error.empty());
}
