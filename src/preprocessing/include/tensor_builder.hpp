#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace cnn::preprocessing {

struct TensorSpec {
    int width = 224;
    int height = 224;
    std::vector<float> mean{0.485f, 0.456f, 0.406f};
    std::vector<float> stddev{0.229f, 0.224f, 0.225f};
};

bool build_normalized_tensor(const cv::Mat& decoded, const TensorSpec& spec,
                             std::vector<float>& tensor, std::string& error);

bool decode_and_build_tensor(const std::filesystem::path& input,
                             const TensorSpec& spec,
                             std::vector<float>& tensor, std::string& error);

bool write_normalized_tensor(const std::filesystem::path& input,
                             const std::filesystem::path& output,
                             const TensorSpec& spec,
                             std::string& error);

}  // namespace cnn::preprocessing
