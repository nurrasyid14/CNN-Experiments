#include "tensor_builder.hpp"

#include <fstream>
#include <iterator>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "binary_export.hpp"

namespace cnn::preprocessing {

namespace {
bool prepare_image(cv::Mat image, const TensorSpec& spec, cv::Mat& prepared,
                   std::string& error) {
    if (image.empty()) {
        error = "decode failed: corrupted or unsupported image data";
        return false;
    }
    if (spec.width <= 0 || spec.height <= 0) {
        error = "tensor dimensions must be positive";
        return false;
    }
    cv::resize(image, image, cv::Size(spec.width, spec.height), 0, 0,
               cv::INTER_AREA);
    if (image.channels() == 1) {
        cv::cvtColor(image, prepared, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, prepared, cv::COLOR_BGRA2RGB);
    } else if (image.channels() == 3) {
        cv::cvtColor(image, prepared, cv::COLOR_BGR2RGB);
    } else {
        error = "unsupported channel count";
        return false;
    }
    return true;
}
}  // namespace

bool build_normalized_tensor(const cv::Mat& decoded, const TensorSpec& spec,
                             std::vector<float>& tensor, std::string& error) {
    cv::Mat prepared;
    if (!prepare_image(decoded, spec, prepared, error)) return false;
    if (spec.mean.size() != 3 || spec.stddev.size() != 3) {
        error = "normalization requires three mean and standard-deviation values";
        return false;
    }
    prepared.convertTo(prepared, CV_32FC3, 1.0 / 255.0);
    tensor.resize(static_cast<std::size_t>(3 * spec.width * spec.height));
    for (int channel = 0; channel < 3; ++channel) {
        if (spec.stddev[channel] == 0.0f) {
            error = "normalization standard deviation must not be zero";
            return false;
        }
        float* destination = tensor.data() + static_cast<std::size_t>(
            channel * spec.width * spec.height);
        for (int row = 0; row < prepared.rows; ++row) {
            const cv::Vec3f* pixels = prepared.ptr<cv::Vec3f>(row);
            for (int column = 0; column < prepared.cols; ++column) {
                destination[static_cast<std::size_t>(row * prepared.cols + column)] =
                    (pixels[column][channel] - spec.mean[channel]) /
                    spec.stddev[channel];
            }
        }
    }
    return true;
}

bool decode_and_build_tensor(const std::filesystem::path& input,
                             const TensorSpec& spec,
                             std::vector<float>& tensor, std::string& error) {
    std::ifstream in(input, std::ios::binary);
    if (!in) {
        error = "cannot open file";
        return false;
    }
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
    cv::Mat decoded = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
    return build_normalized_tensor(decoded, spec, tensor, error);
}

bool write_normalized_tensor(const std::filesystem::path& input,
                             const std::filesystem::path& output,
                             const TensorSpec& spec, std::string& error) {
    std::vector<float> tensor;
    if (!decode_and_build_tensor(input, spec, tensor, error)) return false;
    return export_binary_tensor(
        output, {3U, static_cast<unsigned int>(spec.height),
                 static_cast<unsigned int>(spec.width)},
        tensor, error);
}

}  // namespace cnn::preprocessing
