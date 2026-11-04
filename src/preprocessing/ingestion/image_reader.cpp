#include "image_reader.hpp"

#include <cstring>
#include <fstream>
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace cnn::ingestion {

namespace {
bool matches(const std::vector<unsigned char>& b, std::size_t offset, const char* sig, std::size_t n) {
    return b.size() >= offset + n && std::memcmp(b.data() + offset, sig, n) == 0;
}
}  // namespace

std::string detect_format(const std::vector<unsigned char>& b) {
    if (matches(b, 0, "\xFF\xD8\xFF", 3)) return "JPEG";
    if (matches(b, 0, "\x89PNG\r\n\x1a\n", 8)) return "PNG";
    if (matches(b, 0, "BM", 2)) return "BMP";
    if (matches(b, 0, "II*\0", 4) || matches(b, 0, "MM\0*", 4)) return "TIFF";
    if (matches(b, 0, "RIFF", 4) && matches(b, 8, "WEBP", 4)) return "WEBP";
    return "UNKNOWN";
}

ReadResult read_image_info(const fs::path& p) {
    ReadResult result;

    // Read bytes ourselves and use imdecode: avoids imread's trouble with non-ASCII Windows paths.
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        result.error = "cannot open file";
        return result;
    }

    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());

    ImageInfo info;
    info.file_size = bytes.size();
    info.format = detect_format(bytes);
    if (info.format == "UNKNOWN") {
        result.error = "unrecognized image signature";
        return result;
    }

    cv::Mat image;
    try {
        image = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception& e) {
        result.error = std::string("decode failed: ") + e.what();
        return result;
    }
    if (image.empty()) {
        result.error = "decode failed: corrupted or unsupported image data";
        return result;
    }

    info.width = image.cols;
    info.height = image.rows;
    info.channels = image.channels();
    result.info = info;
    return result;
}

bool write_normalized_tensor(const fs::path& input, const fs::path& output,
                             int width, int height, std::string& error) {
    std::ifstream in(input, std::ios::binary);
    if (!in) {
        error = "cannot open file";
        return false;
    }
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
    cv::Mat image;
    try {
        image = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            error = "decode failed: corrupted or unsupported image data";
            return false;
        }
        if (image.rows != height || image.cols != width) {
            cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
        }
        if (image.channels() == 1) cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
        else if (image.channels() == 4) cv::cvtColor(image, image, cv::COLOR_BGRA2RGB);
        else if (image.channels() == 3) cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        else {
            error = "unsupported channel count";
            return false;
        }
        image.convertTo(image, CV_32FC3, 1.0 / 255.0);
        if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
        std::ofstream out(output, std::ios::binary | std::ios::trunc);
        if (!out) {
            error = "cannot open tensor output";
            return false;
        }
        const char magic[8] = {'C', 'N', 'N', 'T', 'N', 'S', '1', '\0'};
        const std::uint32_t dims[3] = {3, static_cast<std::uint32_t>(height),
                                       static_cast<std::uint32_t>(width)};
        out.write(magic, sizeof(magic));
        out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
        static const float mean[3] = {0.485f, 0.456f, 0.406f};
        static const float stddev[3] = {0.229f, 0.224f, 0.225f};
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < image.rows; ++row) {
                const cv::Vec3f* pixels = image.ptr<cv::Vec3f>(row);
                for (int column = 0; column < image.cols; ++column) {
                    const float value = (pixels[column][channel] - mean[channel]) / stddev[channel];
                    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
                }
            }
        }
        return static_cast<bool>(out);
    } catch (const cv::Exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace cnn::ingestion
