#include "image_reader.hpp"

#include <cstring>
#include <fstream>
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "tensor_builder.hpp"

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
    cnn::preprocessing::TensorSpec spec;
    spec.width = width;
    spec.height = height;
    try {
        return cnn::preprocessing::write_normalized_tensor(input, output, spec, error);
    } catch (const cv::Exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace cnn::ingestion
