#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cnn::ingestion {

struct ImageInfo {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::string format;              // from file signature, not from the extension
    std::uintmax_t file_size = 0;    // bytes
};

struct ReadResult {
    std::optional<ImageInfo> info;   // set on success
    std::string error;               // set on failure
};

// "JPEG", "PNG", "BMP", "TIFF", "WEBP" or "UNKNOWN", from the leading bytes.
std::string detect_format(const std::vector<unsigned char>& bytes);

// Fully decodes the image (so corrupt files are detected) and reports its metadata.
// Never throws; failures are reported in ReadResult::error.
ReadResult read_image_info(const std::filesystem::path& p);

// Writes a normalized CHW float32 tensor. If channels are 1/3/4, OpenCV grayscale
// values are replicated to RGB; output dimensions are always 3 x height x width.
bool write_normalized_tensor(const std::filesystem::path& input,
                             const std::filesystem::path& output,
                             int width,
                             int height,
                             std::string& error);

}  // namespace cnn::ingestion
