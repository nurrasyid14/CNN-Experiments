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

}  // namespace cnn::ingestion
