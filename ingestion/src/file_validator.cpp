#include "file_validator.hpp"

#include <array>
#include <system_error>

#include "text_utils.hpp"

namespace fs = std::filesystem;

namespace cnn::ingestion {

bool is_supported_extension(const fs::path& p) {
    static constexpr std::array<const char*, 8> kExt = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"};
    const std::string ext = to_lower(to_utf8(p.extension()));
    for (const char* e : kExt) {
        if (ext == e) return true;
    }
    return false;
}

std::optional<std::string> validate_file(const fs::path& p) {
    std::error_code ec;
    if (!fs::exists(p, ec) || ec) return "file does not exist";
    if (!fs::is_regular_file(p, ec) || ec) return "not a regular file";
    const auto size = fs::file_size(p, ec);
    if (ec) return "cannot read file size: " + ec.message();
    if (size == 0) return "file is empty";
    return std::nullopt;
}

}  // namespace cnn::ingestion
