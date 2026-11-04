#pragma once
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

namespace cnn::ingestion {

// UTF-8 string with forward slashes; safe for JSON on every platform.
inline std::string to_utf8(const std::filesystem::path& p) {
    const auto s = p.generic_u8string();
    return std::string(s.begin(), s.end());
}

// Inverse of to_utf8: interprets a UTF-8 command-line string as a path.
inline std::filesystem::path from_utf8(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace cnn::ingestion
