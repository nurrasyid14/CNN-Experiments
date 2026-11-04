#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "file_validator.hpp"

namespace fs = std::filesystem;
using namespace cnn::ingestion;

TEST(FileValidator, ExtensionsAreCaseInsensitive) {
    EXPECT_TRUE(is_supported_extension("a.JPG"));
    EXPECT_TRUE(is_supported_extension("a.png"));
    EXPECT_FALSE(is_supported_extension("a.txt"));
    EXPECT_FALSE(is_supported_extension("noext"));
}

TEST(FileValidator, RejectsMissingFile) {
    EXPECT_TRUE(validate_file("definitely/not/here.jpg").has_value());
}

TEST(FileValidator, RejectsEmptyFile) {
    const fs::path p = fs::temp_directory_path() / "cnn_validator_empty.jpg";
    std::ofstream(p).close();
    EXPECT_TRUE(validate_file(p).has_value());
    fs::remove(p);
}

TEST(FileValidator, AcceptsNonEmptyFile) {
    const fs::path p = fs::temp_directory_path() / "cnn_validator_ok.jpg";
    std::ofstream(p) << "x";
    EXPECT_FALSE(validate_file(p).has_value());
    fs::remove(p);
}
