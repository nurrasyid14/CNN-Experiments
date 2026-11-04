#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "image_reader.hpp"

namespace fs = std::filesystem;
using namespace cnn::ingestion;

TEST(ImageReader, ReadsColorPng) {
    const fs::path p = fs::temp_directory_path() / "cnn_reader_color.png";
    cv::imwrite(p.string(), cv::Mat(10, 20, CV_8UC3, cv::Scalar(10, 20, 30)));
    const ReadResult r = read_image_info(p);
    ASSERT_TRUE(r.info.has_value()) << r.error;
    EXPECT_EQ(r.info->width, 20);
    EXPECT_EQ(r.info->height, 10);
    EXPECT_EQ(r.info->channels, 3);
    EXPECT_EQ(r.info->format, "PNG");
    fs::remove(p);
}

TEST(ImageReader, ReadsGrayscaleJpeg) {
    const fs::path p = fs::temp_directory_path() / "cnn_reader_gray.jpg";
    cv::imwrite(p.string(), cv::Mat(8, 8, CV_8UC1, cv::Scalar(128)));
    const ReadResult r = read_image_info(p);
    ASSERT_TRUE(r.info.has_value()) << r.error;
    EXPECT_EQ(r.info->channels, 1);
    EXPECT_EQ(r.info->format, "JPEG");
    fs::remove(p);
}

TEST(ImageReader, ReportsCorruptFileWithoutThrowing) {
    const fs::path p = fs::temp_directory_path() / "cnn_reader_bad.jpg";
    std::ofstream(p) << "this is not an image";
    const ReadResult r = read_image_info(p);
    EXPECT_FALSE(r.info.has_value());
    EXPECT_FALSE(r.error.empty());
    fs::remove(p);
}

TEST(ImageReader, DetectsFormatFromSignature) {
    EXPECT_EQ(detect_format({0xFF, 0xD8, 0xFF, 0xE0}), "JPEG");
    EXPECT_EQ(detect_format({'B', 'M', 0, 0}), "BMP");
    EXPECT_EQ(detect_format({1, 2, 3}), "UNKNOWN");
}
