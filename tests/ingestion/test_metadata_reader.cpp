#include <gtest/gtest.h>

#include "metadata_reader.hpp"

using namespace cnn::ingestion;

TEST(MetadataReader, NumericStemLabel) {
    const LabelRule rule{LabelMode::NumericStem, 100};
    EXPECT_EQ(resolve_label("r/train/250.jpg", "r", rule), "2");
    EXPECT_EQ(resolve_label("r/test/0.jpg", "r", rule), "0");
    EXPECT_EQ(resolve_label("r/validation/999.jpg", "r", rule), "9");
    EXPECT_FALSE(resolve_label("r/train/abc.jpg", "r", rule).has_value());
}

TEST(MetadataReader, ParentFolderLabel) {
    const LabelRule rule{LabelMode::Parent, 1};
    EXPECT_EQ(resolve_label("r/cat/1.jpg", "r", rule), "cat");
    EXPECT_EQ(resolve_label("r/cat/sub/1.jpg", "r", rule), "cat");
    EXPECT_FALSE(resolve_label("r/1.jpg", "r", rule).has_value());
}

TEST(MetadataReader, SplitFromFolderName) {
    const std::set<std::string> names = {"train", "validation", "test"};
    EXPECT_EQ(resolve_split("r/validation/1.jpg", "r", names), "validation");
    EXPECT_EQ(resolve_split("r/Train/1.jpg", "r", names), "train");
    EXPECT_EQ(resolve_split("r/other/1.jpg", "r", names), "");
}

TEST(MetadataReader, JsonOmitsEmptyLabelAndSplit) {
    ImageRecord rec;
    rec.path = "a.jpg";
    rec.info = {4, 3, 3, "JPEG", 10};
    const auto j = to_json(rec);
    EXPECT_FALSE(j.contains("label"));
    EXPECT_FALSE(j.contains("split"));
    EXPECT_EQ(j["width"], 4);
}
