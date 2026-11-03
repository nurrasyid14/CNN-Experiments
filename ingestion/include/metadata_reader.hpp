#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <set>
#include <string>

#include <nlohmann/json.hpp>

#include "image_reader.hpp"

namespace cnn::ingestion {

enum class LabelMode {
    None,         // no label
    Parent,       // label = name of the class folder directly under the root
    NumericStem,  // label = floor(number in file name / divisor)
    PrefixStem,   // label = text before '_' in file name
};

struct LabelRule {
    LabelMode mode = LabelMode::Parent;
    int divisor = 1;  // used by NumericStem
};

struct ImageRecord {
    std::string path;
    std::string label;  // empty = unlabeled
    std::string split;  // empty = no predefined split
    ImageInfo info;
    std::string tensor_path;
};

// std::nullopt means the label could not be resolved (an error for labeled modes).
std::optional<std::string> resolve_label(const std::filesystem::path& file,
                                         const std::filesystem::path& root,
                                         const LabelRule& rule);

// Returns the first directory component (under root) that is one of split_names, else "".
std::string resolve_split(const std::filesystem::path& file,
                          const std::filesystem::path& root,
                          const std::set<std::string>& split_names);

nlohmann::json to_json(const ImageRecord& record);

}  // namespace cnn::ingestion
