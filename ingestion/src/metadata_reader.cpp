#include "metadata_reader.hpp"

#include <algorithm>
#include <cctype>

#include "text_utils.hpp"

namespace fs = std::filesystem;

namespace cnn::ingestion {

std::optional<std::string> resolve_label(const fs::path& file, const fs::path& root,
                                         const LabelRule& rule) {
    switch (rule.mode) {
        case LabelMode::None:
            return std::string{};

        case LabelMode::Parent: {
            const fs::path rel = file.lexically_relative(root);
            const fs::path dir = rel.parent_path();
            if (dir.empty() || dir == ".") return std::nullopt;  // file sits directly in root
            return to_utf8(*dir.begin());                        // first folder under root
        }

        case LabelMode::NumericStem: {
            const std::string stem = to_utf8(file.stem());
            const bool digits = !stem.empty() && stem.size() <= 9 &&
                                std::all_of(stem.begin(), stem.end(),
                                            [](unsigned char c) { return std::isdigit(c) != 0; });
            if (!digits || rule.divisor <= 0) return std::nullopt;
            return std::to_string(std::stoi(stem) / rule.divisor);
        }

        case LabelMode::PrefixStem: {
            const std::string stem = to_utf8(file.stem());
            const auto separator = stem.find('_');
            if (separator == std::string::npos || separator == 0) return std::nullopt;
            const std::string prefix = stem.substr(0, separator);
            const bool numeric = std::all_of(
                prefix.begin(), prefix.end(),
                [](unsigned char c) { return std::isdigit(c) != 0; });
            return numeric ? std::optional<std::string>(prefix) : std::nullopt;
        }
    }
    return std::nullopt;
}

std::string resolve_split(const fs::path& file, const fs::path& root,
                          const std::set<std::string>& split_names) {
    const fs::path dir = file.lexically_relative(root).parent_path();
    for (const auto& part : dir) {
        const std::string name = to_lower(to_utf8(part));
        if (split_names.count(name)) return name;
    }
    return {};
}

nlohmann::json to_json(const ImageRecord& r) {
    nlohmann::json j;
    j["path"] = r.path;
    if (!r.label.empty()) j["label"] = r.label;
    if (!r.split.empty()) j["split"] = r.split;
    j["width"] = r.info.width;
    j["height"] = r.info.height;
    j["channels"] = r.info.channels;
    j["format"] = r.info.format;
    j["file_size_bytes"] = r.info.file_size;
    if (!r.tensor_path.empty()) j["tensor_path"] = r.tensor_path;
    return j;
}

}  // namespace cnn::ingestion
