#pragma once
#include <filesystem>
#include <optional>
#include <string>

namespace cnn::ingestion {

// True for extensions the pipeline ingests (case-insensitive).
bool is_supported_extension(const std::filesystem::path& p);

// Returns an error message if the file cannot be ingested, std::nullopt if it is usable.
std::optional<std::string> validate_file(const std::filesystem::path& p);

}  // namespace cnn::ingestion
