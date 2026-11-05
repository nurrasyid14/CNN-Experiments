#pragma once

#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

namespace cnn::preprocessing {

bool export_binary_tensor(const std::filesystem::path& output,
                          std::initializer_list<unsigned int> dimensions,
                          const std::vector<float>& values, std::string& error);

}  // namespace cnn::preprocessing
