#include "binary_export.hpp"

#include <fstream>

namespace cnn::preprocessing {

bool export_binary_tensor(const std::filesystem::path& output,
                          std::initializer_list<unsigned int> dimensions,
                          const std::vector<float>& values, std::string& error) {
    if (dimensions.size() != 3) {
        error = "binary tensor export requires exactly three dimensions";
        return false;
    }
    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }
    const char magic[8] = {'C', 'N', 'N', 'T', 'N', 'S', '1', '\0'};
    std::ofstream out(output, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = "cannot open tensor output";
        return false;
    }
    out.write(magic, sizeof(magic));
    for (unsigned int dimension : dimensions) {
        out.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(
                  values.size() * sizeof(float)));
    if (!out) {
        error = "failed while writing tensor output";
        return false;
    }
    return true;
}

}  // namespace cnn::preprocessing
