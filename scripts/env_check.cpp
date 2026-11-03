#include <filesystem>
#include <iostream>

int main() {
    std::cout << "C++ standard macro: " << __cplusplus << "\n";
#ifdef _MSC_VER
    std::cout << "MSVC version: " << _MSC_VER << "\n";
#endif
    std::cout << "Current path (std::filesystem): "
              << std::filesystem::current_path().string() << "\n";
    std::cout << "C++ build environment OK\n";
    return 0;
}
