// cnn_ingest: scan a directory tree, validate/decode images, write a JSONL manifest.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "file_validator.hpp"
#include "image_reader.hpp"
#include "metadata_reader.hpp"
#include "text_utils.hpp"

namespace fs = std::filesystem;
using namespace cnn::ingestion;

namespace {

void print_usage() {
    std::cout <<
        "Usage: cnn_ingest --root DIR [options]\n"
        "  --root DIR             directory to scan recursively (required)\n"
        "  --out FILE             manifest (JSON Lines), default manifest.jsonl\n"
        "  --errors FILE          error log (JSON Lines), default <out>.errors.jsonl\n"
        "  --label-mode MODE      parent | numeric-stem | none   (default parent)\n"
        "                           parent:       class = first folder under --root\n"
        "                           numeric-stem: class = floor(file number / --label-divisor)\n"
        "  --label-divisor N      divisor for numeric-stem (default 1)\n"
        "  --split-names A,B,C    folder names recorded as the 'split' field\n"
        "                         (default train,validation,test; empty string disables)\n";
}

std::set<std::string> parse_list(const std::string& s) {
    std::set<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = to_lower(item);
        if (!item.empty()) out.insert(item);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    fs::path root, out = "manifest.jsonl", errors_path;
    LabelRule rule;
    std::set<std::string> split_names = {"train", "validation", "test"};

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << key << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (key == "--help" || key == "-h") { print_usage(); return 0; }
        else if (key == "--root") root = from_utf8(value());
        else if (key == "--out") out = from_utf8(value());
        else if (key == "--errors") errors_path = from_utf8(value());
        else if (key == "--label-divisor") rule.divisor = std::atoi(value().c_str());
        else if (key == "--split-names") split_names = parse_list(value());
        else if (key == "--label-mode") {
            const std::string m = value();
            if (m == "parent") rule.mode = LabelMode::Parent;
            else if (m == "numeric-stem") rule.mode = LabelMode::NumericStem;
            else if (m == "none") rule.mode = LabelMode::None;
            else { std::cerr << "Unknown label mode: " << m << "\n"; return 2; }
        } else { std::cerr << "Unknown option: " << key << "\n"; print_usage(); return 2; }
    }

    if (root.empty()) { print_usage(); return 2; }
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        std::cerr << "Root is not a directory: " << to_utf8(root) << "\n";
        return 1;
    }
    if (errors_path.empty()) {
        errors_path = out;
        errors_path.replace_extension(".errors.jsonl");
    }
    for (const fs::path& p : {out, errors_path}) {
        if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    }

    // Collect files, then sort so the manifest is deterministic.
    std::vector<fs::path> files;
    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
    if (ec) { std::cerr << "Cannot traverse root: " << ec.message() << "\n"; return 1; }
    for (; it != end; it.increment(ec)) {
        if (ec) { std::cerr << "Traversal stopped: " << ec.message() << "\n"; break; }
        if (it->is_regular_file(ec)) files.push_back(it->path());
    }
    std::sort(files.begin(), files.end(),
              [](const fs::path& a, const fs::path& b) { return to_utf8(a) < to_utf8(b); });

    std::ofstream manifest(out, std::ios::binary);
    std::ofstream error_log(errors_path, std::ios::binary);
    if (!manifest || !error_log) { std::cerr << "Cannot open output files\n"; return 1; }

    std::size_t ingested = 0, failed = 0, skipped = 0;
    std::map<std::string, std::size_t> per_label, per_split;

    auto log_error = [&](const fs::path& p, const char* stage, const std::string& reason) {
        nlohmann::json j = {{"path", to_utf8(p)}, {"stage", stage}, {"reason", reason}};
        error_log << j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
        ++failed;
    };

    for (const fs::path& file : files) {
        if (!is_supported_extension(file)) { ++skipped; continue; }

        if (auto err = validate_file(file)) { log_error(file, "validate", *err); continue; }

        ReadResult read = read_image_info(file);
        if (!read.info) { log_error(file, "decode", read.error); continue; }

        auto label = resolve_label(file, root, rule);
        if (!label) { log_error(file, "label", "label could not be resolved with the selected label mode"); continue; }

        ImageRecord rec;
        rec.path = to_utf8(file);
        rec.label = *label;
        rec.split = resolve_split(file, root, split_names);
        rec.info = *read.info;

        manifest << to_json(rec).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
        ++ingested;
        ++per_label[rec.label.empty() ? "(none)" : rec.label];
        ++per_split[rec.split.empty() ? "(none)" : rec.split];
    }

    std::cout << "Scanned files      : " << files.size() << "\n"
              << "Ingested images    : " << ingested << "  -> " << to_utf8(out) << "\n"
              << "Failed (reported)  : " << failed << "  -> " << to_utf8(errors_path) << "\n"
              << "Skipped (non-image): " << skipped << "\n"
              << "Classes            : " << per_label.size() << "\n";
    for (const auto& [k, v] : per_label) std::cout << "  class " << k << ": " << v << "\n";
    std::cout << "Splits:\n";
    for (const auto& [k, v] : per_split) std::cout << "  " << k << ": " << v << "\n";
    return 0;
}
