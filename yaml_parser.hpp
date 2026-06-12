#ifndef YAML_PARSER_HPP
#define YAML_PARSER_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>

class YAMLConfig {
private:
    std::unordered_map<std::string, std::string> data;

    // Helper to trim leading/trailing whitespace
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

    // Helper to strip quotes
    static std::string strip_quotes(const std::string& str) {
        std::string s = trim(str);
        if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
            return s.substr(1, s.size() - 2);
        }
        return s;
    }

public:
    YAMLConfig() = default;

    bool load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open config file: " << filepath << std::endl;
            return false;
        }

        std::vector<std::pair<int, std::string>> hierarchy; // list of {indent_level, key}
        std::string line;

        while (std::getline(file, line)) {
            // 1. Remove comments
            size_t hash_pos = line.find('#');
            if (hash_pos != std::string::npos) {
                line = line.substr(0, hash_pos);
            }

            // 2. Determine indentation level
            size_t indent = 0;
            while (indent < line.size() && (line[indent] == ' ' || line[indent] == '\t')) {
                indent++;
            }

            std::string cleaned = trim(line);
            if (cleaned.empty()) continue; // skip empty lines

            // 3. Parse key-value
            size_t colon_pos = cleaned.find(':');
            if (colon_pos == std::string::npos) continue; // invalid line format

            std::string key = trim(cleaned.substr(0, colon_pos));
            std::string val = trim(cleaned.substr(colon_pos + 1));

            // Adjust hierarchy based on indentation
            while (!hierarchy.empty() && hierarchy.back().first >= (int)indent) {
                hierarchy.pop_back();
            }

            // Construct full path prefix
            std::string prefix = "";
            for (const auto& level : hierarchy) {
                prefix += level.second + ".";
            }

            if (val.empty() || val[0] == '\n' || val[0] == '\r') {
                // Nested dictionary start
                hierarchy.push_back({(int)indent, key});
            } else {
                // Leaf value
                std::string full_key = prefix + key;
                data[full_key] = strip_quotes(val);
            }
        }

        return true;
    }

    std::string getString(const std::string& key, const std::string& default_val = "") const {
        auto it = data.find(key);
        if (it != data.end()) {
            return it->second;
        }
        return default_val;
    }

    int getInt(const std::string& key, int default_val = 0) const {
        auto it = data.find(key);
        if (it != data.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {}
        }
        return default_val;
    }

    double getDouble(const std::string& key, double default_val = 0.0) const {
        auto it = data.find(key);
        if (it != data.end()) {
            try {
                return std::stod(it->second);
            } catch (...) {}
        }
        return default_val;
    }

    bool getBool(const std::string& key, bool default_val = false) const {
        auto it = data.find(key);
        if (it != data.end()) {
            std::string val = it->second;
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "true" || val == "yes" || val == "1") return true;
            if (val == "false" || val == "no" || val == "0") return false;
        }
        return default_val;
    }

    // Retrieve a dictionary of subkeys under a prefix
    std::map<std::string, double> getDict(const std::string& prefix) const {
        std::map<std::string, double> dict;
        std::string search_prefix = prefix + ".";
        for (const auto& pair : data) {
            if (pair.first.rfind(search_prefix, 0) == 0) {
                std::string subkey = pair.first.substr(search_prefix.size());
                try {
                    dict[subkey] = std::stod(pair.second);
                } catch (...) {}
            }
        }
        return dict;
    }

    void printConfig() const {
        std::cout << "--- Parsed Configuration Keys ---" << std::endl;
        for (const auto& pair : data) {
            std::cout << pair.first << ": " << pair.second << std::endl;
        }
        std::cout << "---------------------------------" << std::endl;
    }
};

#endif // YAML_PARSER_HPP
