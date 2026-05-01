#include "axyb/template_data.hpp"

#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>

namespace axyb {
namespace {

template <class T>
void rd(std::ifstream& in, T* p, size_t n) {
    in.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(sizeof(T) * n));
    if (!in) throw std::runtime_error("template read failed");
}

std::string path(const std::string& dir, int action) {
    std::string resolved = dir.empty() ? "." : dir;
    if (resolved.back() != '/' && resolved.back() != '\\') resolved += '/';
    return resolved + "AXYB_complete_x" + std::to_string(action) + ".tpl";
}

SparseTemplatePattern make_pattern(int rows, int cols, const std::vector<unsigned long long>& linear,
                                   const std::vector<unsigned int>& coeff_index) {
    struct Entry {
        int col = 0;
        int row = 0;
        unsigned int coeff = 0;
    };

    std::vector<Entry> entries;
    entries.reserve(linear.size());
    for (size_t i = 0; i < linear.size(); ++i) {
        entries.push_back({
            static_cast<int>(linear[i] / static_cast<unsigned long long>(rows)),
            static_cast<int>(linear[i] % static_cast<unsigned long long>(rows)),
            coeff_index[i]
        });
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
        return lhs.col == rhs.col ? lhs.row < rhs.row : lhs.col < rhs.col;
    });

    SparseTemplatePattern pattern;
    pattern.rows = rows;
    pattern.cols = cols;
    pattern.col_ptr.assign(static_cast<size_t>(cols) + 1, 0);
    pattern.row_idx.resize(entries.size());
    pattern.coeff_index.resize(entries.size());
    for (const auto& entry : entries) ++pattern.col_ptr[static_cast<size_t>(entry.col) + 1];
    for (int col = 0; col < cols; ++col) pattern.col_ptr[static_cast<size_t>(col) + 1] += pattern.col_ptr[static_cast<size_t>(col)];
    auto current = pattern.col_ptr;
    for (const auto& entry : entries) {
        const int pos = current[static_cast<size_t>(entry.col)]++;
        pattern.row_idx[static_cast<size_t>(pos)] = entry.row;
        pattern.coeff_index[static_cast<size_t>(pos)] = entry.coeff;
    }
    return pattern;
}

TemplateData load_template_uncached(const std::string& dir, int action) {
    std::ifstream in(path(dir, action), std::ios::binary);
    if (!in) throw std::runtime_error("cannot open template x" + std::to_string(action) + " in " + dir);

    char magic[8];
    rd(in, magic, 8);
    if (std::string(magic, magic + 8) != "AXYBTPL1") throw std::runtime_error("bad template");

    uint32_t h[10];
    rd(in, h, 10);

    TemplateData t;
    t.version = static_cast<int>(h[0]);
    t.action = static_cast<int>(h[1]);
    t.n = static_cast<int>(h[2]);
    t.m = static_cast<int>(h[3]);
    const uint32_t c0 = h[4];
    const uint32_t c1 = h[5];
    const uint32_t cc = h[6];
    const uint32_t am = h[7];
    const uint32_t tail = h[8];
    const uint32_t bc = h[9];
    t.tail_count = static_cast<int>(tail);

    t.coeff_map.resize(cc);
    rd(in, t.coeff_map.data(), t.coeff_map.size());
    t.c0_linear.resize(c0);
    rd(in, t.c0_linear.data(), t.c0_linear.size());
    t.c0_coeff.resize(c0);
    rd(in, t.c0_coeff.data(), t.c0_coeff.size());
    t.c1_linear.resize(c1);
    rd(in, t.c1_linear.data(), t.c1_linear.size());
    t.c1_coeff.resize(c1);
    rd(in, t.c1_coeff.data(), t.c1_coeff.size());
    t.am_ind.resize(am);
    rd(in, t.am_ind.data(), t.am_ind.size());
    rd(in, t.sol_sources.data(), t.sol_sources.size());

    std::vector<uint32_t> block_pairs(2 * static_cast<size_t>(bc));
    rd(in, block_pairs.data(), block_pairs.size());
    for (uint32_t i = 0; i < bc; ++i) {
        t.blocks.emplace_back(static_cast<int>(block_pairs[2 * static_cast<size_t>(i)]),
                              static_cast<int>(block_pairs[2 * static_cast<size_t>(i) + 1]));
    }

    t.c0_pattern = make_pattern(t.n, t.n, t.c0_linear, t.c0_coeff);
    t.c1_pattern = make_pattern(t.n, t.m, t.c1_linear, t.c1_coeff);
    return t;
}

SparseMatrixCSC build_from_pattern(const SparseTemplatePattern& pattern, const std::vector<double>& coeffs) {
    SparseMatrixCSC out;
    out.rows = pattern.rows;
    out.cols = pattern.cols;
    out.col_ptr = pattern.col_ptr;
    out.row_idx = pattern.row_idx;
    out.values.resize(pattern.coeff_index.size());
    for (size_t i = 0; i < pattern.coeff_index.size(); ++i) out.values[i] = coeffs[pattern.coeff_index[i]];
    return out;
}

std::mutex cache_mutex;
std::map<std::string, TemplateData> cache;

} // namespace

TemplateData load_template(const std::string& dir, int action) { return load_template_uncached(dir, action); }

const TemplateData& load_template_cached(const std::string& dir, int action) {
    const std::string key = path(dir, action);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, load_template_uncached(dir, action)).first;
    return it->second;
}

std::vector<double> map_coefficients(const TemplateData& tpl, const std::vector<double>& data) {
    std::vector<double> coeffs(tpl.coeff_map.size());
    for (size_t i = 0; i < coeffs.size(); ++i) coeffs[i] = data[tpl.coeff_map[i]];
    return coeffs;
}

SparseMatrixCSC build_c0_from_template(const TemplateData& tpl, const std::vector<double>& coeffs) {
    return build_from_pattern(tpl.c0_pattern, coeffs);
}

SparseMatrixCSC build_c1_from_template(const TemplateData& tpl, const std::vector<double>& coeffs) {
    return build_from_pattern(tpl.c1_pattern, coeffs);
}

SparseMatrixCSC build_sparse_from_template(int rows, int cols, const std::vector<unsigned long long>& linear,
                                           const std::vector<unsigned int>& coeff_index, const std::vector<double>& coeffs) {
    std::vector<double> values(linear.size());
    for (size_t i = 0; i < linear.size(); ++i) values[i] = coeffs[coeff_index[i]];
    return SparseMatrixCSC::from_linear_indices(rows, cols, linear, values);
}

} // namespace axyb
