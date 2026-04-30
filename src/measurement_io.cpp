#include "axyb/measurement_io.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace {

std::string read_token(std::istream& in, const std::string& path, const std::string& context) {
    std::string token;
    if (!(in >> token)) {
        throw std::runtime_error("unexpected end of file in " + path + " while reading " + context);
    }
    return token;
}

void expect_token(std::istream& in, const std::string& path, const std::string& expected, const std::string& context) {
    const std::string token = read_token(in, path, context);
    if (token != expected) {
        throw std::runtime_error("expected '" + expected + "' in " + path + " while reading " + context + ", got '" + token + "'");
    }
}

int read_int(std::istream& in, const std::string& path, const std::string& context) {
    int value = 0;
    if (!(in >> value)) {
        throw std::runtime_error("failed to read integer in " + path + " while reading " + context);
    }
    return value;
}

double read_double(std::istream& in, const std::string& path, const std::string& context) {
    double value = 0.0;
    if (!(in >> value)) {
        throw std::runtime_error("failed to read floating-point value in " + path + " while reading " + context);
    }
    return value;
}

void write_matrix_line(std::ostream& out, const char* label, const axyb::Matrix& M) {
    out << label;
    for (double value : M.a) out << ' ' << value;
    out << '\n';
}

axyb::Matrix read_matrix_line(std::istream& in, const std::string& path, const char* label) {
    expect_token(in, path, label, label);
    axyb::Matrix M(4, 4);
    for (double& value : M.a) value = read_double(in, path, label);
    return M;
}

} // namespace

namespace axyb {

void write_problem_sequence_header(std::ostream& out, int problem_count) {
    out << "AXYB_LOOP_MEAS_V1\n";
    out << "problems " << problem_count << '\n';
}

void write_problem(std::ostream& out, int index, const ProblemData& problem) {
    out << "problem " << index << " len " << problem.As.size() << '\n';
    out << "g_ground_truth";
    for (double value : problem.g_ground_truth) out << ' ' << value;
    out << '\n';
    write_matrix_line(out, "X0", problem.X0);
    write_matrix_line(out, "Y0", problem.Y0);
    for (size_t i = 0; i < problem.As.size(); ++i) {
        write_matrix_line(out, "A", problem.As[i]);
        write_matrix_line(out, "B", problem.Bs[i]);
    }
    out << "end_problem\n";
}

void write_problem_sequence_footer(std::ostream& out) {
    out << "end_file\n";
}

std::vector<ProblemData> load_problem_sequence(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input measurement file " + path);

    expect_token(in, path, "AXYB_LOOP_MEAS_V1", "file header");
    expect_token(in, path, "problems", "problem count");
    const int problem_count = read_int(in, path, "problem count");
    if (problem_count < 0) throw std::runtime_error("invalid negative problem count in " + path);

    std::vector<ProblemData> problems;
    problems.reserve(static_cast<size_t>(problem_count));
    for (int expected_index = 1; expected_index <= problem_count; ++expected_index) {
        expect_token(in, path, "problem", "problem header");
        const int file_index = read_int(in, path, "problem index");
        if (file_index != expected_index) {
            throw std::runtime_error("unexpected problem index in " + path + ": expected " + std::to_string(expected_index) +
                                     ", got " + std::to_string(file_index));
        }
        expect_token(in, path, "len", "problem length");
        const int len = read_int(in, path, "problem length");
        if (len < 0) throw std::runtime_error("invalid negative problem length in " + path);

        ProblemData problem;
        problem.As.resize(static_cast<size_t>(len));
        problem.Bs.resize(static_cast<size_t>(len));
        expect_token(in, path, "g_ground_truth", "ground truth");
        for (double& value : problem.g_ground_truth) value = read_double(in, path, "ground truth");
        problem.X0 = read_matrix_line(in, path, "X0");
        problem.Y0 = read_matrix_line(in, path, "Y0");
        for (int i = 0; i < len; ++i) {
            problem.As[static_cast<size_t>(i)] = read_matrix_line(in, path, "A");
            problem.Bs[static_cast<size_t>(i)] = read_matrix_line(in, path, "B");
        }
        expect_token(in, path, "end_problem", "problem terminator");
        problems.push_back(std::move(problem));
    }
    expect_token(in, path, "end_file", "file terminator");
    return problems;
}

} // namespace axyb
