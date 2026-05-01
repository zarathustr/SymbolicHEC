#include "axyb/axyb_solver.hpp"
#include "cli.hpp"

#include <tbb/global_control.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;

struct RwheCli {
    axyb::CommonCli common;
    std::string measurement_dir;
    int min_measurements = 3;
    std::optional<int> tag_filter;
    std::optional<int> camera_filter;
    bool help = false;
};

struct MeasurementSet {
    int tag = 0;
    int camera = 0;
    std::vector<axyb::Matrix> As;
    std::vector<axyb::Matrix> Bs;
};

struct PairSolution {
    MeasurementSet measurements;
    axyb::SolveResult result;
};

struct AggregatePose {
    double weight_sum = 0.0;
    axyb::Matrix rotation_sum = axyb::Matrix(3, 3);
    std::vector<double> translation_sum = std::vector<double>(3, 0.0);
    int pair_count = 0;
};

struct PoseEstimate {
    axyb::Matrix T = axyb::Matrix::eye(4);
    double weight_sum = 0.0;
    int pair_count = 0;
};

std::vector<std::string> measurement_dir_candidates(const char* argv0) {
    std::vector<std::string> candidates;
    std::string exe_dir = axyb::executable_dir(argv0);
    candidates.push_back(axyb::join_path(axyb::join_path(exe_dir, ".."), "certifiable-rwhe-calibration/data/real-world/combined"));
    candidates.push_back("certifiable-rwhe-calibration/data/real-world/combined");
    candidates.push_back("../certifiable-rwhe-calibration/data/real-world/combined");
    return candidates;
}

bool has_rwhe_csvs(const std::string& dir) {
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return false;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) return false;
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".csv") return true;
    }
    return false;
}

std::string resolve_measurement_dir(const char* argv0, const std::string& explicit_dir) {
    if (!explicit_dir.empty()) return explicit_dir;
    for (const auto& candidate : measurement_dir_candidates(argv0))
        if (has_rwhe_csvs(candidate))
            return candidate;
    throw std::runtime_error("could not locate certifiable-rwhe-calibration/data/real-world/combined; pass --measurement-dir");
}

void print_help(const char* exe) {
    std::cout
        << "Process the certifiable-rwhe-calibration real-world tag/camera measurements,\n"
        << "solve each observed tag/camera pair with the current AXYB solver, and\n"
        << "aggregate one pose estimate per tag and per camera.\n\n";
    axyb::print_common_help(exe, false);
    std::cout
        << "\nAdditional options:\n"
        << "  --measurement-dir DIR  Directory containing tag_*_cam_*_{A,B}.csv files\n"
        << "  --min-measurements N   Skip pairs with fewer than N rows; default 3\n"
        << "  --tag ID               Process only one tag id\n"
        << "  --camera ID            Process only one camera id\n"
        << "\nNotes:\n"
        << "  `--backend`, `--retry_tol`, template selection, and linear-solver options are\n"
        << "  applied to each tag/camera pair solve. `--len`, `--noise`, and `--seed` are\n"
        << "  accepted for CLI compatibility but are not used by this executable.\n";
}

RwheCli parse_rwhe_cli(int argc, char** argv) {
    RwheCli cli;
    std::vector<char*> common_argv;
    common_argv.reserve(static_cast<size_t>(argc));
    common_argv.push_back(argc ? argv[0] : const_cast<char*>("run_rw_multi_eye_multi_hand"));
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto val = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + name);
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") {
            cli.help = true;
        } else if (arg == "--measurement-dir") {
            cli.measurement_dir = val("--measurement-dir");
        } else if (arg == "--min-measurements") {
            cli.min_measurements = std::stoi(val("--min-measurements"));
        } else if (arg == "--tag") {
            cli.tag_filter = std::stoi(val("--tag"));
        } else if (arg == "--camera") {
            cli.camera_filter = std::stoi(val("--camera"));
        } else {
            common_argv.push_back(argv[i]);
        }
    }
    if (cli.min_measurements < 1) throw std::runtime_error("--min-measurements must be >= 1");
    if (!cli.help) {
        cli.common = axyb::parse_common_cli(static_cast<int>(common_argv.size()), common_argv.data(), false);
        cli.measurement_dir = resolve_measurement_dir(argc ? argv[0] : nullptr, cli.measurement_dir);
    }
    return cli;
}

std::vector<double> parse_csv_row(const std::string& line, const fs::path& path, int line_number) {
    std::vector<double> values;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        if (!cell.empty() && (cell.back() == '\r' || cell.back() == '\n')) cell.pop_back();
        if (!cell.empty()) values.push_back(std::stod(cell));
    }
    if (values.empty()) return values;
    if (values.size() != 7) throw std::runtime_error("expected 7 comma-separated values in " + path.string() + " line " + std::to_string(line_number));
    return values;
}

axyb::Matrix quat_trans_to_pose(const std::vector<double>& row) {
    const double w = row[0];
    const double x = row[1];
    const double y = row[2];
    const double z = row[3];
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (!(n > 0.0)) throw std::runtime_error("invalid zero-norm quaternion in measurement row");
    const double qw = w / n;
    const double qx = x / n;
    const double qy = y / n;
    const double qz = z / n;

    axyb::Matrix T = axyb::Matrix::eye(4);
    T(0, 0) = 1.0 - 2.0 * (qy * qy + qz * qz);
    T(0, 1) = 2.0 * (qx * qy - qz * qw);
    T(0, 2) = 2.0 * (qx * qz + qy * qw);
    T(1, 0) = 2.0 * (qx * qy + qz * qw);
    T(1, 1) = 1.0 - 2.0 * (qx * qx + qz * qz);
    T(1, 2) = 2.0 * (qy * qz - qx * qw);
    T(2, 0) = 2.0 * (qx * qz - qy * qw);
    T(2, 1) = 2.0 * (qy * qz + qx * qw);
    T(2, 2) = 1.0 - 2.0 * (qx * qx + qy * qy);
    T(0, 3) = row[4];
    T(1, 3) = row[5];
    T(2, 3) = row[6];
    return T;
}

std::vector<axyb::Matrix> load_pose_csv(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open measurement file " + path.string());
    std::vector<axyb::Matrix> poses;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        if (line.empty()) continue;
        auto values = parse_csv_row(line, path, line_number);
        if (values.empty()) continue;
        poses.push_back(quat_trans_to_pose(values));
    }
    return poses;
}

std::vector<MeasurementSet> discover_measurements(const RwheCli& cli) {
    std::regex pattern(R"(^tag_(\d+)_cam_(\d+)_([AB])\.csv$)");
    std::map<std::pair<int, int>, fs::path> a_files;
    std::map<std::pair<int, int>, fs::path> b_files;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(cli.measurement_dir, ec)) {
        if (ec) throw std::runtime_error("cannot iterate measurement directory " + cli.measurement_dir);
        if (!entry.is_regular_file()) continue;
        std::smatch match;
        const std::string name = entry.path().filename().string();
        if (!std::regex_match(name, match, pattern)) continue;
        const int tag = std::stoi(match[1].str());
        const int camera = std::stoi(match[2].str());
        if (cli.tag_filter && tag != *cli.tag_filter) continue;
        if (cli.camera_filter && camera != *cli.camera_filter) continue;
        const std::pair<int, int> key{tag, camera};
        if (match[3].str() == "A") a_files[key] = entry.path();
        else b_files[key] = entry.path();
    }

    std::vector<MeasurementSet> measurements;
    for (const auto& entry : a_files) {
        auto it = b_files.find(entry.first);
        if (it == b_files.end()) continue;
        MeasurementSet set;
        set.tag = entry.first.first;
        set.camera = entry.first.second;
        set.As = load_pose_csv(entry.second);
        set.Bs = load_pose_csv(it->second);
        if (set.As.size() != set.Bs.size()) {
            throw std::runtime_error("mismatched row counts for tag " + std::to_string(set.tag) + " cam " + std::to_string(set.camera));
        }
        measurements.push_back(std::move(set));
    }
    std::sort(measurements.begin(), measurements.end(), [](const MeasurementSet& lhs, const MeasurementSet& rhs) {
        return std::tie(lhs.tag, lhs.camera) < std::tie(rhs.tag, rhs.camera);
    });
    return measurements;
}

axyb::Matrix rotation_part(const axyb::Matrix& T) {
    axyb::Matrix R(3, 3);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            R(r, c) = T(r, c);
    return R;
}

std::vector<double> translation_part(const axyb::Matrix& T) { return {T(0, 3), T(1, 3), T(2, 3)}; }

void accumulate_pose(AggregatePose& aggregate, const axyb::Matrix& T, double weight) {
    auto R = rotation_part(T);
    auto t = translation_part(T);
    for (size_t i = 0; i < aggregate.rotation_sum.a.size(); ++i) aggregate.rotation_sum.a[i] += weight * R.a[i];
    for (int i = 0; i < 3; ++i) aggregate.translation_sum[static_cast<size_t>(i)] += weight * t[static_cast<size_t>(i)];
    aggregate.weight_sum += weight;
    aggregate.pair_count += 1;
}

PoseEstimate finalize_pose(const AggregatePose& aggregate) {
    if (!(aggregate.weight_sum > 0.0)) throw std::runtime_error("cannot finalize pose with zero weight");
    axyb::Matrix R = aggregate.rotation_sum;
    for (double& value : R.a) value /= aggregate.weight_sum;
    R = axyb::orthonormalize3(R);
    std::vector<double> t(3);
    for (int i = 0; i < 3; ++i) t[static_cast<size_t>(i)] = aggregate.translation_sum[static_cast<size_t>(i)] / aggregate.weight_sum;
    PoseEstimate estimate;
    estimate.T = axyb::Matrix::eye(4);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            estimate.T(r, c) = R(r, c);
    for (int i = 0; i < 3; ++i) estimate.T(i, 3) = t[static_cast<size_t>(i)];
    estimate.weight_sum = aggregate.weight_sum;
    estimate.pair_count = aggregate.pair_count;
    return estimate;
}

void print_pose_summary(const std::string& label, int id, const PoseEstimate& pose) {
    std::cout << label << " " << id << ":\n";
    std::cout << "  pairs = " << pose.pair_count << ", weight = " << pose.weight_sum << "\n";
    axyb::print_matrix(pose.T, "  T");
}

} // namespace

int main(int argc, char** argv) {
    try {
        RwheCli cli = parse_rwhe_cli(argc, argv);
        if (cli.help) {
            print_help(argv[0]);
            return 0;
        }

        std::unique_ptr<tbb::global_control> gc;
        if (cli.common.solver.threads > 0) {
            gc.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, cli.common.solver.threads));
        }

        auto measurements = discover_measurements(cli);
        if (measurements.empty()) throw std::runtime_error("no tag/camera measurement pairs found in " + cli.measurement_dir);

        std::set<int> tags;
        std::set<int> cameras;
        for (const auto& m : measurements) {
            tags.insert(m.tag);
            cameras.insert(m.camera);
        }

        std::cout << "measurement_dir = " << cli.measurement_dir << "\n";
        std::cout << "tags found =";
        for (int tag : tags) std::cout << ' ' << tag;
        std::cout << "\ncameras found =";
        for (int camera : cameras) std::cout << ' ' << camera;
        std::cout << "\nbackend = " << cli.common.solver.backend_name;
        if (cli.common.retry_tol_set) std::cout << "\nretry_tol = " << cli.common.retry_tol;
        if (cli.common.solver.prescale != 1.0) std::cout << "\nprescale = " << cli.common.solver.prescale;
        if (cli.common.solver.asymmetric) std::cout << "\nasymmetric = true";
        std::cout << "\n";

        std::vector<PairSolution> solved_pairs;
        std::map<int, AggregatePose> tag_aggregates;
        std::map<int, AggregatePose> camera_aggregates;

        for (const auto& measurement : measurements) {
            const int count = static_cast<int>(measurement.As.size());
            if (count < cli.min_measurements) {
                std::cout << "skip tag " << measurement.tag << " cam " << measurement.camera << ": only " << count
                          << " measurements\n";
                continue;
            }
            auto result = axyb::AXYB_complete_grobner_with_retry(
                measurement.As, measurement.Bs, cli.common.mu, cli.common.solver,
                cli.common.retry_tol_set ? cli.common.retry_tol : -1.0);
            PairSolution pair_solution{measurement, result};
            solved_pairs.push_back(pair_solution);
            const double weight = static_cast<double>(count);
            accumulate_pose(tag_aggregates[measurement.tag], result.X, weight);
            accumulate_pose(camera_aggregates[measurement.camera], result.Y, weight);

            std::cout << "pair tag " << measurement.tag << " cam " << measurement.camera << ": measurements = " << count
                      << ", objective = " << result.objective;
            if (cli.common.retry_tol_set) {
                std::cout << ", backend = " << result.backend_name << ", attempts = " << result.attempts;
            }
            std::cout << "\n";
            axyb::print_dense_solve_diagnostics(result);
        }

        if (solved_pairs.empty()) throw std::runtime_error("no tag/camera pair met --min-measurements");

        std::map<int, PoseEstimate> tag_estimates;
        for (const auto& entry : tag_aggregates) tag_estimates.emplace(entry.first, finalize_pose(entry.second));
        std::map<int, PoseEstimate> camera_estimates;
        for (const auto& entry : camera_aggregates) camera_estimates.emplace(entry.first, finalize_pose(entry.second));

        double weighted_objective_sum = 0.0;
        double total_measurements = 0.0;
        std::cout << "aggregated pair objectives:\n";
        for (const auto& pair : solved_pairs) {
            const auto tag_it = tag_estimates.find(pair.measurements.tag);
            const auto cam_it = camera_estimates.find(pair.measurements.camera);
            const double pair_objective = axyb::J_AXYB(pair.measurements.As, pair.measurements.Bs, tag_it->second.T, cam_it->second.T);
            const double count = static_cast<double>(pair.measurements.As.size());
            weighted_objective_sum += pair_objective * count;
            total_measurements += count;
            std::cout << "  tag " << pair.measurements.tag << " cam " << pair.measurements.camera << ": objective = "
                      << pair_objective << "\n";
        }
        std::cout << "overall measurement-weighted objective = " << (weighted_objective_sum / std::max(total_measurements, 1.0))
                  << "\n";

        std::cout << "tag solutions:\n";
        for (const auto& entry : tag_estimates) print_pose_summary("tag", entry.first, entry.second);
        std::cout << "camera solutions:\n";
        for (const auto& entry : camera_estimates) print_pose_summary("camera", entry.first, entry.second);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
