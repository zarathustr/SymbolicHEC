#include "axyb/axyb_solver.hpp"
#include "axyb/measurement_io.hpp"
#include "cli.hpp"

#include <tbb/global_control.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct InputFileCli {
    axyb::CommonCli common;
    std::string input_meas;
    int problem_index = 1;
    double success_tol = 1e-2;
    bool help = false;
};

void print_help(const char* exe) {
    axyb::print_common_help(exe, false);
    std::cout
        << "\nAdditional options:\n"
        << "  --input_meas PATH   Read problems from a loop measurement file\n"
        << "  --problem N         1-based problem index to solve; default 1\n"
        << "  --success_tol X     Pose-error threshold for both X and Y; default 1e-2\n"
        << "\nNotes:\n"
        << "  Success is reported when both pose errors are finite and <= --success_tol.\n"
        << "  The ground truth is taken from X0 and Y0 stored in the input file.\n";
}

InputFileCli parse_input_file_cli(int argc, char** argv) {
    InputFileCli cli;
    std::vector<char*> common_argv;
    common_argv.reserve(static_cast<size_t>(argc));
    common_argv.push_back(argc ? argv[0] : const_cast<char*>("run_AXYB_input_file_example"));
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto val = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + name);
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") {
            cli.help = true;
        } else if (arg == "--input_meas") {
            cli.input_meas = val("--input_meas");
        } else if (arg == "--problem") {
            cli.problem_index = std::stoi(val("--problem"));
        } else if (arg == "--success_tol") {
            cli.success_tol = std::stod(val("--success_tol"));
        } else {
            common_argv.push_back(argv[i]);
        }
    }
    if (!cli.help) {
        if (cli.input_meas.empty()) throw std::runtime_error("--input_meas is required");
        if (cli.problem_index <= 0) throw std::runtime_error("--problem must be >= 1");
        if (cli.success_tol < 0.0) throw std::runtime_error("--success_tol must be >= 0");
        cli.common = axyb::parse_common_cli(static_cast<int>(common_argv.size()), common_argv.data(), false);
    }
    return cli;
}

double safe_pose_error(const axyb::Matrix& estimate, const axyb::Matrix& ground_truth) {
    if (estimate.rows != ground_truth.rows || estimate.cols != ground_truth.cols) {
        return std::numeric_limits<double>::infinity();
    }
    return axyb::pose_error(estimate, ground_truth);
}

} // namespace

int main(int argc, char** argv) {
    try {
        InputFileCli cli = parse_input_file_cli(argc, argv);
        if (cli.help) {
            print_help(argv[0]);
            return 0;
        }

        std::unique_ptr<tbb::global_control> gc;
        if (cli.common.solver.threads > 0) {
            gc.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, cli.common.solver.threads));
        }

        std::vector<axyb::ProblemData> problems = axyb::load_problem_sequence(cli.input_meas);
        if (problems.empty()) throw std::runtime_error("no problems found in " + cli.input_meas);
        if (cli.problem_index > static_cast<int>(problems.size())) {
            throw std::runtime_error("--problem " + std::to_string(cli.problem_index) +
                                     " is out of range for file with " + std::to_string(problems.size()) + " problems");
        }

        const axyb::ProblemData& problem = problems[static_cast<size_t>(cli.problem_index - 1)];
        const double objective_gt = axyb::J_AXYB(problem.As, problem.Bs, problem.X0, problem.Y0);
        auto result = axyb::AXYB_complete_grobner_with_retry(problem.As, problem.Bs, cli.common.mu, cli.common.solver,
                                                             cli.common.retry_tol_set ? cli.common.retry_tol : -1.0);
        const double x_error = safe_pose_error(result.X, problem.X0);
        const double y_error = safe_pose_error(result.Y, problem.Y0);
        const bool successful = std::isfinite(result.objective) && std::isfinite(x_error) && std::isfinite(y_error) &&
                                x_error <= cli.success_tol && y_error <= cli.success_tol;

        std::cout << "input_meas = " << cli.input_meas
                  << "\nproblem = " << cli.problem_index << "/" << problems.size()
                  << "\nlen = " << problem.As.size()
                  << "\nbackend = " << cli.common.solver.backend_name;
        if (cli.common.retry_tol_set) std::cout << "\nretry_tol = " << cli.common.retry_tol;
        std::cout << "\nsuccess_tol = " << cli.success_tol
                  << "\nground_truth objective = " << objective_gt
                  << "\nestimated objective = " << result.objective
                  << "\npose error X = " << x_error
                  << "\npose error Y = " << y_error
                  << "\nsuccessful = " << (successful ? "true" : "false");
        if (cli.common.retry_tol_set) {
            std::cout << "\nbackend used = " << result.backend_name
                      << "\nbackend attempts = " << result.attempts;
        }
        std::cout << "\n";

        axyb::print_matrix(result.X, "X_est");
        axyb::print_matrix(problem.X0, "X_gt");
        axyb::print_matrix(result.Y, "Y_est");
        axyb::print_matrix(problem.Y0, "Y_gt");
        return successful ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
