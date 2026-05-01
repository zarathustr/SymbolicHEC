#include "axyb/axyb_solver.hpp"
#include "axyb/measurement_io.hpp"
#include "cli.hpp"

#include <tbb/global_control.h>

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct LoopCli {
    axyb::CommonCli common;
    std::string output_meas;
    std::string input_meas;
    bool help = false;
};

void print_help(const char* exe) {
    axyb::print_common_help(exe, true);
    std::cout
        << "\nAdditional options:\n"
        << "  --output_meas PATH  Write generated measurement sequences to one file\n"
        << "  --input_meas PATH   Read generated measurement sequences from one file\n"
        << "\nNotes:\n"
        << "  `--output_meas` is used only with RNG-generated synthetic problems.\n"
        << "  With `--input_meas`, the loop reads every stored problem from the file and\n"
        << "  ignores `--iters`, `--len`, `--noise`, and `--seed`.\n";
}

LoopCli parse_loop_cli(int argc, char** argv) {
    LoopCli cli;
    std::vector<char*> common_argv;
    common_argv.reserve(static_cast<size_t>(argc));
    common_argv.push_back(argc ? argv[0] : const_cast<char*>("test_AXYB_grobner_solver_loop"));
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto val = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + name);
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") {
            cli.help = true;
        } else if (arg == "--output_meas") {
            cli.output_meas = val("--output_meas");
        } else if (arg == "--input_meas") {
            cli.input_meas = val("--input_meas");
        } else {
            common_argv.push_back(argv[i]);
        }
    }
    if (!cli.help) {
        if (!cli.output_meas.empty() && !cli.input_meas.empty()) {
            throw std::runtime_error("cannot use --output_meas and --input_meas together");
        }
        cli.common = axyb::parse_common_cli(static_cast<int>(common_argv.size()), common_argv.data(), true);
        if (cli.common.iters < 0) throw std::runtime_error("--iters must be >= 0");
    }
    return cli;
}

} // namespace

int main(int argc, char** argv) {
    try {
        LoopCli cli = parse_loop_cli(argc, argv);
        if (cli.help) {
            print_help(argv[0]);
            return 0;
        }

        std::unique_ptr<tbb::global_control> gc;
        if (cli.common.solver.threads > 0) {
            gc.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, cli.common.solver.threads));
        }

        std::ofstream out;
        if (!cli.output_meas.empty()) {
            out.open(cli.output_meas);
            if (!out) throw std::runtime_error("cannot open output measurement file " + cli.output_meas);
            out << std::setprecision(17);
            axyb::write_problem_sequence_header(out, cli.common.iters);
        }

        std::vector<axyb::ProblemData> input_problems;
        std::mt19937_64 rng;
        bool use_input = !cli.input_meas.empty();
        unsigned long long seed = 0;
        if (use_input) {
            input_problems = axyb::load_problem_sequence(cli.input_meas);
            if (input_problems.empty()) throw std::runtime_error("no problems found in input measurement file " + cli.input_meas);
        } else {
            seed = axyb::choose_seed(cli.common);
            rng.seed(seed);
        }

        if (use_input) {
            std::cout << "input_meas = " << cli.input_meas;
        } else {
            std::cout << "seed = " << seed;
        }
        std::cout << "\nbackend = " << cli.common.solver.backend_name;
        if (cli.common.retry_tol_set) std::cout << "\nretry_tol = " << cli.common.retry_tol;
        if (cli.common.solver.prescale != 1.0) std::cout << "\nprescale = " << cli.common.solver.prescale;
        if (cli.common.solver.asymmetric) std::cout << "\nasymmetric = true";
        if (!cli.output_meas.empty()) std::cout << "\noutput_meas = " << cli.output_meas;
        std::cout << "\n";

        const int problem_count = use_input ? static_cast<int>(input_problems.size()) : cli.common.iters;
        for (int j = 1; j <= problem_count; ++j) {
            axyb::ProblemData generated_problem;
            const axyb::ProblemData* problem = nullptr;
            if (use_input) {
                problem = &input_problems[static_cast<size_t>(j - 1)];
            } else {
                generated_problem = axyb::make_random_problem(cli.common.len, cli.common.noise, rng);
                problem = &generated_problem;
                if (out) axyb::write_problem(out, j, generated_problem);
            }
            auto r = axyb::AXYB_complete_grobner_with_retry(problem->As, problem->Bs, cli.common.mu, cli.common.solver,
                                                            cli.common.retry_tol_set ? cli.common.retry_tol : -1.0);
            std::cout << "iter " << j << ": objective = " << r.objective
                      << ", pose error X: " << axyb::pose_error(r.X, problem->X0)
                      << ", Y: " << axyb::pose_error(r.Y, problem->Y0);
            if (cli.common.retry_tol_set) std::cout << ", backend: " << r.backend_name << ", attempts: " << r.attempts;
            std::cout << "\n";
            axyb::print_dense_solve_diagnostics(r);
        }

        if (out) axyb::write_problem_sequence_footer(out);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
