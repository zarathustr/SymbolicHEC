#include "axyb/axyb_solver.hpp"
#include "axyb/generated_symbolics.hpp"
#include "cli.hpp"

#include <tbb/global_control.h>

#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct BenchmarkCli {
    axyb::CommonCli common;
    int repeats = 3;
    int warmup = 1;
    bool help = false;
};

struct BenchmarkCase {
    int action = 0;
    double scale = 5e-3;
    axyb::TemplateData tpl;
    axyb::SparseMatrixCSC c0;
    axyb::Matrix bb;
};

struct BenchmarkResult {
    std::string backend;
    bool ok = false;
    double total_ms = 0.0;
    double avg_ms = 0.0;
    double checksum = 0.0;
    std::vector<double> template_avg_ms;
    std::string error;
};

void print_help(const char* exe) {
    std::cout
        << "Usage: " << exe << " [options]\n\n"
        << "Benchmarks solve_template_system() across all linear-system backends.\n"
        << "Coefficient generation and sparse template assembly are shared, so the timing\n"
        << "focuses on backend solve cost.\n\n"
        << "  --data-dir DIR       template directory; default is executable_dir/templates\n"
        << "  --template SEL       benchmark one template or inclusive range (x3 or x2-x4)\n"
        << "  --templates A-B      alias for inclusive template range selection\n"
        << "  --threads N          limit oneTBB workers\n"
        << "  --len N              motion pairs used to generate the benchmark problem, default 5\n"
        << "  --noise X            synthetic noise level, default 5e-2\n"
        << "  --mu X               Tikhonov/normal-equation regularization, default 5e-9\n"
        << "  --seed N             reproducible seed\n"
        << "  --pcg-tol X          default 1e-8\n"
        << "  --pcg-maxit N        default 300\n"
        << "  --no-fallback        disable dense direct fallback for iterative methods\n"
        << "  --repeats N          timed passes per backend, default 3\n"
        << "  --warmup N           untimed passes per backend, default 1\n"
        << "  --help\n\n"
        << "Benchmarked backends: tbb-pcg, tbb-block-jacobi, lapack-posv, lapack-gesv,\n"
        << "                      eigen-llt, eigen-ldlt, eigen-partial-piv-lu, eigen-sparse-lu\n";
}

BenchmarkCli parse_benchmark_cli(int argc, char** argv) {
    BenchmarkCli cli;
    std::vector<char*> common_argv;
    common_argv.reserve(static_cast<size_t>(argc));
    common_argv.push_back(argc ? argv[0] : const_cast<char*>("benchmark_AXYB_linear_backends"));
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto val = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + name);
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") {
            cli.help = true;
        } else if (arg == "--repeats") {
            cli.repeats = std::stoi(val("--repeats"));
        } else if (arg == "--warmup") {
            cli.warmup = std::stoi(val("--warmup"));
        } else if (arg == "--backend") {
            throw std::runtime_error("--backend is not supported by the benchmark executable; it benchmarks all backends");
        } else if (arg == "--outer-parallel" || arg == "--no-template-parallel") {
            throw std::runtime_error("template-level parallel selection is not used by the linear benchmark");
        } else if (arg == "--no-backend-parallel") {
            throw std::runtime_error("retry-backend parallel selection is not used by the linear benchmark");
        } else if (arg == "--retry_tol") {
            throw std::runtime_error("--retry_tol is not used by the linear benchmark");
        } else {
            common_argv.push_back(argv[i]);
        }
    }
    if (cli.repeats < 1) throw std::runtime_error("--repeats must be >= 1");
    if (cli.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    if (!cli.help) cli.common = axyb::parse_common_cli(static_cast<int>(common_argv.size()), common_argv.data(), false);
    return cli;
}

std::vector<int> selected_actions(const axyb::SolverOptions& opts) {
    std::vector<int> actions;
    for (int action = opts.template_first; action <= opts.template_last; ++action) actions.push_back(action);
    return actions;
}

axyb::Matrix rotation_of(const axyb::Matrix& t) {
    axyb::Matrix r(3, 3);
    for (int c = 0; c < 3; ++c)
        for (int row = 0; row < 3; ++row)
            r(row, c) = t(row, c);
    return r;
}

std::vector<double> translation_of(const axyb::Matrix& t) {
    return {t(0, 3), t(1, 3), t(2, 3)};
}

void add_scaled(std::vector<double>& dst, const std::vector<double>& src, double alpha) {
    if (dst.empty()) dst.assign(src.size(), 0.0);
    for (size_t i = 0; i < src.size(); ++i) dst[i] += alpha * src[i];
}

axyb::Matrix vector_to_matrix(const std::vector<double>& values, int rows, int cols) {
    axyb::Matrix out(rows, cols);
    out.a = values;
    return out;
}

std::vector<double> compute_problem_weights(const axyb::ProblemData& problem) {
    const double invlen = 1.0 / static_cast<double>(problem.As.size());
    std::vector<double> aeq(36), ares(54), bsum(6);
    for (size_t i = 0; i < problem.As.size(); ++i) {
        auto ra = rotation_of(problem.As[i]);
        auto ta = translation_of(problem.As[i]);
        auto tb = translation_of(problem.Bs[i]);
        add_scaled(aeq, axyb::A_eq_tXtY_func(axyb::flatten_col_major(ra)), invlen);
        add_scaled(ares, axyb::A_res_eq_tXtY_RY_func(axyb::flatten_col_major(ra), ta, tb), invlen);
        add_scaled(bsum, axyb::b_eq_tXtY_func(axyb::flatten_col_major(ra), ta), invlen);
    }
    axyb::Matrix aeq_matrix = vector_to_matrix(aeq, 6, 6);
    axyb::Matrix at = axyb::solve_dense(aeq_matrix, vector_to_matrix(ares, 6, 9), false);
    axyb::Matrix bt = axyb::solve_dense(aeq_matrix, vector_to_matrix(bsum, 6, 1), false);
    for (double& value : at.a) value = -value;
    std::vector<double> wt(60), wr(240);
    for (size_t i = 0; i < problem.As.size(); ++i) {
        auto ra = rotation_of(problem.As[i]);
        auto rb = rotation_of(problem.Bs[i]);
        auto ta = translation_of(problem.As[i]);
        auto tbv = translation_of(problem.Bs[i]);
        add_scaled(wt, axyb::W_J_trans_cayley_gY_func(axyb::flatten_col_major(ra), axyb::flatten_col_major(rb), ta, tbv,
                                                      axyb::flatten_col_major(at), axyb::flatten_col_major(bt)),
                   invlen);
        add_scaled(wr, axyb::W_J_rot_gXgY_func(axyb::flatten_col_major(ra), axyb::flatten_col_major(rb)), invlen);
    }
    return axyb::W_AXYB_gXgY_func(wt, wr);
}

std::vector<BenchmarkCase> prepare_cases(const axyb::SolverOptions& opts, const std::vector<double>& weights) {
    std::vector<BenchmarkCase> cases;
    for (int action : selected_actions(opts)) {
        BenchmarkCase bench_case;
        bench_case.action = action;
        bench_case.tpl = axyb::load_template(opts.template_dir, action);
        auto coeff = axyb::map_coefficients(bench_case.tpl, weights);
        bench_case.c0 = axyb::build_sparse_from_template(bench_case.tpl.n, bench_case.tpl.n, bench_case.tpl.c0_linear,
                                                         bench_case.tpl.c0_coeff, coeff);
        auto c1 = axyb::build_sparse_from_template(bench_case.tpl.n, bench_case.tpl.m, bench_case.tpl.c1_linear,
                                                   bench_case.tpl.c1_coeff, coeff);
        bench_case.bb = axyb::compute_BB(bench_case.c0, c1, bench_case.scale);
        cases.push_back(std::move(bench_case));
    }
    return cases;
}

double checksum_of(const axyb::Matrix& x) {
    if (x.a.empty()) return 0.0;
    return x.a.front() + x.a.back();
}

void run_pass(const std::vector<BenchmarkCase>& cases, const axyb::SolverOptions& opts, double mu,
              std::vector<double>* per_case_ms, double* checksum) {
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& bench_case = cases[i];
        auto start = std::chrono::steady_clock::now();
        axyb::Matrix x = axyb::solve_template_system(bench_case.c0, bench_case.bb, mu, bench_case.scale, bench_case.tpl, opts);
        auto stop = std::chrono::steady_clock::now();
        if (per_case_ms) {
            (*per_case_ms)[i] += std::chrono::duration<double, std::milli>(stop - start).count();
        }
        if (checksum) *checksum += checksum_of(x);
    }
}

BenchmarkResult benchmark_backend(const std::string& backend_name, const std::vector<BenchmarkCase>& cases,
                                  const BenchmarkCli& cli) {
    BenchmarkResult result;
    result.backend = backend_name;
    result.template_avg_ms.assign(cases.size(), 0.0);
    axyb::SolverOptions opts = cli.common.solver;
    opts.backend_name = backend_name;
    opts.backend = axyb::parse_backend(backend_name);
    try {
        for (int i = 0; i < cli.warmup; ++i) run_pass(cases, opts, cli.common.mu, nullptr, nullptr);
        for (int i = 0; i < cli.repeats; ++i) run_pass(cases, opts, cli.common.mu, &result.template_avg_ms, &result.checksum);
        result.ok = true;
        result.total_ms = std::accumulate(result.template_avg_ms.begin(), result.template_avg_ms.end(), 0.0);
        result.avg_ms = result.total_ms / static_cast<double>(cli.repeats * static_cast<int>(cases.size()));
        for (double& value : result.template_avg_ms) value /= static_cast<double>(cli.repeats);
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

void print_action_list(const std::vector<BenchmarkCase>& cases) {
    std::cout << "templates = ";
    for (size_t i = 0; i < cases.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << 'x' << cases[i].action;
    }
    std::cout << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        BenchmarkCli cli = parse_benchmark_cli(argc, argv);
        if (cli.help) {
            print_help(argv[0]);
            return 0;
        }
        std::unique_ptr<tbb::global_control> control;
        if (cli.common.solver.threads > 0) {
            control.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, cli.common.solver.threads));
        }
        unsigned long long seed = axyb::choose_seed(cli.common);
        std::mt19937_64 rng(seed);
        axyb::ProblemData problem = axyb::make_random_problem(cli.common.len, cli.common.noise, rng);
        std::vector<double> weights = compute_problem_weights(problem);
        std::vector<BenchmarkCase> cases = prepare_cases(cli.common.solver, weights);
        std::vector<std::string> backends = {
            "tbb-pcg",
            "tbb-block-jacobi",
            "lapack-posv",
            "lapack-gesv",
            "eigen-llt",
            "eigen-ldlt",
            "eigen-partial-piv-lu",
            "eigen-sparse-lu"
        };

        std::cout << "seed = " << seed << "\n";
        std::cout << "data_dir = " << cli.common.solver.template_dir << "\n";
        print_action_list(cases);
        std::cout << "repeats = " << cli.repeats << "\n";
        std::cout << "warmup = " << cli.warmup << "\n";
        std::cout << "len = " << cli.common.len << ", noise = " << cli.common.noise << ", mu = " << cli.common.mu << "\n";
        std::cout << std::left << std::setw(22) << "backend"
                  << std::right << std::setw(14) << "total ms"
                  << std::setw(14) << "avg ms"
                  << std::setw(18) << "checksum"
                  << std::setw(12) << "status" << "\n";
        for (const auto& backend_name : backends) {
            BenchmarkResult result = benchmark_backend(backend_name, cases, cli);
            if (result.ok) {
                std::cout << std::left << std::setw(22) << result.backend
                          << std::right << std::setw(14) << std::fixed << std::setprecision(3) << result.total_ms
                          << std::setw(14) << result.avg_ms
                          << std::setw(18) << std::setprecision(6) << std::scientific << result.checksum
                          << std::setw(12) << "ok" << "\n";
                std::cout << "  template avg ms:";
                for (size_t i = 0; i < cases.size(); ++i) {
                    std::cout << " x" << cases[i].action << "=" << std::fixed << std::setprecision(3) << result.template_avg_ms[i];
                }
                std::cout << "\n";
            } else {
                std::cout << std::left << std::setw(22) << result.backend
                          << std::right << std::setw(14) << "-"
                          << std::setw(14) << "-"
                          << std::setw(18) << "-"
                          << std::setw(12) << "error" << "\n";
                std::cout << "  error: " << result.error << "\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
