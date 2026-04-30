#include "axyb/axyb_solver.hpp"
#include "cli.hpp"

#include <tbb/global_control.h>

#include <exception>
#include <iostream>
#include <memory>
#include <random>

int main(int argc, char** argv) {
    try {
        auto cli = axyb::parse_common_cli(argc, argv, false);
        if (cli.help) {
            axyb::print_common_help(argv[0], false);
            return 0;
        }
        std::unique_ptr<tbb::global_control> gc;
        if (cli.solver.threads > 0) {
            gc.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, cli.solver.threads));
        }
        auto seed = axyb::choose_seed(cli);
        std::mt19937_64 rng(seed);
        auto p = axyb::make_random_problem(cli.len, cli.noise, rng);
        std::cout << "seed = " << seed << "\nbackend = " << cli.solver.backend_name;
        if (cli.retry_tol_set) std::cout << "\nretry_tol = " << cli.retry_tol;
        std::cout << "\ng_GT = [";
        for (int i = 0; i < 6; ++i) std::cout << (i ? ", " : "") << p.g_ground_truth[i];
        std::cout << "]\ninitial objective = " << axyb::J_AXYB(p.As, p.Bs, p.X0, p.Y0) << "\n";
        auto r = axyb::AXYB_complete_grobner_with_retry(p.As, p.Bs, cli.mu, cli.solver, cli.retry_tol_set ? cli.retry_tol : -1.0);
        if (cli.retry_tol_set) std::cout << "backend used = " << r.backend_name << "\nbackend attempts = " << r.attempts << "\n";
        axyb::print_matrix(r.X, "X1");
        axyb::print_matrix(p.X0, "X0");
        axyb::print_matrix(r.Y, "Y1");
        axyb::print_matrix(p.Y0, "Y0");
        std::cout << "f1 = " << r.objective << "\npose error X = " << axyb::pose_error(r.X, p.X0) << ", Y = " << axyb::pose_error(r.Y, p.Y0) << "\nroots evaluated = " << r.roots.size() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
