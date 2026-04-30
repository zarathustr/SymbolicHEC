#include "axyb/axyb_solver.hpp"
#include "axyb/generated_symbolics.hpp"
#include "axyb/template_data.hpp"

#include <tbb/parallel_for.h>

#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>

namespace axyb {
namespace {

Matrix makeT(const Matrix& R, const std::vector<double>& t) {
    Matrix T = Matrix::eye(4);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            T(r, c) = R(r, c);
    for (int r = 0; r < 3; ++r) T(r, 3) = t[r];
    return T;
}

std::vector<double> randv(int n, std::mt19937_64& rng) {
    std::normal_distribution<double> d;
    std::vector<double> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

Matrix randm(int r, int c, std::mt19937_64& rng) {
    std::normal_distribution<double> d;
    Matrix M(r, c);
    for (auto& x : M.a) x = d(rng);
    return M;
}

Matrix invT(const Matrix& T) {
    Matrix Rt(3, 3);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            Rt(c, r) = T(r, c);
    std::vector<double> t(3), ti(3);
    for (int r = 0; r < 3; ++r) t[r] = T(r, 3);
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            ti[i] -= Rt(i, k) * t[k];
    return makeT(Rt, ti);
}

Matrix Rof(const Matrix& T) {
    Matrix R(3, 3);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            R(r, c) = T(r, c);
    return R;
}

std::vector<double> tof(const Matrix& T) { return {T(0, 3), T(1, 3), T(2, 3)}; }

void addsc(std::vector<double>& d, const std::vector<double>& s, double a) {
    if (d.empty()) d.assign(s.size(), 0.0);
    for (size_t i = 0; i < s.size(); ++i) d[i] += a * s[i];
}

Matrix v2m(const std::vector<double>& v, int r, int c) {
    Matrix M(r, c);
    M.a = v;
    return M;
}

Matrix cayleyL(const std::vector<double>& g) {
    Matrix G = skew(g), I = Matrix::eye(3);
    return matmul(add(I, G, -1), inverse(add(I, G, 1)));
}

Matrix cayleyR(const std::vector<double>& g) {
    Matrix G = skew(g), I = Matrix::eye(3);
    return matmul(inverse(add(I, G, 1)), add(I, G, -1));
}

std::vector<std::array<std::complex<double>, 6>> solve_action(int action, const std::vector<double>& W, double mu,
                                                              const SolverOptions& opts) {
    auto tpl = load_template(opts.template_dir, action);
    auto coeff = map_coefficients(tpl, W);
    auto C0 = build_sparse_from_template(tpl.n, tpl.n, tpl.c0_linear, tpl.c0_coeff, coeff);
    auto C1t = build_sparse_from_template(tpl.n, tpl.m, tpl.c1_linear, tpl.c1_coeff, coeff);
    double sc = 5e-3;
    Matrix BB = compute_BB(C0, C1t, sc);
    Matrix C1 = solve_template_system(C0, BB, mu, sc, tpl, opts);
    Matrix AM(tpl.m, tpl.m);
    for (int r = 0; r < tpl.m; ++r) {
        int src = static_cast<int>(tpl.am_ind[r]);
        if (src < tpl.tail_count) {
            int rr = tpl.n - tpl.tail_count + src;
            for (int c = 0; c < tpl.m; ++c) AM(r, c) = -C1(rr, c);
        } else {
            int er = src - tpl.tail_count;
            AM(r, er) = 1;
        }
    }
    std::vector<std::complex<double>> evals;
    CMatrix V = eig_complex(AM, evals);
    std::vector<std::array<std::complex<double>, 6>> roots(tpl.m);
    for (int c = 0; c < tpl.m; ++c)
        for (int k = 0; k < 6; ++k)
            roots[c][k] = tpl.sol_sources[k] < 0 ? evals[c] : V(tpl.sol_sources[k], c);
    return roots;
}

bool retry_accept(const SolveResult& result, double retry_tol) {
    return std::isfinite(result.objective) && result.objective <= retry_tol;
}

bool better_objective(double candidate, double current_best) {
    if (!std::isfinite(candidate)) return false;
    if (!std::isfinite(current_best)) return true;
    return candidate < current_best;
}

const std::vector<std::pair<LinearBackend, const char*>>& retry_backends() {
    static const std::vector<std::pair<LinearBackend, const char*>> order = {
        {LinearBackend::LapackDenseLU, "lapack-gesv"},
        {LinearBackend::TbbBlockJacobi, "tbb-block-jacobi"},
        {LinearBackend::EigenLLT, "eigen-llt"},
        {LinearBackend::EigenPartialPivLU, "eigen-partial-piv-lu"},
        {LinearBackend::EigenLDLT, "eigen-ldlt"},
        {LinearBackend::EigenSparseLU, "eigen-sparse-lu"},
        {LinearBackend::TbbPcg, "tbb-pcg"}
    };
    return order;
}

struct BackendAttemptSpec {
    LinearBackend backend;
    std::string name;
};

struct BackendAttemptOutcome {
    bool ok = false;
    SolveResult result;
    std::exception_ptr error;
};

BackendAttemptOutcome run_backend_attempt(const std::vector<Matrix>& As, const std::vector<Matrix>& Bs, double mu,
                                          const SolverOptions& opts, const BackendAttemptSpec& spec) {
    SolverOptions current_opts = opts;
    current_opts.backend = spec.backend;
    current_opts.backend_name = spec.name;
    try {
        BackendAttemptOutcome outcome;
        outcome.ok = true;
        outcome.result = AXYB_complete_grobner(As, Bs, mu, current_opts);
        outcome.result.backend_name = spec.name;
        return outcome;
    } catch (...) {
        BackendAttemptOutcome outcome;
        outcome.error = std::current_exception();
        return outcome;
    }
}

SolveResult solve_once(const std::vector<Matrix>& As, const std::vector<Matrix>& Bs, double mu, const SolverOptions& opts) {
    SolveResult result = AXYB_complete_grobner(As, Bs, mu, opts);
    result.backend_name = opts.backend_name;
    result.attempts = 1;
    return result;
}

} // namespace

ProblemData make_random_problem(int len, double noise, std::mt19937_64& rng) {
    ProblemData p;
    p.As.resize(len);
    p.Bs.resize(len);
    auto gx = randv(3, rng), gy = randv(3, rng);
    for (int i = 0; i < 3; ++i) {
        p.g_ground_truth[i] = gx[i];
        p.g_ground_truth[i + 3] = gy[i];
    }
    p.X0 = makeT(rod2dcm(gx), randv(3, rng));
    p.Y0 = makeT(rod2dcm(gy), randv(3, rng));
    Matrix iY = invT(p.Y0);
    std::normal_distribution<double> d;
    for (int i = 0; i < len; ++i) {
        Matrix A = makeT(orthonormalize3(randm(3, 3, rng)), randv(3, rng));
        Matrix B = matmul(matmul(iY, A), p.X0);
        Matrix RA = Rof(A), RB = Rof(B), NA = randm(3, 3, rng), NB = randm(3, 3, rng);
        for (size_t k = 0; k < RA.a.size(); ++k) {
            RA.a[k] += noise * NA.a[k];
            RB.a[k] += noise * NB.a[k];
        }
        RA = orthonormalize3(RA);
        RB = orthonormalize3(RB);
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) {
                A(r, c) = RA(r, c);
                B(r, c) = RB(r, c);
            }
        for (int r = 0; r < 3; ++r) {
            A(r, 3) += noise * d(rng);
            B(r, 3) += noise * d(rng);
        }
        p.As[i] = A;
        p.Bs[i] = B;
    }
    return p;
}

double J_AXYB(const std::vector<Matrix>& As, const std::vector<Matrix>& Bs, const Matrix& X, const Matrix& Y) {
    double J = 0.0;
    for (size_t i = 0; i < As.size(); ++i)
        J += frobenius_squared(add(matmul(As[i], X), matmul(Y, Bs[i]), -1)) / static_cast<double>(As.size());
    return J;
}

double pose_error(const Matrix& X, const Matrix& X0) { return frobenius_squared(add(X, X0, -1)); }

SolveResult AXYB_complete_grobner(const std::vector<Matrix>& As, const std::vector<Matrix>& Bs, double mu,
                                  const SolverOptions& opts) {
    double invlen = 1.0 / static_cast<double>(As.size());
    std::vector<double> Aeq(36), Ares(54), bsum(6);
    for (size_t i = 0; i < As.size(); ++i) {
        auto RA = Rof(As[i]);
        auto tA = tof(As[i]), tB = tof(Bs[i]);
        addsc(Aeq, A_eq_tXtY_func(flatten_col_major(RA)), invlen);
        addsc(Ares, A_res_eq_tXtY_RY_func(flatten_col_major(RA), tA, tB), invlen);
        addsc(bsum, b_eq_tXtY_func(flatten_col_major(RA), tA), invlen);
    }
    Matrix AeqM = v2m(Aeq, 6, 6);
    Matrix At = solve_dense(AeqM, v2m(Ares, 6, 9), false);
    Matrix bt = solve_dense(AeqM, v2m(bsum, 6, 1), false);
    for (auto& v : At.a) v = -v;
    std::vector<double> Wt(60), Wr(240);
    for (size_t i = 0; i < As.size(); ++i) {
        auto RA = Rof(As[i]), RB = Rof(Bs[i]);
        auto tA = tof(As[i]), tB = tof(Bs[i]);
        addsc(Wt, W_J_trans_cayley_gY_func(flatten_col_major(RA), flatten_col_major(RB), tA, tB,
                                           flatten_col_major(At), flatten_col_major(bt)),
              invlen);
        addsc(Wr, W_J_rot_gXgY_func(flatten_col_major(RA), flatten_col_major(RB)), invlen);
    }
    auto W = W_AXYB_gXgY_func(Wt, Wr);
    std::vector<int> actions;
    for (int action = opts.template_first; action <= opts.template_last; ++action) actions.push_back(action);
    std::vector<std::vector<std::array<std::complex<double>, 6>>> ar(actions.size());
    if (opts.parallel_templates && actions.size() > 1) {
        tbb::parallel_for(0, static_cast<int>(actions.size()),
                          [&](int i) { ar[static_cast<size_t>(i)] = solve_action(actions[static_cast<size_t>(i)], W, mu, opts); });
    } else {
        for (size_t i = 0; i < actions.size(); ++i) ar[i] = solve_action(actions[i], W, mu, opts);
    }
    std::vector<std::array<std::complex<double>, 6>> roots;
    for (auto& r : ar) roots.insert(roots.end(), r.begin(), r.end());
    SolveResult best;
    best.objective = std::numeric_limits<double>::infinity();
    best.roots = roots;
    best.backend_name = opts.backend_name;
    best.attempts = 1;
    for (auto& root : roots) {
        std::vector<double> gx(3), gy(3);
        for (int i = 0; i < 3; ++i) {
            gx[i] = root[i].real();
            gy[i] = root[i + 3].real();
        }
        Matrix RX = cayleyL(gx), RY = cayleyR(gy);
        auto ry = flatten_col_major(RY);
        std::vector<double> tx(3), ty(3);
        for (int r = 0; r < 3; ++r) {
            for (int k = 0; k < 9; ++k) {
                tx[r] += At(r, k) * ry[k];
                ty[r] += At(r + 3, k) * ry[k];
            }
            tx[r] -= bt(r, 0);
            ty[r] -= bt(r + 3, 0);
        }
        Matrix X = makeT(RX, tx), Y = makeT(RY, ty);
        double f = J_AXYB(As, Bs, X, Y);
        if (std::isfinite(f) && f < best.objective) {
            best.objective = f;
            best.X = std::move(X);
            best.Y = std::move(Y);
        }
    }
    return best;
}

SolveResult AXYB_complete_grobner_with_retry(const std::vector<Matrix>& As, const std::vector<Matrix>& Bs, double mu,
                                             const SolverOptions& opts, double retry_tol) {
    if (retry_tol < 0.0) return solve_once(As, Bs, mu, opts);

    SolveResult best;
    bool have_best = false;
    int attempts = 0;
    std::exception_ptr last_error;
    auto consider_success = [&](const SolveResult& current) {
        if (!have_best || better_objective(current.objective, best.objective)) {
            best = current;
            have_best = true;
        }
    };

    BackendAttemptSpec requested{opts.backend, opts.backend_name};
    ++attempts;
    BackendAttemptOutcome first = run_backend_attempt(As, Bs, mu, opts, requested);
    if (first.ok) {
        consider_success(first.result);
        if (retry_accept(first.result, retry_tol)) {
            SolveResult accepted = first.result;
            accepted.attempts = attempts;
            return accepted;
        }
    } else {
        last_error = first.error;
    }

    std::vector<BackendAttemptSpec> remaining;
    for (const auto& entry : retry_backends())
        if (entry.first != opts.backend)
            remaining.push_back({entry.first, entry.second});

    if (!opts.parallel_backend_retries || remaining.size() <= 1) {
        for (const auto& spec : remaining) {
            ++attempts;
            BackendAttemptOutcome current = run_backend_attempt(As, Bs, mu, opts, spec);
            if (!current.ok) {
                last_error = current.error;
                continue;
            }
            consider_success(current.result);
            if (retry_accept(current.result, retry_tol)) {
                SolveResult accepted = current.result;
                accepted.attempts = attempts;
                return accepted;
            }
        }
    } else {
        std::vector<BackendAttemptOutcome> outcomes(remaining.size());
        tbb::parallel_for(0, static_cast<int>(remaining.size()), [&](int i) {
            outcomes[static_cast<size_t>(i)] = run_backend_attempt(As, Bs, mu, opts, remaining[static_cast<size_t>(i)]);
        });
        attempts += static_cast<int>(remaining.size());
        for (size_t i = 0; i < outcomes.size(); ++i) {
            const auto& current = outcomes[i];
            if (!current.ok) {
                if (current.error) last_error = current.error;
                continue;
            }
            consider_success(current.result);
            if (retry_accept(current.result, retry_tol)) {
                SolveResult accepted = current.result;
                accepted.attempts = 2 + static_cast<int>(i);
                return accepted;
            }
        }
    }

    if (have_best) {
        best.attempts = attempts;
        return best;
    }
    if (last_error) std::rethrow_exception(last_error);
    throw std::runtime_error("no backend succeeded");
}

void print_matrix(const Matrix& M, const char* name) {
    std::cout << name << " =\n" << std::setprecision(10);
    for (int r = 0; r < M.rows; ++r) {
        for (int c = 0; c < M.cols; ++c) std::cout << std::setw(14) << M(r, c) << ' ';
        std::cout << '\n';
    }
}

} // namespace axyb
