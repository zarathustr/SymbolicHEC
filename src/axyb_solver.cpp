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

constexpr double kActionScale = 5e-3;

Matrix dense_from_sparse(const SparseMatrixCSC& sparse) {
    Matrix dense(sparse.rows, sparse.cols);
    for (int col = 0; col < sparse.cols; ++col)
        for (int p = sparse.col_ptr[col]; p < sparse.col_ptr[col + 1]; ++p)
            dense(sparse.row_idx[p], col) = sparse.values[p];
    return dense;
}

void scale_sparse_inplace(SparseMatrixCSC& matrix, double alpha) {
    if (alpha == 1.0) return;
    for (double& value : matrix.values) value *= alpha;
}

struct PreparedActionSystem {
    const TemplateData* tpl = nullptr;
    SparseMatrixCSC c0;
    Matrix rhs;
};

struct PreparedActionOutcome {
    std::vector<std::array<std::complex<double>, 6>> roots;
    DenseSolveDiagnostic dense_solve;
};

struct PreparedGrobnerProblem {
    const std::vector<Matrix>* As = nullptr;
    const std::vector<Matrix>* Bs = nullptr;
    Matrix at;
    Matrix bt;
    std::vector<PreparedActionSystem> actions;
};

PreparedActionSystem prepare_action_system(int action, const std::vector<double>& W, const SolverOptions& opts) {
    const auto& tpl = load_template_cached(opts.template_dir, action);
    auto coeff = map_coefficients(tpl, W);
    auto c0 = build_c0_from_template(tpl, coeff);
    auto c1t = build_c1_from_template(tpl, coeff);
    scale_sparse_inplace(c0, opts.prescale);
    scale_sparse_inplace(c1t, opts.prescale);
    Matrix rhs = opts.asymmetric ? dense_from_sparse(c1t) : compute_BB(c0, c1t, kActionScale);
    return {&tpl, std::move(c0), std::move(rhs)};
}

PreparedActionOutcome solve_prepared_action(const PreparedActionSystem& prepared, double mu, const SolverOptions& opts) {
    const TemplateData& tpl = *prepared.tpl;
    DenseSolveDiagnostic diag;
    DenseSolveDiagnostic* diag_ptr = opts.verbose ? &diag : nullptr;
    Matrix C1 = solve_template_system(prepared.c0, prepared.rhs, mu, kActionScale, tpl, opts, diag_ptr);
    if (diag_ptr) diag.action = tpl.action;
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
    PreparedActionOutcome outcome;
    if (diag_ptr) outcome.dense_solve = diag;
    outcome.roots.resize(tpl.m);
    for (int c = 0; c < tpl.m; ++c)
        for (int k = 0; k < 6; ++k)
            outcome.roots[static_cast<size_t>(c)][k] = tpl.sol_sources[k] < 0 ? evals[c] : V(tpl.sol_sources[k], c);
    return outcome;
}

bool retry_accept(const SolveResult& result, double retry_tol) {
    return std::isfinite(result.objective) && result.objective <= retry_tol;
}

bool better_objective(double candidate, double current_best) {
    if (!std::isfinite(candidate)) return false;
    if (!std::isfinite(current_best)) return true;
    return candidate < current_best;
}

bool backend_usable_for_options(LinearBackend backend, const SolverOptions& opts) {
    return !opts.asymmetric || backend_supports_asymmetric(backend);
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

PreparedGrobnerProblem prepare_grobner_problem(const std::vector<Matrix>& As, const std::vector<Matrix>& Bs,
                                               const SolverOptions& opts) {
    PreparedGrobnerProblem prepared;
    prepared.As = &As;
    prepared.Bs = &Bs;

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
    prepared.at = solve_dense(AeqM, v2m(Ares, 6, 9), false);
    prepared.bt = solve_dense(AeqM, v2m(bsum, 6, 1), false);
    for (auto& v : prepared.at.a) v = -v;

    std::vector<double> Wt(60), Wr(240);
    for (size_t i = 0; i < As.size(); ++i) {
        auto RA = Rof(As[i]), RB = Rof(Bs[i]);
        auto tA = tof(As[i]), tB = tof(Bs[i]);
        addsc(Wt, W_J_trans_cayley_gY_func(flatten_col_major(RA), flatten_col_major(RB), tA, tB,
                                           flatten_col_major(prepared.at), flatten_col_major(prepared.bt)),
              invlen);
        addsc(Wr, W_J_rot_gXgY_func(flatten_col_major(RA), flatten_col_major(RB)), invlen);
    }
    auto W = W_AXYB_gXgY_func(Wt, Wr);
    std::vector<int> action_ids;
    for (int action = opts.template_first; action <= opts.template_last; ++action) action_ids.push_back(action);
    prepared.actions.resize(action_ids.size());
    if (opts.parallel_templates && action_ids.size() > 1) {
        tbb::parallel_for(0, static_cast<int>(action_ids.size()), [&](int i) {
            prepared.actions[static_cast<size_t>(i)] = prepare_action_system(action_ids[static_cast<size_t>(i)], W, opts);
        });
    } else {
        for (size_t i = 0; i < action_ids.size(); ++i) prepared.actions[i] = prepare_action_system(action_ids[i], W, opts);
    }
    return prepared;
}

SolveResult solve_prepared_grobner(const PreparedGrobnerProblem& prepared, double mu, const SolverOptions& opts) {
    std::vector<PreparedActionOutcome> action_outcomes(prepared.actions.size());
    if (opts.parallel_templates && prepared.actions.size() > 1) {
        tbb::parallel_for(0, static_cast<int>(prepared.actions.size()), [&](int i) {
            action_outcomes[static_cast<size_t>(i)] = solve_prepared_action(prepared.actions[static_cast<size_t>(i)], mu, opts);
        });
    } else {
        for (size_t i = 0; i < prepared.actions.size(); ++i) action_outcomes[i] = solve_prepared_action(prepared.actions[i], mu, opts);
    }

    std::vector<std::array<std::complex<double>, 6>> roots;
    std::vector<DenseSolveDiagnostic> dense_solve_diagnostics;
    for (auto& outcome : action_outcomes) {
        roots.insert(roots.end(), outcome.roots.begin(), outcome.roots.end());
        if (outcome.dense_solve.used_dense_solve) dense_solve_diagnostics.push_back(outcome.dense_solve);
    }

    SolveResult best;
    best.objective = std::numeric_limits<double>::infinity();
    best.roots = roots;
    best.backend_name = opts.backend_name;
    best.attempts = 1;
    best.dense_solve_diagnostics = std::move(dense_solve_diagnostics);
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
                tx[r] += prepared.at(r, k) * ry[k];
                ty[r] += prepared.at(r + 3, k) * ry[k];
            }
            tx[r] -= prepared.bt(r, 0);
            ty[r] -= prepared.bt(r + 3, 0);
        }
        Matrix X = makeT(RX, tx), Y = makeT(RY, ty);
        double f = J_AXYB(*prepared.As, *prepared.Bs, X, Y);
        if (std::isfinite(f) && f < best.objective) {
            best.objective = f;
            best.X = std::move(X);
            best.Y = std::move(Y);
        }
    }
    return best;
}

BackendAttemptOutcome run_backend_attempt(const PreparedGrobnerProblem& prepared, double mu, const SolverOptions& opts,
                                          const BackendAttemptSpec& spec) {
    SolverOptions current_opts = opts;
    current_opts.backend = spec.backend;
    current_opts.backend_name = spec.name;
    try {
        BackendAttemptOutcome outcome;
        outcome.ok = true;
        outcome.result = solve_prepared_grobner(prepared, mu, current_opts);
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
    return solve_prepared_grobner(prepare_grobner_problem(As, Bs, opts), mu, opts);
}

SolveResult AXYB_complete_grobner_with_retry(const std::vector<Matrix>& As, const std::vector<Matrix>& Bs, double mu,
                                             const SolverOptions& opts, double retry_tol) {
    if (retry_tol < 0.0) return solve_once(As, Bs, mu, opts);

    PreparedGrobnerProblem prepared = prepare_grobner_problem(As, Bs, opts);
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
    BackendAttemptOutcome first = run_backend_attempt(prepared, mu, opts, requested);
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
        if (entry.first != opts.backend && backend_usable_for_options(entry.first, opts))
            remaining.push_back({entry.first, entry.second});

    if (!opts.parallel_backend_retries || remaining.size() <= 1) {
        for (const auto& spec : remaining) {
            ++attempts;
            BackendAttemptOutcome current = run_backend_attempt(prepared, mu, opts, spec);
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
            outcomes[static_cast<size_t>(i)] = run_backend_attempt(prepared, mu, opts, remaining[static_cast<size_t>(i)]);
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

void print_dense_solve_diagnostics(const SolveResult& result) {
    if (result.dense_solve_diagnostics.empty()) return;
    const std::streamsize old_precision = std::cout.precision();
    std::cout << "dense solve determinants:\n" << std::setprecision(18);
    for (const auto& diag : result.dense_solve_diagnostics) {
        std::cout << "  x" << diag.action << ": det(C0) = ";
        if (!diag.c0_determinant.computed) {
            std::cout << "unavailable";
        } else if (diag.c0_determinant.sign == 0) {
            std::cout << "0";
        } else {
            std::cout << (diag.c0_determinant.sign > 0 ? "+" : "-")
                      << diag.c0_determinant.mantissa
                      << "e" << diag.c0_determinant.exponent10;
        }
        std::cout << "\n";
    }
    std::cout << std::setprecision(old_precision);
}

} // namespace axyb
