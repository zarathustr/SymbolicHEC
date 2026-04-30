#include "axyb/linear_solver.hpp"
#include "axyb/lapack.hpp"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

#if AXYB_ENABLE_EIGEN
#include <Eigen/Dense>
#include <Eigen/Sparse>
#endif

namespace axyb {
namespace {
std::string lo(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double nrm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

void applyA(const SparseMatrixCSC& C0, double sc, double mu,
            const std::vector<double>& x, std::vector<double>& y, std::vector<double>& tmp) {
    C0.multiply(x, tmp);
    C0.transpose_multiply(tmp, y);
    for (size_t i = 0; i < y.size(); ++i) y[i] = sc * y[i] + mu * x[i];
}

struct Fac {
    int s = 0, e = 0, n = 0;
    bool chol = false;
    std::vector<double> A;
    std::vector<int> ipiv;
};

void solve_fac(const Fac& f, double* rhs, int nrhs, int ldb) {
    int info = 0;
    if (f.chol) {
        char u = 'L';
        dpotrs_(&u, &f.n, &nrhs, f.A.data(), &f.n, rhs, &ldb, &info);
    } else {
        char tr = 'N';
        dgetrs_(&tr, &f.n, &nrhs, f.A.data(), &f.n, f.ipiv.data(), rhs, &ldb, &info);
    }
    if (info) throw std::runtime_error("block solve failed");
}

std::vector<Fac> factor(const SparseMatrixCSC& C0, const TemplateData& tpl, double sc, double mu, double jit) {
    std::vector<Fac> F(tpl.blocks.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, tpl.blocks.size()), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t bi = r.begin(); bi != r.end(); ++bi) {
            int s = tpl.blocks[bi].first, e = tpl.blocks[bi].second, n = e - s;
            Fac f;
            f.s = s;
            f.e = e;
            f.n = n;
            f.A.assign(static_cast<size_t>(n) * n, 0.0);
            for (int j = 0; j < n; ++j) {
                for (int i = j; i < n; ++i) {
                    double v = sc * C0.column_dot(s + i, s + j) + (i == j ? mu + jit : 0.0);
                    f.A[static_cast<size_t>(j) * n + i] = v;
                    f.A[static_cast<size_t>(i) * n + j] = v;
                }
            }
            int info = 0;
            char u = 'L';
            dpotrf_(&u, &n, f.A.data(), &n, &info);
            if (info == 0) {
                f.chol = true;
            } else {
                f.chol = false;
                f.ipiv.resize(n);
                std::fill(f.A.begin(), f.A.end(), 0.0);
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        f.A[static_cast<size_t>(j) * n + i] = sc * C0.column_dot(s + i, s + j) + (i == j ? mu + jit : 0.0);
                dgetrf_(&n, &n, f.A.data(), &n, f.ipiv.data(), &info);
                if (info) throw std::runtime_error("block factor failed");
            }
            F[bi] = std::move(f);
        }
    });
    return F;
}

void precond(const std::vector<Fac>& F, const std::vector<double>& r, std::vector<double>& z) {
    z.assign(r.size(), 0.0);
    for (const auto& f : F) {
        std::vector<double> b(f.n);
        for (int i = 0; i < f.n; ++i) b[i] = r[f.s + i];
        solve_fac(f, b.data(), 1, f.n);
        for (int i = 0; i < f.n; ++i) z[f.s + i] = b[i];
    }
}

int pcg(const SparseMatrixCSC& C0, const std::vector<Fac>& F, const std::vector<double>& b,
        double mu, double sc, double tol, int maxit, std::vector<double>& x) {
    int n = static_cast<int>(b.size());
    x.assign(n, 0.0);
    std::vector<double> r = b, z, p, Ap, tmp(C0.rows);
    double bn = std::max(nrm(b), 1.0);
    if (nrm(r) / bn <= tol) return 0;
    precond(F, r, z);
    p = z;
    double rz = dot(r, z);
    for (int it = 0; it < maxit; ++it) {
        applyA(C0, sc, mu, p, Ap, tmp);
        double den = dot(p, Ap);
        if (!std::isfinite(den) || std::abs(den) < std::numeric_limits<double>::min()) return 2;
        double a = rz / den;
        for (int i = 0; i < n; ++i) {
            x[i] += a * p[i];
            r[i] -= a * Ap[i];
        }
        if (nrm(r) / bn <= tol) return 0;
        precond(F, r, z);
        double rz2 = dot(r, z);
        double beta = rz2 / rz;
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
        rz = rz2;
    }
    return 1;
}

Matrix dense_normal(const SparseMatrixCSC& C0, double sc, double mu) {
    Matrix A(C0.cols, C0.cols);
    std::vector<std::vector<std::pair<int, double>>> rows(C0.rows);
    for (int c = 0; c < C0.cols; ++c)
        for (int p = C0.col_ptr[c]; p < C0.col_ptr[c + 1]; ++p)
            rows[C0.row_idx[p]].push_back({c, C0.values[p]});
    for (const auto& row : rows)
        for (const auto& a : row)
            for (const auto& b : row)
                A(a.first, b.first) += sc * a.second * b.second;
    for (int i = 0; i < C0.cols; ++i) A(i, i) += mu;
    return A;
}

Matrix dense_solve(const SparseMatrixCSC& C0, const Matrix& BB, double mu, double sc, bool chol) {
    return solve_dense(dense_normal(C0, sc, mu), BB, chol);
}

#if AXYB_ENABLE_EIGEN
using EigenDense = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

Matrix from_eigen(const Eigen::Ref<const EigenDense>& X) {
    Matrix M(static_cast<int>(X.rows()), static_cast<int>(X.cols()));
    std::copy(X.data(), X.data() + X.size(), M.a.begin());
    return M;
}

Eigen::SparseMatrix<double, Eigen::ColMajor, int> to_eigen_sparse(const SparseMatrixCSC& C0) {
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> C(C0.rows, C0.cols);
    C.reserve(C0.values.size());
    std::vector<Eigen::Triplet<double, int>> triplets;
    triplets.reserve(C0.values.size());
    for (int c = 0; c < C0.cols; ++c)
        for (int p = C0.col_ptr[c]; p < C0.col_ptr[c + 1]; ++p)
            triplets.emplace_back(C0.row_idx[p], c, C0.values[p]);
    C.setFromTriplets(triplets.begin(), triplets.end());
    C.makeCompressed();
    return C;
}

Matrix eigen_dense_solve(const SparseMatrixCSC& C0, const Matrix& BB, double mu, double sc, LinearBackend backend) {
    Matrix A0 = dense_normal(C0, sc, mu);
    Eigen::Map<const EigenDense> A(A0.a.data(), A0.rows, A0.cols);
    Eigen::Map<const EigenDense> B(BB.a.data(), BB.rows, BB.cols);
    EigenDense X;
    switch (backend) {
        case LinearBackend::EigenLLT: X = A.llt().solve(B); break;
        case LinearBackend::EigenLDLT: X = A.ldlt().solve(B); break;
        case LinearBackend::EigenPartialPivLU: X = A.partialPivLu().solve(B); break;
        default: throw std::runtime_error("invalid Eigen dense backend");
    }
    return from_eigen(X);
}

Matrix eigen_sparse_lu_solve(const SparseMatrixCSC& C0, const Matrix& BB, double mu, double sc) {
    auto C = to_eigen_sparse(C0);
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A = sc * (C.transpose() * C);
    A.makeCompressed();
    for (int i = 0; i < A.rows(); ++i) A.coeffRef(i, i) += mu;
    A.makeCompressed();
    Eigen::Map<const EigenDense> B(BB.a.data(), BB.rows, BB.cols);
    Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) throw std::runtime_error("Eigen SparseLU factorization failed");
    EigenDense X = solver.solve(B);
    if (solver.info() != Eigen::Success) throw std::runtime_error("Eigen SparseLU solve failed");
    return from_eigen(X);
}
#else
Matrix eigen_disabled() {
    throw std::runtime_error("Eigen backend requested, but Eigen3 was not found/enabled at configure time");
}
#endif

Matrix solve_pcg(const SparseMatrixCSC& C0, const Matrix& BB, double mu, double sc,
                 const TemplateData& tpl, const SolverOptions& opts) {
    auto F = factor(C0, tpl, sc, mu, opts.block_jitter);
    Matrix X(BB.rows, BB.cols);
    std::vector<int> flag(BB.cols);
    tbb::parallel_for(tbb::blocked_range<int>(0, BB.cols), [&](const tbb::blocked_range<int>& r) {
        for (int j = r.begin(); j != r.end(); ++j) {
            std::vector<double> b(BB.rows), x;
            for (int i = 0; i < BB.rows; ++i) b[i] = BB(i, j);
            flag[j] = pcg(C0, F, b, mu, sc, opts.pcg_tol, opts.pcg_maxit, x);
            for (int i = 0; i < BB.rows; ++i) X(i, j) = x[i];
        }
    });
    bool fail = false;
    for (int f : flag) fail = fail || (f != 0);
    if (fail) {
        if (!opts.fallback_direct) throw std::runtime_error("PCG did not converge");
        Matrix D = dense_solve(C0, BB, mu, sc, true);
        for (int j = 0; j < BB.cols; ++j)
            if (flag[j])
                for (int i = 0; i < BB.rows; ++i) X(i, j) = D(i, j);
    }
    return X;
}

Matrix solve_block(const SparseMatrixCSC& C0, const Matrix& BB, double mu, double sc,
                   const TemplateData& tpl, const SolverOptions& opts) {
    auto F = factor(C0, tpl, sc, mu, opts.block_jitter);
    auto blocksolve = [&](const Matrix& R) {
        Matrix Y(R.rows, R.cols);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, F.size()), [&](const tbb::blocked_range<size_t>& rg) {
            for (size_t bi = rg.begin(); bi != rg.end(); ++bi) {
                const auto& f = F[bi];
                std::vector<double> rhs(static_cast<size_t>(f.n) * R.cols);
                for (int j = 0; j < R.cols; ++j)
                    for (int i = 0; i < f.n; ++i)
                        rhs[static_cast<size_t>(j) * f.n + i] = R(f.s + i, j);
                solve_fac(f, rhs.data(), R.cols, f.n);
                for (int j = 0; j < R.cols; ++j)
                    for (int i = 0; i < f.n; ++i)
                        Y(f.s + i, j) = rhs[static_cast<size_t>(j) * f.n + i];
            }
        });
        return Y;
    };
    auto residual = [&](const Matrix& X) {
        Matrix R(BB.rows, BB.cols);
        tbb::parallel_for(tbb::blocked_range<int>(0, BB.cols), [&](const tbb::blocked_range<int>& rg) {
            for (int j = rg.begin(); j != rg.end(); ++j) {
                std::vector<double> x(BB.rows), y, tmp(C0.rows);
                for (int i = 0; i < BB.rows; ++i) x[i] = X(i, j);
                applyA(C0, sc, mu, x, y, tmp);
                for (int i = 0; i < BB.rows; ++i) R(i, j) = BB(i, j) - y[i];
            }
        });
        return R;
    };
    Matrix X = blocksolve(BB), R = residual(X), bestX = X;
    double bnorm = std::max(std::sqrt(frobenius_squared(BB)), 1.0);
    double best = std::sqrt(frobenius_squared(R)) / bnorm;
    for (int it = 0; it < opts.block_maxit && best > opts.block_tol; ++it) {
        Matrix D = blocksolve(R);
        for (size_t k = 0; k < X.a.size(); ++k) X.a[k] += opts.block_relaxation * D.a[k];
        R = residual(X);
        double rr = std::sqrt(frobenius_squared(R)) / bnorm;
        if (std::isfinite(rr) && rr < best) {
            best = rr;
            bestX = X;
        }
    }
    if (best <= opts.block_tol) return bestX;
    if (!opts.fallback_direct) throw std::runtime_error("block solver did not converge");
    return dense_solve(C0, BB, mu, sc, true);
}
} // namespace

LinearBackend parse_backend(const std::string& name) {
    auto s = lo(name);
    if (s == "pcg" || s == "tbb-pcg" || s == "parallel_pcg" || s == "parallel-pcg") return LinearBackend::TbbPcg;
    if (s == "backslash" || s == "block_backslash" || s == "block-backslash" || s == "block-jacobi" || s == "tbb-block-jacobi") return LinearBackend::TbbBlockJacobi;
    if (s == "matlab_backslash" || s == "matlab-backslash" || s == "global_backslash" || s == "direct" ||
        s == "dense-cholesky" || s == "lapack-cholesky" || s == "lapack-posv" || s == "accelerate" ||
        s == "blas-cholesky") return LinearBackend::LapackDenseCholesky;
    if (s == "dense-lu" || s == "lapack-lu" || s == "lapack-gesv" || s == "blas-lu") return LinearBackend::LapackDenseLU;
    if (s == "eigen-llt" || s == "eigen_llt" || s == "llt") return LinearBackend::EigenLLT;
    if (s == "eigen-ldlt" || s == "eigen_ldlt" || s == "ldlt") return LinearBackend::EigenLDLT;
    if (s == "eigen-lu" || s == "eigen_lu" || s == "eigen-partial-piv-lu" || s == "partial-piv-lu") return LinearBackend::EigenPartialPivLU;
    if (s == "eigen-sparse-lu" || s == "eigen_sparse_lu" || s == "sparse-lu") return LinearBackend::EigenSparseLU;
    throw std::runtime_error("unknown backend " + name);
}

std::string backend_help() {
    return "tbb-pcg, tbb-block-jacobi/backslash, lapack-posv/dense-cholesky/matlab_backslash, "
           "lapack-gesv/dense-lu, eigen-llt, eigen-ldlt, eigen-partial-piv-lu, eigen-sparse-lu";
}

Matrix compute_BB(const SparseMatrixCSC& C0, const SparseMatrixCSC& C1, double sc) {
    Matrix BB(C0.cols, C1.cols);
    tbb::parallel_for(tbb::blocked_range<int>(0, C1.cols), [&](const tbb::blocked_range<int>& r) {
        for (int col = r.begin(); col != r.end(); ++col) {
            std::vector<double> y(C1.rows), z;
            for (int p = C1.col_ptr[col]; p < C1.col_ptr[col + 1]; ++p) y[C1.row_idx[p]] = C1.values[p];
            C0.transpose_multiply(y, z);
            for (int i = 0; i < C0.cols; ++i) BB(i, col) = sc * z[i];
        }
    });
    return BB;
}

Matrix solve_template_system(const SparseMatrixCSC& C0, const Matrix& BB, double mu, double sc,
                             const TemplateData& tpl, const SolverOptions& opts) {
    switch (opts.backend) {
        case LinearBackend::TbbPcg: return solve_pcg(C0, BB, mu, sc, tpl, opts);
        case LinearBackend::TbbBlockJacobi: return solve_block(C0, BB, mu, sc, tpl, opts);
        case LinearBackend::LapackDenseCholesky: return dense_solve(C0, BB, mu, sc, true);
        case LinearBackend::LapackDenseLU: return dense_solve(C0, BB, mu, sc, false);
#if AXYB_ENABLE_EIGEN
        case LinearBackend::EigenLLT:
        case LinearBackend::EigenLDLT:
        case LinearBackend::EigenPartialPivLU:
            return eigen_dense_solve(C0, BB, mu, sc, opts.backend);
        case LinearBackend::EigenSparseLU:
            return eigen_sparse_lu_solve(C0, BB, mu, sc);
#else
        case LinearBackend::EigenLLT:
        case LinearBackend::EigenLDLT:
        case LinearBackend::EigenPartialPivLU:
        case LinearBackend::EigenSparseLU:
            return eigen_disabled();
#endif
    }
    throw std::runtime_error("unsupported backend");
}
} // namespace axyb
