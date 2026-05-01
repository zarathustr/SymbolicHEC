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

Matrix dense_from_sparse(const SparseMatrixCSC& sparse) {
    Matrix dense(sparse.rows, sparse.cols);
    for (int col = 0; col < sparse.cols; ++col)
        for (int p = sparse.col_ptr[col]; p < sparse.col_ptr[col + 1]; ++p)
            dense(sparse.row_idx[p], col) = sparse.values[p];
    return dense;
}

struct SparseRowEntry {
    int col = 0;
    long double value = 0.0L;
};

using SparseRow = std::vector<SparseRowEntry>;

auto row_entry_lower_bound(SparseRow& row, int col) {
    return std::lower_bound(row.begin(), row.end(), col, [](const SparseRowEntry& entry, int target) {
        return entry.col < target;
    });
}

auto row_entry_lower_bound(const SparseRow& row, int col) {
    return std::lower_bound(row.begin(), row.end(), col, [](const SparseRowEntry& entry, int target) {
        return entry.col < target;
    });
}

long double row_value(const SparseRow& row, int col) {
    auto it = row_entry_lower_bound(row, col);
    if (it == row.end() || it->col != col) return 0.0L;
    return it->value;
}

void normalize_scientific(ScientificDeterminant& det) {
    if (det.sign == 0 || det.mantissa == 0.0L) {
        det.sign = 0;
        det.mantissa = 0.0L;
        det.exponent10 = 0;
        return;
    }
    while (det.mantissa >= 10.0L) {
        det.mantissa /= 10.0L;
        ++det.exponent10;
    }
    while (det.mantissa < 1.0L) {
        det.mantissa *= 10.0L;
        --det.exponent10;
    }
}

void accumulate_factor(ScientificDeterminant& det, long double factor) {
    if (det.sign == 0 || factor == 0.0L) {
        det.sign = 0;
        det.mantissa = 0.0L;
        det.exponent10 = 0;
        return;
    }
    if (factor < 0.0L) {
        det.sign = -det.sign;
        factor = -factor;
    }
    const long double exp10 = std::floor(std::log10(factor));
    const long long exp10_i = static_cast<long long>(exp10);
    det.mantissa *= factor / std::powl(10.0L, exp10);
    det.exponent10 += exp10_i;
    normalize_scientific(det);
}

std::vector<SparseRow> build_sparse_rows(const SparseMatrixCSC& C0) {
    std::vector<SparseRow> rows(static_cast<size_t>(C0.rows));
    for (int col = 0; col < C0.cols; ++col)
        for (int p = C0.col_ptr[col]; p < C0.col_ptr[col + 1]; ++p)
            rows[static_cast<size_t>(C0.row_idx[p])].push_back({col, static_cast<long double>(C0.values[p])});
    return rows;
}

int find_pivot_row(const std::vector<SparseRow>& rows, int start_col) {
    int best_row = -1;
    long double best_abs = 0.0L;
    for (int row = start_col; row < static_cast<int>(rows.size()); ++row) {
        const long double value = row_value(rows[static_cast<size_t>(row)], start_col);
        const long double abs_value = std::abs(value);
        if (abs_value > best_abs) {
            best_abs = abs_value;
            best_row = row;
        }
    }
    return best_row;
}

void eliminate_row(SparseRow& target, const SparseRow& pivot_row, int pivot_col, long double factor) {
    SparseRow updated;
    updated.reserve(target.size() + pivot_row.size());
    auto target_it = row_entry_lower_bound(target, pivot_col + 1);
    auto pivot_it = row_entry_lower_bound(pivot_row, pivot_col + 1);
    while (target_it != target.end() && pivot_it != pivot_row.end()) {
        if (target_it->col < pivot_it->col) {
            updated.push_back(*target_it);
            ++target_it;
        } else if (pivot_it->col < target_it->col) {
            updated.push_back({pivot_it->col, -factor * pivot_it->value});
            ++pivot_it;
        } else {
            const long double value = target_it->value - factor * pivot_it->value;
            if (value != 0.0L) updated.push_back({target_it->col, value});
            ++target_it;
            ++pivot_it;
        }
    }
    while (target_it != target.end()) {
        updated.push_back(*target_it);
        ++target_it;
    }
    while (pivot_it != pivot_row.end()) {
        updated.push_back({pivot_it->col, -factor * pivot_it->value});
        ++pivot_it;
    }
    target.swap(updated);
}

ScientificDeterminant determinant_sparse_classic(const SparseMatrixCSC& C0) {
    ScientificDeterminant det;
    det.computed = true;
    if (C0.rows != C0.cols) return det;
    det.sign = 1;
    det.mantissa = 1.0L;
    auto rows = build_sparse_rows(C0);
    for (int col = 0; col < C0.cols; ++col) {
        const int pivot_row = find_pivot_row(rows, col);
        if (pivot_row < 0) {
            det.sign = 0;
            det.mantissa = 0.0L;
            det.exponent10 = 0;
            return det;
        }
        if (pivot_row != col) {
            std::swap(rows[static_cast<size_t>(col)], rows[static_cast<size_t>(pivot_row)]);
            det.sign = -det.sign;
        }
        const long double pivot = row_value(rows[static_cast<size_t>(col)], col);
        if (pivot == 0.0L) {
            det.sign = 0;
            det.mantissa = 0.0L;
            det.exponent10 = 0;
            return det;
        }
        accumulate_factor(det, pivot);
        const SparseRow& pivot_entries = rows[static_cast<size_t>(col)];
        for (int row = col + 1; row < C0.rows; ++row) {
            SparseRow& target = rows[static_cast<size_t>(row)];
            const long double value = row_value(target, col);
            if (value == 0.0L) continue;
            eliminate_row(target, pivot_entries, col, value / pivot);
        }
    }
    return det;
}

void record_dense_diagnostic(const SparseMatrixCSC& C0, DenseSolveDiagnostic* diag) {
    if (!diag) return;
    diag->used_dense_solve = true;
    diag->c0_determinant = determinant_sparse_classic(C0);
}

void require_dense_system_shape(const Matrix& A, const Matrix& B, const char* name) {
    if (A.rows != A.cols || A.rows != B.rows) throw std::runtime_error(std::string(name) + " solve mismatch");
}

Matrix lapack_llt_solve(Matrix A, Matrix B) {
    require_dense_system_shape(A, B, "dpotrf/dpotrs");
    const int n = A.rows;
    const int nrhs = B.cols;
    int info = 0;
    char u = 'L';
    dpotrf_(&u, &n, A.a.data(), &n, &info);
    if (info) throw std::runtime_error("dpotrf failed");
    dpotrs_(&u, &n, &nrhs, A.a.data(), &n, B.a.data(), &n, &info);
    if (info) throw std::runtime_error("dpotrs failed");
    return B;
}

Matrix lapack_partial_piv_lu_solve(Matrix A, Matrix B) {
    require_dense_system_shape(A, B, "dgetrf/dgetrs");
    const int n = A.rows;
    const int nrhs = B.cols;
    int info = 0;
    std::vector<int> ipiv(n);
    dgetrf_(&n, &n, A.a.data(), &n, ipiv.data(), &info);
    if (info) throw std::runtime_error("dgetrf failed");
    char tr = 'N';
    dgetrs_(&tr, &n, &nrhs, A.a.data(), &n, ipiv.data(), B.a.data(), &n, &info);
    if (info) throw std::runtime_error("dgetrs failed");
    return B;
}

Matrix lapack_ldlt_solve(Matrix A, Matrix B) {
    require_dense_system_shape(A, B, "dsysv");
    const int n = A.rows;
    const int nrhs = B.cols;
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    std::vector<int> ipiv(n);
    char u = 'L';
    dsysv_(&u, &n, &nrhs, A.a.data(), &n, ipiv.data(), B.a.data(), &n, &work_query, &lwork, &info);
    if (info) throw std::runtime_error("dsysv workspace query failed");
    lwork = std::max(1, static_cast<int>(std::ceil(work_query)));
    std::vector<double> work(static_cast<size_t>(lwork));
    dsysv_(&u, &n, &nrhs, A.a.data(), &n, ipiv.data(), B.a.data(), &n, work.data(), &lwork, &info);
    if (info) throw std::runtime_error("dsysv failed");
    return B;
}

struct DenseSVD {
    int rows = 0;
    int cols = 0;
    std::vector<double> sigma;
    Matrix U;
    Matrix VT;
    double tol = 0.0;
};

DenseSVD lapack_full_svd(Matrix A) {
    DenseSVD svd;
    svd.rows = A.rows;
    svd.cols = A.cols;
    const int m = A.rows;
    const int n = A.cols;
    const int k = std::min(m, n);
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    char job = 'A';
    svd.sigma.resize(static_cast<size_t>(k));
    svd.U = Matrix(m, m);
    svd.VT = Matrix(n, n);
    dgesvd_(&job, &job, &m, &n, A.a.data(), &m, svd.sigma.data(), svd.U.a.data(), &m, svd.VT.a.data(), &n,
            &work_query, &lwork, &info);
    if (info) throw std::runtime_error("dgesvd workspace query failed");
    lwork = std::max(1, static_cast<int>(std::ceil(work_query)));
    std::vector<double> work(static_cast<size_t>(lwork));
    dgesvd_(&job, &job, &m, &n, A.a.data(), &m, svd.sigma.data(), svd.U.a.data(), &m, svd.VT.a.data(), &n,
            work.data(), &lwork, &info);
    if (info) throw std::runtime_error("dgesvd failed");
    if (!svd.sigma.empty())
        svd.tol = std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(m, n)) * svd.sigma.front();
    return svd;
}

Matrix lapack_svd_solve_from_factors(const DenseSVD& svd, const Matrix& B) {
    if (B.rows != svd.rows) throw std::runtime_error("dgesvd solve mismatch");
    Matrix X(svd.cols, B.cols);
    const int k = std::min(svd.rows, svd.cols);
    for (int j = 0; j < B.cols; ++j) {
        for (int i = 0; i < k; ++i) {
            const double sigma = svd.sigma[static_cast<size_t>(i)];
            if (!(sigma > svd.tol)) continue;
            double coeff = 0.0;
            for (int r = 0; r < svd.rows; ++r) coeff += svd.U(r, i) * B(r, j);
            coeff /= sigma;
            for (int r = 0; r < svd.cols; ++r) X(r, j) += svd.VT(i, r) * coeff;
        }
    }
    return X;
}

Matrix lapack_pseudo_inverse_from_factors(const DenseSVD& svd) {
    Matrix pinv(svd.cols, svd.rows);
    const int k = std::min(svd.rows, svd.cols);
    for (int i = 0; i < k; ++i) {
        const double sigma = svd.sigma[static_cast<size_t>(i)];
        if (!(sigma > svd.tol)) continue;
        const double alpha = 1.0 / sigma;
        for (int c = 0; c < svd.rows; ++c) {
            const double uc = svd.U(c, i);
            if (uc == 0.0) continue;
            for (int r = 0; r < svd.cols; ++r) {
                pinv(r, c) += alpha * svd.VT(i, r) * uc;
            }
        }
    }
    return pinv;
}

Matrix lapack_svd_solve(Matrix A, const Matrix& B) {
    return lapack_svd_solve_from_factors(lapack_full_svd(std::move(A)), B);
}

Matrix lapack_pinv_solve(Matrix A, const Matrix& B) {
    return matmul(lapack_pseudo_inverse_from_factors(lapack_full_svd(std::move(A))), B);
}

Matrix solve_dense_backend_matrix(Matrix A, const Matrix& B, LinearBackend backend) {
    switch (backend) {
        case LinearBackend::LapackDenseCholesky: return solve_dense(std::move(A), B, true);
        case LinearBackend::LapackDenseLU: return solve_dense(std::move(A), B, false);
        case LinearBackend::LapackLLT: return lapack_llt_solve(std::move(A), B);
        case LinearBackend::LapackPartialPivLU: return lapack_partial_piv_lu_solve(std::move(A), B);
        case LinearBackend::LapackLDLT: return lapack_ldlt_solve(std::move(A), B);
        case LinearBackend::LapackPseudoInverse: return lapack_pinv_solve(std::move(A), B);
        case LinearBackend::LapackSVD: return lapack_svd_solve(std::move(A), B);
        default: throw std::runtime_error("invalid LAPACK dense backend");
    }
}

Matrix dense_backend_solve(const SparseMatrixCSC& C0, const Matrix& rhs, double mu, double sc, LinearBackend backend,
                           bool asymmetric, DenseSolveDiagnostic* diag = nullptr) {
    record_dense_diagnostic(C0, diag);
    Matrix A = asymmetric ? dense_from_sparse(C0) : dense_normal(C0, sc, mu);
    return solve_dense_backend_matrix(std::move(A), rhs, backend);
}

Matrix dense_solve(const SparseMatrixCSC& C0, const Matrix& BB, double mu, double sc, bool chol,
                   DenseSolveDiagnostic* diag = nullptr) {
    return dense_backend_solve(C0, BB, mu, sc,
                               chol ? LinearBackend::LapackDenseCholesky : LinearBackend::LapackDenseLU,
                               false, diag);
}

Matrix dense_asymmetric_solve(const SparseMatrixCSC& C0, const Matrix& C1t, DenseSolveDiagnostic* diag = nullptr) {
    return dense_backend_solve(C0, C1t, 0.0, 1.0, LinearBackend::LapackDenseLU, true, diag);
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

Matrix eigen_dense_asymmetric_solve(const SparseMatrixCSC& C0, const Matrix& C1t, LinearBackend backend) {
    Matrix A0 = dense_from_sparse(C0);
    Eigen::Map<const EigenDense> A(A0.a.data(), A0.rows, A0.cols);
    Eigen::Map<const EigenDense> B(C1t.a.data(), C1t.rows, C1t.cols);
    EigenDense X;
    switch (backend) {
        case LinearBackend::EigenPartialPivLU: X = A.partialPivLu().solve(B); break;
        default: throw std::runtime_error("invalid asymmetric Eigen dense backend");
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

Matrix eigen_sparse_asymmetric_lu_solve(const SparseMatrixCSC& C0, const Matrix& C1t) {
    auto A = to_eigen_sparse(C0);
    Eigen::Map<const EigenDense> B(C1t.a.data(), C1t.rows, C1t.cols);
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
                 const TemplateData& tpl, const SolverOptions& opts, DenseSolveDiagnostic* diag) {
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
        Matrix D = dense_solve(C0, BB, mu, sc, true, diag);
        for (int j = 0; j < BB.cols; ++j)
            if (flag[j])
                for (int i = 0; i < BB.rows; ++i) X(i, j) = D(i, j);
    }
    return X;
}

Matrix solve_block(const SparseMatrixCSC& C0, const Matrix& BB, double mu, double sc,
                   const TemplateData& tpl, const SolverOptions& opts, DenseSolveDiagnostic* diag) {
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
    return dense_solve(C0, BB, mu, sc, true, diag);
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
    if (s == "lapack-llt" || s == "lapack-potrf" || s == "lapack-potrs") return LinearBackend::LapackLLT;
    if (s == "lapack-partial-piv-lu" || s == "lapack-partial_piv_lu" || s == "lapack-getrf" || s == "lapack-getrs")
        return LinearBackend::LapackPartialPivLU;
    if (s == "lapack-ldlt" || s == "lapack-sysv") return LinearBackend::LapackLDLT;
    if (s == "lapack-pinv" || s == "lapack-pseudoinverse") return LinearBackend::LapackPseudoInverse;
    if (s == "lapack-svd") return LinearBackend::LapackSVD;
    if (s == "eigen-llt" || s == "eigen_llt" || s == "llt") return LinearBackend::EigenLLT;
    if (s == "eigen-ldlt" || s == "eigen_ldlt" || s == "ldlt") return LinearBackend::EigenLDLT;
    if (s == "eigen-lu" || s == "eigen_lu" || s == "eigen-partial-piv-lu" || s == "partial-piv-lu") return LinearBackend::EigenPartialPivLU;
    if (s == "eigen-sparse-lu" || s == "eigen_sparse_lu" || s == "sparse-lu") return LinearBackend::EigenSparseLU;
    throw std::runtime_error("unknown backend " + name);
}

std::string backend_help() {
    return "tbb-pcg, tbb-block-jacobi/backslash, lapack-posv/dense-cholesky/matlab_backslash, "
           "lapack-gesv/dense-lu, lapack-llt, lapack-partial-piv-lu, lapack-ldlt, lapack-pinv, lapack-svd, "
           "eigen-llt, eigen-ldlt, eigen-partial-piv-lu, eigen-sparse-lu";
}

bool backend_supports_asymmetric(LinearBackend backend) {
    switch (backend) {
        case LinearBackend::LapackDenseLU:
        case LinearBackend::LapackPartialPivLU:
        case LinearBackend::LapackPseudoInverse:
        case LinearBackend::LapackSVD:
        case LinearBackend::EigenPartialPivLU:
        case LinearBackend::EigenSparseLU:
            return true;
        default:
            return false;
    }
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

Matrix solve_template_system(const SparseMatrixCSC& C0, const Matrix& rhs, double mu, double sc,
                             const TemplateData& tpl, const SolverOptions& opts, DenseSolveDiagnostic* diag) {
    if (diag) {
        diag->used_dense_solve = false;
        diag->c0_determinant = {};
    }
    if (opts.asymmetric) {
        switch (opts.backend) {
            case LinearBackend::LapackDenseLU:
                return dense_backend_solve(C0, rhs, mu, sc, opts.backend, true, diag);
            case LinearBackend::LapackPartialPivLU:
            case LinearBackend::LapackPseudoInverse:
            case LinearBackend::LapackSVD:
                return dense_backend_solve(C0, rhs, mu, sc, opts.backend, true, diag);
#if AXYB_ENABLE_EIGEN
            case LinearBackend::EigenPartialPivLU:
                return eigen_dense_asymmetric_solve(C0, rhs, opts.backend);
            case LinearBackend::EigenSparseLU:
                return eigen_sparse_asymmetric_lu_solve(C0, rhs);
#else
            case LinearBackend::EigenPartialPivLU:
            case LinearBackend::EigenSparseLU:
                return eigen_disabled();
#endif
            default:
                throw std::runtime_error("--asymmetric requires lapack-gesv/dense-lu, lapack-partial-piv-lu, lapack-pinv, lapack-svd, eigen-partial-piv-lu, or eigen-sparse-lu");
        }
    }
    switch (opts.backend) {
        case LinearBackend::TbbPcg: return solve_pcg(C0, rhs, mu, sc, tpl, opts, diag);
        case LinearBackend::TbbBlockJacobi: return solve_block(C0, rhs, mu, sc, tpl, opts, diag);
        case LinearBackend::LapackDenseCholesky: return dense_solve(C0, rhs, mu, sc, true, diag);
        case LinearBackend::LapackDenseLU:
        case LinearBackend::LapackLLT:
        case LinearBackend::LapackPartialPivLU:
        case LinearBackend::LapackLDLT:
        case LinearBackend::LapackPseudoInverse:
        case LinearBackend::LapackSVD:
            return dense_backend_solve(C0, rhs, mu, sc, opts.backend, false, diag);
#if AXYB_ENABLE_EIGEN
        case LinearBackend::EigenLLT:
        case LinearBackend::EigenLDLT:
        case LinearBackend::EigenPartialPivLU:
            return eigen_dense_solve(C0, rhs, mu, sc, opts.backend);
        case LinearBackend::EigenSparseLU:
            return eigen_sparse_lu_solve(C0, rhs, mu, sc);
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
