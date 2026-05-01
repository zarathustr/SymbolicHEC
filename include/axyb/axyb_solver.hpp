#pragma once
#include "axyb/linear_solver.hpp"
#include "axyb/matrix.hpp"
#include <array>
#include <complex>
#include <random>
#include <string>
#include <vector>
namespace axyb{struct ProblemData{std::vector<Matrix>As,Bs;Matrix X0,Y0;std::array<double,6>g_ground_truth{};};struct SolveResult{Matrix X,Y;double objective=0;std::vector<std::array<std::complex<double>,6>>roots;std::string backend_name;int attempts=0;std::vector<DenseSolveDiagnostic>dense_solve_diagnostics;};ProblemData make_random_problem(int len,double noise,std::mt19937_64&rng);double J_AXYB(const std::vector<Matrix>&As,const std::vector<Matrix>&Bs,const Matrix&X,const Matrix&Y);double pose_error(const Matrix&X,const Matrix&X0);SolveResult AXYB_complete_grobner(const std::vector<Matrix>&As,const std::vector<Matrix>&Bs,double mu,const SolverOptions&opts);SolveResult AXYB_complete_grobner_with_retry(const std::vector<Matrix>&As,const std::vector<Matrix>&Bs,double mu,const SolverOptions&opts,double retry_tol);void print_dense_solve_diagnostics(const SolveResult&result);void print_matrix(const Matrix&M,const char*name);}
