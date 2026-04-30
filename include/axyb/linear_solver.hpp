#pragma once
#include "axyb/matrix.hpp"
#include "axyb/template_data.hpp"
#include <string>
namespace axyb{enum class LinearBackend{TbbPcg,TbbBlockJacobi,LapackDenseCholesky,LapackDenseLU,EigenLLT,EigenLDLT,EigenPartialPivLU,EigenSparseLU};struct SolverOptions{LinearBackend backend=LinearBackend::TbbPcg;std::string backend_name="tbb-pcg";double pcg_tol=1e-8;int pcg_maxit=300;double block_tol=1e-8;int block_maxit=50;double block_relaxation=1.0;double block_jitter=0;bool fallback_direct=true;bool parallel_templates=true;bool parallel_backend_retries=true;std::string template_dir="templates";int template_first=1;int template_last=6;int threads=0;};LinearBackend parse_backend(const std::string&name);std::string backend_help();Matrix compute_BB(const SparseMatrixCSC&C0,const SparseMatrixCSC&C1,double scale);Matrix solve_template_system(const SparseMatrixCSC&C0,const Matrix&BB,double mu,double normal_scale,const TemplateData&tpl,const SolverOptions&opts);}
