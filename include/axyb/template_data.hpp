#pragma once
#include "axyb/matrix.hpp"
#include <array>
#include <string>
#include <utility>
#include <vector>
namespace axyb{struct SparseTemplatePattern{int rows=0,cols=0;std::vector<int>col_ptr,row_idx;std::vector<unsigned int>coeff_index;};struct TemplateData{int version=0,action=0,n=0,m=0,tail_count=0;std::vector<unsigned int>coeff_map,c0_coeff,c1_coeff,am_ind;std::vector<unsigned long long>c0_linear,c1_linear;std::array<int,6>sol_sources{};std::vector<std::pair<int,int>>blocks;SparseTemplatePattern c0_pattern,c1_pattern;};TemplateData load_template(const std::string&dir,int action);const TemplateData& load_template_cached(const std::string&dir,int action);std::vector<double> map_coefficients(const TemplateData&tpl,const std::vector<double>&data);SparseMatrixCSC build_c0_from_template(const TemplateData&tpl,const std::vector<double>&coeffs);SparseMatrixCSC build_c1_from_template(const TemplateData&tpl,const std::vector<double>&coeffs);SparseMatrixCSC build_sparse_from_template(int rows,int cols,const std::vector<unsigned long long>&lin,const std::vector<unsigned int>&ci,const std::vector<double>&coeffs);}
