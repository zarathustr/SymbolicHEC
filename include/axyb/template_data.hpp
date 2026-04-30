#pragma once
#include "axyb/matrix.hpp"
#include <array>
#include <string>
#include <utility>
#include <vector>
namespace axyb{struct TemplateData{int version=0,action=0,n=0,m=0,tail_count=0;std::vector<unsigned int>coeff_map,c0_coeff,c1_coeff,am_ind;std::vector<unsigned long long>c0_linear,c1_linear;std::array<int,6>sol_sources{};std::vector<std::pair<int,int>>blocks;};TemplateData load_template(const std::string&dir,int action);std::vector<double> map_coefficients(const TemplateData&tpl,const std::vector<double>&data);SparseMatrixCSC build_sparse_from_template(int rows,int cols,const std::vector<unsigned long long>&lin,const std::vector<unsigned int>&ci,const std::vector<double>&coeffs);}
