#pragma once
#include <vector>
namespace axyb {
std::vector<double> A_eq_tXtY_func(const std::vector<double>& in1);
std::vector<double> A_res_eq_tXtY_RY_func(const std::vector<double>& in1, const std::vector<double>& in2, const std::vector<double>& in3);
std::vector<double> b_eq_tXtY_func(const std::vector<double>& in1, const std::vector<double>& in2);
std::vector<double> W_J_trans_cayley_gY_func(const std::vector<double>& in1, const std::vector<double>& in2, const std::vector<double>& in3, const std::vector<double>& in4, const std::vector<double>& in5, const std::vector<double>& in6);
std::vector<double> W_J_rot_gXgY_func(const std::vector<double>& in1, const std::vector<double>& in2);
std::vector<double> W_AXYB_gXgY_func(const std::vector<double>& in1, const std::vector<double>& in2);
}
