#pragma once

#include "axyb/axyb_solver.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace axyb {

void write_problem_sequence_header(std::ostream& out, int problem_count);
void write_problem(std::ostream& out, int index, const ProblemData& problem);
void write_problem_sequence_footer(std::ostream& out);
std::vector<ProblemData> load_problem_sequence(const std::string& path);

} // namespace axyb
