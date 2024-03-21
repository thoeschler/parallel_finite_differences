#pragma once
#include "crs.hpp"

#include <vector>

void cg(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &u, const double tol=1e-7);
