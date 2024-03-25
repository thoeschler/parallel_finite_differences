#pragma once
#include "crs.hpp"

#include <vector>
#include <mpi.h>

void matmul(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &out);
double dot(std::vector<double> const& a, std::vector<double> const& b);
void add(std::vector<double> &inout, std::vector<double> const&in);
void cg(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &u, const double tol=1e-7);
