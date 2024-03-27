#pragma once
#include "crs.hpp"
#include "grid.hpp"

#include <vector>
#include <mpi.h>

void matvec(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &result);
double dot(std::vector<double> const& a, std::vector<double> const& b);
void parallel_cg(CRSMatrix const&A_loc, std::vector<double> const&b_loc, std::vector<double> &u_loc,
                 LocalUnitSquareGrid const& local_grid, MPI_Comm comm_cart, const double tol=1e-7);
