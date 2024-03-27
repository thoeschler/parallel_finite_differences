#pragma once
#include "crs.hpp"
#include "grid.hpp"

#include <vector>
#include <cmath>
#include <mpi.h>

void parallel_cg(CRSMatrix const&A_loc, std::vector<double> const&b_loc, std::vector<double> &u_loc,
                 LocalUnitSquareGrid const& local_grid, MPI_Comm comm_cart, const double tol=1e-7);