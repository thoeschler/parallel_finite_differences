#pragma once
#include "crs.hpp"

#include <vector>
#include <mpi.h>

void matvec(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &out);
double dot(std::vector<double> const& a, std::vector<double> const& b);
void parallel_cg(CRSMatrix const&A_loc, std::vector<double> const&b_loc, std::vector<double> &u_loc, MPI_Comm comm_cart,
                 const double tol=1e-7);
