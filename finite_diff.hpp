#pragma once
#include "la.hpp"
#include "crs.hpp"
#include "grid.hpp"

#include <vector>
#include <functional>
#include <mpi.h>

void assemble_local_rhs(std::vector<double> &b_loc, UnitSquareGrid const& global_grid, 
                        LocalUnitSquareGrid const& local_grid, std::vector<int> const &coords,
                        std::vector<int> const& dims, std::function<double(double, double)> bc);

void assemble_local_matrix(CRSMatrix &A, UnitSquareGrid const& global_grid, LocalUnitSquareGrid const& local_grid);
