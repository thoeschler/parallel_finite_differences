#pragma once
#include "la.hpp"
#include "crs.hpp"
#include "grid.hpp"

#include <vector>
#include <functional>
#include <mpi.h>

void assemble_rhs(std::vector<double> &b, const UnitSquareGrid &grid, std::function<double(double, double)> bc);

void assemble_matrix(CRSMatrix &A, const UnitSquareGrid &grid);

void assemble_local_rhs(std::vector<double> &b_loc, UnitSquareGrid const& global_grid, 
    const LocalUnitSquareGrid &local_grid, const std::vector<int> &coords, const std::vector<int> &dims,
    std::function<double(double, double)> boundary_condition);

void assemble_local_matrix(CRSMatrix &A_loc, const std::vector<int> &dims, const std::vector<int> &coords,
    const UnitSquareGrid &global_grid, const LocalUnitSquareGrid &local_grid);
