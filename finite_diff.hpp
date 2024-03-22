#pragma once
#include "la.hpp"
#include "crs.hpp"
#include "grid.hpp"

#include <vector>
#include <functional>
#include <mpi.h>

void initialize_rhs(std::vector<double> &b, UnitSquareGrid const& global_grid, std::vector<int> const& coords,
                    std::vector<int> const& dims);
void assemble_rhs(std::vector<double> &b, UnitSquareGrid const& grid, std::function<double(double, double)> bc);

void assemble_matrix(CRSMatrix &A, UnitSquareGrid const& grid);
