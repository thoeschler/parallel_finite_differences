#pragma once
#include "grid.hpp"

#include <vector>
#include <mpi.h>
#include <tuple>


void initialize_cartesian_topology_dimensions(std::vector<int> &dims, UnitSquareGrid const& global_grid);
