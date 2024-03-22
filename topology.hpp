#include "grid.hpp"

#include <vector>
#include <mpi.h>
#include <iostream>

void initialize_cartesian_topology_dimensions(const int ndims, std::vector<int> &dims, UnitSquareGrid const& global_grid);
