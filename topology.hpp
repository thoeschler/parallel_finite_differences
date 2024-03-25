#include "grid.hpp"

#include <vector>
#include <mpi.h>
#include <iostream>
#include <tuple>

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> get_local_dimensions(
    UnitSquareGrid const& global_grid, std::vector<int> const& dims, std::vector<int> const& coords
    );
void initialize_cartesian_topology_dimensions(const int ndims, std::vector<int> &dims, UnitSquareGrid const& global_grid);
