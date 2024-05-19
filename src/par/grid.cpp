#include "grid.hpp"

void UnitSquareGrid::info() const {
    std::cout << "LocalUnitSquareGrid with\n\n";
    std::cout << " Dofs: Nx = " << Nx << ", Ny = " << Ny << "\n";

    std::cout << "\n\n";
}

void LocalUnitSquareGrid::info() const {
    std::cout << "LocalUnitSquareGrid with\n\n";
    std::cout << " Local dofs: Nx = " << Nx << ", Ny = " << Ny << "\n";
    std::cout << " Global starting indices: idx = " << idx_glob_start << ", idy = " << idy_glob_start << "\n";
    std::cout << " Neighbors: top = " << has_top_neighbor << ", bottom = " << has_bottom_neighbor;
    std::cout << ", left = " << has_left_neighbor << ", right = " << has_right_neighbor;

    std::cout << "\n\n";
}

/**
 * @brief Get the processes local dimensions.
 * 
 * @param global_grid Global UnitSquareGrid.
 * @param dims Dimensions of cartesian topology.
 * @param coords Coordinates of the process.
 * @return std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
 * Local dimensions, global starting indices.
 */
std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> LocalUnitSquareGrid::get_local_dimensions(
    UnitSquareGrid const& global_grid, std::vector<int> const& dims, std::vector<int> const& coords) {
    std::size_t px = coords[1];
    std::size_t py = dims[0] - coords[0] - 1;

    std::size_t Nxt = global_grid.Nx - 2, Nyt = global_grid.Ny - 2;
    std::size_t Nx_loc = Nxt / dims[1]; 
    std::size_t Ny_loc = Nyt / dims[0];

    std::size_t idx_glob_start = Nx_loc * px;
    std::size_t idy_glob_start = Ny_loc * py;

    std::size_t rest_x = Nxt % dims[1];
    std::size_t rest_y = Nyt % dims[0];

    if (px < rest_x) {
        Nx_loc += 1;
        idx_glob_start += px;
    }
    else {
        idx_glob_start += rest_x;
    }

    if (py < rest_y) {
        Ny_loc += 1;
        idy_glob_start += py;
    }
    else {
        idy_glob_start += rest_y;
    }

    return std::make_tuple(Nx_loc, Ny_loc, idx_glob_start, idy_glob_start);
}
