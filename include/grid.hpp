#pragma once
#include <vector>
#include <tuple>
#include <iostream>

struct UnitSquareGrid {
    std::size_t Nx, Ny; // number of grid points in x- and y-directions
    UnitSquareGrid(std::size_t Nx_, std::size_t Ny_) : Nx{Nx_}, Ny{Ny_} {};
    void info() const;
};

class LocalUnitSquareGrid {
    private:
        std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> get_local_dimensions(
            const UnitSquareGrid &global_grid, const std::vector<int> &dims, const std::vector<int> &coords
        );
    public:
        std::size_t Nx, Ny; // number of local grid points in x- and y-directions
        std::size_t idx_glob_start, idy_glob_start; // starting x- and y-indices in global grid
        bool has_bottom_neighbor, has_top_neighbor, has_left_neighbor, has_right_neighbor;
        LocalUnitSquareGrid(const UnitSquareGrid &global_grid, const std::vector<int> &dims, const std::vector<int> &coords) {
            std::tie(Nx, Ny, idx_glob_start, idy_glob_start) = get_local_dimensions(global_grid, dims, coords);
            has_top_neighbor = coords[0] > 0;
            has_bottom_neighbor = coords[0] < dims[0] - 1;
            has_left_neighbor = coords[1] > 0;
            has_right_neighbor = coords[1] < dims[1] - 1;
        }
        void info() const;
};
