#pragma once
#include <vector>

struct UnitSquareGrid {
    std::size_t Nx;
    std::size_t Ny;
    UnitSquareGrid(std::size_t Nx_, std::size_t Ny_) : Nx{Nx_}, Ny{Ny_} {}; 
};

struct LocalUnitSquareGrid {
    std::size_t Nx;
    std::size_t Ny;
    std::size_t idx_glob_start;
    std::size_t idy_glob_start;
    bool has_lower_neighbor, has_upper_neighbor, has_left_neighbor, has_right_neighbor;
    LocalUnitSquareGrid(std::size_t Nx_, std::size_t Ny_, std::size_t idx_glob_start_, std::size_t idy_glob_start_,
                        std::vector<int> const& coords, std::vector<int> const& dims) {
                            Nx = Nx_;
                            Ny = Ny_;
                            idx_glob_start = idx_glob_start_;
                            idy_glob_start = idy_glob_start_;
                            has_upper_neighbor = coords[0] > 0;
                            has_lower_neighbor = coords[0] < dims[0] - 1;
                            has_left_neighbor = coords[1] > 0;
                            has_right_neighbor = coords[1] < dims[1] - 1;
                        }
};
