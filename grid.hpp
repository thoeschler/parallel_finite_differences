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
    LocalUnitSquareGrid(std::size_t Nx_, std::size_t Ny_, std::size_t idx_glob_start_,
                        std::size_t idy_glob_start_) : Nx{Nx_}, Ny{Ny_},
                            idx_glob_start{idx_glob_start_}, idy_glob_start{idy_glob_start_} {}; 
};
