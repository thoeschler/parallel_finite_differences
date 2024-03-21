#pragma once
#include <vector>

struct UnitSquareGrid {
    std::size_t Nx;
    std::size_t Ny;
    UnitSquareGrid(std::size_t Nx_, std::size_t Ny_) : Nx{Nx_}, Ny{Ny_} {}; 
};
