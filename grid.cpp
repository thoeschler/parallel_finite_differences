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
