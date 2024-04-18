#pragma once
#include "grid.hpp"

#include <functional>
#include <vector>
#include <assert.h>

double compute_l1_error(std::vector<double> const& u, std::function<double(double, double)> u_ana, UnitSquareGrid const& grid) {
    assert(u.size() == (grid.Nx - 2) * (grid.Ny - 2));
    std::size_t Nxt = grid.Nx - 2, Nyt = grid.Ny - 2;
    double hx = 1.0 / (grid.Nx - 1), hy = 1.0 / (grid.Ny - 1);
    double error = 0.0;
    double x, y;
    for (std::size_t idx = 0; idx < Nxt; ++idx) {
        for (std::size_t idy = 0; idy < Nyt; ++idy) {
            x = (idx + 1) * hx;
            y = (idy + 1) * hy;
            error += std::abs(u[idy * Nxt + idx] - u_ana(x, y)); 
        }
    }
    return error / (Nxt * Nyt);
}

double compute_linf_error(std::vector<double> const& u, std::function<double(double, double)> u_ana, UnitSquareGrid const& grid) {
    assert(u.size() == (grid.Nx - 2) * (grid.Ny - 2));
    std::size_t Nxt = grid.Nx - 2, Nyt = grid.Ny - 2;
    double hx = 1.0 / (grid.Nx - 1), hy = 1.0 / (grid.Ny - 1);
    double error = 0.0;
    double x, y, e;
    for (std::size_t idx = 0; idx < Nxt; ++idx) {
        for (std::size_t idy = 0; idy < Nyt; ++idy) {
            x = (idx + 1) * hx;
            y = (idy + 1) * hy;
            e = std::abs(u[idy * Nxt + idx] - u_ana(x, y));
            if (e > error) error = e;
        }
    }
    return error;
}