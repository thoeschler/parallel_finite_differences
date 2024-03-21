#include "finite_diff.hpp"
#include "grid.hpp"

#include <iostream>

bool on_boundary(double x, double y) {
    constexpr double tol = 1e-7;
    return (std::abs(x) - 1) < tol || (std::abs(y) - 1.0) < tol;
}

void assemble_rhs(std::vector<double> &b, UnitSquareGrid const& grid, std::function<double(double, double)> bc) {
    std::fill(b.begin(), b.end(), 0.0);
    std::size_t Nxt = grid.Nx - 2;
    std::size_t Nyt = grid.Ny - 2;

    const double hx = 1.0 / (grid.Nx - 1);
    const double hy = 1.0 / (grid.Ny - 1);
    const double hx2 = hx * hx;
    const double hy2 = hy * hy;
    double x_bndry, y_bndry;

    // lower part
    y_bndry = 0.0;
    for (std::size_t idx = 0; idx < Nxt; ++idx) {
        x_bndry = (idx + 1) * hx;
        b[idx] += bc(x_bndry, y_bndry) / hy2;
    }

    // right part
    x_bndry = 1.0;
    for (std::size_t idy = 0; idy < Nyt; ++idy) {
        y_bndry = (idy + 1) * hy;
        b[(idy + 1) * Nxt - 1] += bc(x_bndry, y_bndry) / hx2;
    }

    // upper part
    y_bndry = 1.0;
    for (std::size_t idx = 0; idx < Nxt; ++idx) {
        x_bndry = (idx + 1) * hx;
        b[(Nyt - 1) * Nxt + idx] += bc(x_bndry, y_bndry) / hy2;
    }

    // left part
    x_bndry = 0.0;
    for (std::size_t idy = 0; idy < Nyt; ++idy) {
        y_bndry = (idy + 1) * hy;
        b[idy * Nxt] += bc(x_bndry, y_bndry) / hx2;
    }
}

void assemble_matrix(CRSMatrix &A, UnitSquareGrid const& grid) {
    const double hx = 1.0 / (grid.Nx - 1);
    const double hy = 1.0 / (grid.Ny - 1);
    std::size_t Nxt = grid.Nx - 2;
    std::size_t Nyt = grid.Ny - 2;
    const double hx2 = hx * hx;
    const double hy2 = hy * hy;

    const double diagonal_value = - 2.0 / hx2 - 2.0 / hy2;
    std::size_t node_number;
    for (std::size_t idx = 0; idx < Nxt; ++idx) {
        for (std::size_t idy = 0; idy < Nyt; ++idy) {
            node_number = idx + Nxt * idy;
            if (idy > 0) { A.append(1.0 / hy2, node_number - Nxt); }
            if (idx > 0) { A.append(1.0 / hx2, node_number - 1); }
            A.append(diagonal_value, node_number);
            if (idx < Nxt - 1) { A.append(1.0 / hx2, node_number + 1); }
            if (idy < Nyt - 1) { A.append(1.0 / hy2, node_number + Nxt); }

            A.next_row();
        }
    }
}
