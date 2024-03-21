#include "grid.hpp"
#include "crs.hpp"
#include "la.hpp"
#include "finite_diff.hpp"
#include "utils.hpp"

#include <iostream>
#include <vector>


double bc(double x, double y) {
    return x + y;
}

int main() {
    // number of nodes
    const std::size_t Nx = 128, Ny = 128;
    const std::size_t Nxt = Nx - 2, Nyt = Ny - 2;
    UnitSquareGrid grid(Nx, Ny);

    // assemble right hand side
    std::vector<double> b(Nxt * Nyt);
    assemble_rhs(b, grid, bc);

    // assemble matrix
    CRSMatrix A;
    assemble_matrix(A, grid);

    // solve system
    std::vector<double> u(Nxt * Nyt);
    cg(A, b, u);

    // compute error
    double error = compute_l1_error(u, bc, grid);
    std::cout << "l1 error: " << error << "\n";
    error = compute_linf_error(u, bc, grid);
    std::cout << "linf error: " << error << "\n";
    
    return 0;
}
