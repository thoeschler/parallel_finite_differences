#include "grid.hpp"
#include "crs.hpp"
#include "la.hpp"
#include "finite_diff.hpp"
#include "utils.hpp"

#include <iostream>
#include <vector>

#define Nx 1000
#define Ny 1000

double bc(double x, double y) {
    return x + y;
}

int main() {
    // number of nodes
    const std::size_t Nxt = Nx - 2, Nyt = Ny - 2;
    UnitSquareGrid grid(Nx, Ny);

    // assemble right hand side
    std::vector<double> b(Nxt * Nyt);
    assemble_rhs(b, grid, bc);

    // assemble matrix
    CRSMatrix A(5 * Nxt * Nyt, Nxt * Nyt);
    assemble_matrix(A, grid);

    // solve system
    std::vector<double> u(Nxt * Nyt);
    const double tol = 1e-9;
    cg(A, b, u, tol, true);

    // compute error
    double error = compute_l1_error(u, bc, grid);
    std::cout << "l1 error: " << error << "\n";
    error = compute_linf_error(u, bc, grid);
    std::cout << "linf error: " << error << "\n";
    
    return 0;
}
