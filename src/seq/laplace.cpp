#include "grid.hpp"
#include "crs.hpp"
#include "cg.hpp"
#include "finite_diff.hpp"
#include "utils.hpp"

#include <iostream>
#include <vector>

double bc(double x, double y) {
    return x + y;
}

int main(int argc, char **argv) {
    std::size_t Nx = atoi(argv[1]);
    std::size_t Ny = atoi(argv[2]);
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
    std::cout << "\nl1 error: " << error << "\n";
    error = compute_linf_error(u, bc, grid);
    std::cout << "linf error: " << error << "\n";
    
    return 0;
}
