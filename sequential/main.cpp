#include <iostream>
#include <vector>
#include "crs.hpp"
#include "la.hpp"
#include "finite_diff.hpp"

double bc(double x, double y) {
    return x + y;
}

int main(int argc, char** argv) {
    // number of nodes
    const std::size_t Nx = 12, Ny = 12;
    const std::size_t Nxt = Nx - 2, Nyt = Ny - 2;

    // assemble right hand side
    std::vector<double> b((Nx - 2) * (Ny - 2));

    // assemble rhs vector
    assemble_rhs(b, Nx, Ny, bc);
    for (double value: b) {
        std::cout << value << " ";
    }
    std::cout << "\n\n";

    // assemble matrix
    CRSMatrix A;
    assemble_matrix(A, Nx, Ny);

    // A.print_values();

    // solve system
    std::vector<double> u(Nxt * Nyt); // solution vector
    cg(A, b, u);

    // for (std::size_t i = 0; i < Nyt; ++i) {
    //     for (std::size_t j = 0; j < Nxt; ++j) {
    //         std::cout << u[i * Nxt + j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    return 0;
}
