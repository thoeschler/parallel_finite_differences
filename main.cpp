#include "grid.hpp"
#include "crs.hpp"
#include "la.hpp"
#include "finite_diff.hpp"
#include "utils.hpp"
#include "topology.hpp"

#include <iostream>
#include <vector>
#include <mpi.h>

#define Nx 1000
#define Ny 400

double bc(double x, double y) {
    return x + y;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // global grid
    // const std::size_t Nxt = Nx - 2, Nyt = Ny - 2;
    UnitSquareGrid global_grid(Nx, Ny);

    // initialize cartesian topology
    const int ndims = 2;
    std::vector<int> dims(ndims), periods(ndims, 0), coords(ndims);
    initialize_cartesian_topology_dimensions(ndims, dims, global_grid);
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims.data(), periods.data(), 1, &comm_cart);
    int cart_rank;
    MPI_Comm_rank(comm_cart, &cart_rank);
    MPI_Cart_coords(comm_cart, rank, ndims, coords.data());

    // assemble right hand side
    std::vector<double> b;
    initialize_rhs(b, global_grid, coords, dims);
    // assemble_rhs(b, grid, bc);

    std::cout << rank << " " << cart_rank << " " << coords[1] << " " << coords[0] << " " << b.size() << "\n";
    
    // assemble_rhs(b, grid, bc);

    // // assemble matrix
    // CRSMatrix A(5 * Nxt * Nyt, Nxt * Nyt);
    // assemble_matrix(A, grid);

    // // solve system
    // std::vector<double> u(Nxt * Nyt);
    // cg(A, b, u);

    // // compute error
    // double error = compute_l1_error(u, bc, grid);
    // std::cout << "l1 error: " << error << "\n";
    // error = compute_linf_error(u, bc, grid);
    // std::cout << "linf error: " << error << "\n";

    MPI_Finalize();

    return 0;
}
