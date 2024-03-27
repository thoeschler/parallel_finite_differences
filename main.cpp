#include "grid.hpp"
#include "crs.hpp"
#include "cg.hpp"
#include "finite_diff.hpp"
#include "utils.hpp"
#include "topology.hpp"

#include <iostream>
#include <vector>
#include <mpi.h>
#include <tuple>

#define Nx 300
#define Ny 400

double bc(double x, double y) {
    return x + y;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size, root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // global grid
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

    // get local dimensions, create local grid
    std::size_t Nx_loc, Ny_loc, idx_glob_start, idy_glob_start;
    std::tie(Nx_loc, Ny_loc, idx_glob_start, idy_glob_start) = get_local_dimensions(global_grid, dims, coords);
    LocalUnitSquareGrid local_grid(Nx_loc, Ny_loc, idx_glob_start, idy_glob_start, coords, dims);

    // assemble right hand side locally
    std::vector<double> b_loc;
    assemble_local_rhs(b_loc, global_grid, local_grid, coords, dims, bc);

    // assemble matrix locally
    CRSMatrix A_loc;
    assemble_local_matrix(A_loc, dims, coords, global_grid, local_grid);

    // solve system
    std::vector<double> u_loc;
    const double tol = 1e-9;
    parallel_cg(A_loc, b_loc, u_loc, local_grid, comm_cart, tol, true);

    // compute error
    double l1_error = 0.0, linf_error = 0.0;
    compute_l1_error(&l1_error, u_loc, global_grid, local_grid, root, bc, comm_cart);
    compute_linf_error(&linf_error, u_loc, global_grid, local_grid, root, bc, comm_cart);

    if (rank == root) {
        std::cout << "\n";
        std::cout << "l1 error:\t" << l1_error << "\n";
        std::cout << "linf error:\t" << linf_error << "\n";
    }

    MPI_Finalize();

    return 0;
}
