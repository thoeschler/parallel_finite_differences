#pragma once
#include "grid.hpp"

#include <functional>
#include <vector>
#include <assert.h>


void compute_l1_error(double *error, std::vector<double> const& u_loc, UnitSquareGrid const& global_grid,
                      LocalUnitSquareGrid const& local_grid, int root, std::function<double(double, double)> u_ana,
                      MPI_Comm comm) {
    assert(u_loc.size() == local_grid.Nx * local_grid.Ny);
    double hx = 1.0 / (global_grid.Nx - 1), hy = 1.0 / (global_grid.Ny - 1);
    double e_loc = 0.0;
    double x, y;
    for (std::size_t idx_loc = 0; idx_loc < local_grid.Nx; ++idx_loc) {
        for (std::size_t idy_loc = 0; idy_loc < local_grid.Ny; ++idy_loc) {
            x = (idx_loc + local_grid.idx_glob_start + 1) * hx; // + 1 because of the boundary
            y = (idy_loc + local_grid.idy_glob_start + 1) * hy; // + 1 because of the boundary
            e_loc += std::abs(u_loc[idy_loc * local_grid.Nx + idx_loc] - u_ana(x, y)); 
        }
    }
    MPI_Reduce(&e_loc, error, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    *error /= (global_grid.Nx - 2) * (global_grid.Ny - 2);
}

void compute_linf_error(double *error, std::vector<double> const& u_loc, UnitSquareGrid const& global_grid,
                        LocalUnitSquareGrid const& local_grid, int root, std::function<double(double, double)> u_ana,
                        MPI_Comm comm) {
    assert(u_loc.size() == local_grid.Nx * local_grid.Ny);
    double hx = 1.0 / (global_grid.Nx - 1), hy = 1.0 / (global_grid.Ny - 1);
    double e, e_loc = 0.0;
    double x, y;
    for (std::size_t idx_loc = 0; idx_loc < local_grid.Nx; ++idx_loc) {
        for (std::size_t idy_loc = 0; idy_loc < local_grid.Ny; ++idy_loc) {
            x = (idx_loc + local_grid.idx_glob_start + 1) * hx;
            y = (idy_loc + local_grid.idy_glob_start + 1) * hy;
            e = std::abs(u_loc[idy_loc * local_grid.Nx + idx_loc] - u_ana(x, y));
            if (e > e_loc) e_loc = e;
        }
    }
    MPI_Reduce(&e_loc, error, 1, MPI_DOUBLE, MPI_MAX, root, comm);
}
