#include "finite_diff.hpp"
#include "grid.hpp"

bool on_boundary(double x, double y) {
    constexpr double tol = 1e-7;
    return (std::abs(x) - 1) < tol || (std::abs(y) - 1.0) < tol;
}

void assemble_local_rhs(std::vector<double> &b_loc, UnitSquareGrid const& global_grid,
                        LocalUnitSquareGrid const& local_grid, std::vector<int> const& coords,
                        std::vector<int> const& dims, std::function<double(double, double)> bc) {
    b_loc.resize(local_grid.Nx * local_grid.Ny); // TODO: not every process needs this? for most just zeros
    const double hx = 1.0 / (global_grid.Nx - 1);
    const double hy = 1.0 / (global_grid.Ny - 1);
    const double hx2 = hx * hx;
    const double hy2 = hy * hy;
    double x_bndry, y_bndry;
    int px = coords[1], py = dims[0] - coords[0] - 1;

    if (px == 0) { // left side
        x_bndry = 0.0;
        for (std::size_t idy_loc = 0; idy_loc < local_grid.Ny; ++idy_loc) {
            y_bndry = (local_grid.idy_glob_start + idy_loc + 1) * hy;
            b_loc[idy_loc * local_grid.Nx] += bc(x_bndry, y_bndry) / hx2;
        }
    }
    else if (px == dims[1] - 1) { // right side
        x_bndry = 1.0;
        for (std::size_t idy_loc = 0; idy_loc < local_grid.Ny; ++idy_loc) {
            y_bndry = (local_grid.idy_glob_start + idy_loc + 1) * hy;
            b_loc[(idy_loc + 1) * local_grid.Nx - 1] += bc(x_bndry, y_bndry) / hx2;
        }
    }
    if (py == 0) { // lower side
        y_bndry = 0.0;
        for (std::size_t idx_loc = 0; idx_loc < local_grid.Nx; ++idx_loc) {
            x_bndry = (local_grid.idx_glob_start + idx_loc + 1) * hx;
            b_loc[idx_loc] += bc(x_bndry, y_bndry) / hy2;
        }
    }
    else if (py == dims[0] - 1) { // upper side
        y_bndry = 1.0;
        for (std::size_t idx_loc = 0; idx_loc < local_grid.Nx; ++idx_loc) {
            x_bndry = (local_grid.idx_glob_start + idx_loc + 1) * hx;
            b_loc[(local_grid.Ny - 1) * local_grid.Nx + idx_loc] += bc(x_bndry, y_bndry) / hy2;
        }
    }
}

void assemble_local_matrix(CRSMatrix &A, UnitSquareGrid const& global_grid, LocalUnitSquareGrid const& local_grid) {
    const double hx = 1.0 / (global_grid.Nx - 1);
    const double hy = 1.0 / (global_grid.Ny - 1);
    const double hx2 = hx * hx;
    const double hy2 = hy * hy;

    const double diagonal_value = 2.0 / hx2 + 2.0 / hy2;
    std::size_t node_loc;
    for (std::size_t idy_loc = 0; idy_loc < local_grid.Ny; ++idy_loc) {
        for (std::size_t idx_loc = 0; idx_loc < local_grid.Nx; ++idx_loc) {
            node_loc = idy_loc * local_grid.Nx + idx_loc;
            if (idy_loc > 0) { A.append(- 1.0 / hy2, node_loc - local_grid.Nx); }
            if (idx_loc > 0) { A.append(- 1.0 / hx2, node_loc - 1); }
            A.append(diagonal_value, node_loc);
            if (idx_loc < local_grid.Nx - 1) { A.append(- 1.0 / hx2, node_loc + 1); }
            if (idy_loc < local_grid.Ny - 1) { A.append(- 1.0 / hy2, node_loc + local_grid.Nx); }

            A.next_row();
        }
    }
}
