#include "finite_diff.hpp"
#include "grid.hpp"

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
    if (px == dims[1] - 1) { // right side
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
    if (py == dims[0] - 1) { // upper side
        y_bndry = 1.0;
        for (std::size_t idx_loc = 0; idx_loc < local_grid.Nx; ++idx_loc) {
            x_bndry = (local_grid.idx_glob_start + idx_loc + 1) * hx;
            b_loc[(local_grid.Ny - 1) * local_grid.Nx + idx_loc] += bc(x_bndry, y_bndry) / hy2;
        }
    }
}

void assemble_local_matrix(CRSMatrix &A_loc, std::vector<int> const& dims, std::vector<int> const& coords,
                           UnitSquareGrid const& global_grid, LocalUnitSquareGrid const& local_grid) {
    const double hx = 1.0 / (global_grid.Nx - 1);
    const double hy = 1.0 / (global_grid.Ny - 1);
    const double hx2 = hx * hx;
    const double hy2 = hy * hy;

    // local matrix size
    std::size_t Nxt = local_grid.Nx + 2;

    /*
    The local finite difference matrix will be of size
    (local_grid.Nx * local_grid.Ny) x 
    ((local_grid.Nx + 2) * (local_grid.Ny + 2)), i.e.
    every row corresponds to a node in the local grid.
    The number of columns also includes neighboring
    nodes that must be included in the 5 point stencil. 
    */

    // loop over all local nodes
    const double diagonal_value = 2.0 / hx2 + 2.0 / hy2;
    std::size_t col_self, col_left, col_right, col_down, col_up;

    for (std::size_t idy_loc = 0; idy_loc < local_grid.Ny; ++idy_loc) {
        for (std::size_t idx_loc = 0; idx_loc < local_grid.Nx; ++idx_loc) {
            col_self = (idy_loc + 1) * Nxt + idx_loc + 1;

            // lower neighbor
            col_down = col_self - Nxt;
            A_loc.append(- 1.0 / hy2, col_down);

            // upper neighbor
            col_up = col_self + Nxt;
            A_loc.append(- 1.0 / hy2, col_up);

            // self
            A_loc.append(diagonal_value, col_self);

            // left neighbor
            col_left = col_self - 1;
            A_loc.append(- 1.0 / hx2, col_left);

            // right neighbor
            col_right = col_self + 1;
            A_loc.append(- 1.0 / hx2, col_right);

            // move to next row
            A_loc.next_row();
        }
    }
}
