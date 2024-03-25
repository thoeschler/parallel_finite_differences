#include "topology.hpp"

inline bool is_divisible(int number, int by) {
    return number % by == 0;
}

inline bool is_closer(double number, double than, double to) {
    return std::abs(number - to) < std::abs(to - than);
}

void initialize_cartesian_topology_dimensions(const int ndims, std::vector<int> &dims, UnitSquareGrid const& global_grid) {
    int nnodes, size;
    MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // first let MPI create dims
    std::fill(dims.begin(), dims.end(), 0);
    MPI_Dims_create(nnodes, ndims, dims.data());

    // swap dims if they do not match the grid size
    if ((global_grid.Nx > global_grid.Ny && dims[1] < dims[0])
        || ((global_grid.Ny > global_grid.Nx && dims[0] < dims[1]))) {
        int tmp = dims[0];
        dims[0] = dims[1];
        dims[1] = tmp;
    }

    // check if better decomposition is possible
    const double node_ratio = double(global_grid.Nx) / global_grid.Ny;
    double proc_ratio = double(dims[1]) / dims[0];

    // try decreasing (-1) or increasing (1) number of processes in x direction
    int direction = proc_ratio > node_ratio ? -1: 1;

    double new_proc_ratio;
    std::vector<int> new_dims(2);
    for (int dimx = dims[1] - 1; dimx > 0 && dimx <= size; dimx += direction) {
        if (is_divisible(nnodes, dimx)) {
            new_dims = {0, dimx};
            MPI_Dims_create(nnodes, ndims, new_dims.data());
            new_proc_ratio = new_dims[1] / new_dims[0];
            if (is_closer(new_proc_ratio, proc_ratio, node_ratio)) {
                proc_ratio = new_proc_ratio;
                dims = new_dims;
            }
        }
    }
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> get_local_dimensions(
    UnitSquareGrid const& global_grid, std::vector<int> const& dims, std::vector<int> const& coords) {
    std::size_t px = coords[1];
    std::size_t py = dims[0] - coords[0] - 1;

    std::size_t Nxt = global_grid.Nx - 2, Nyt = global_grid.Ny - 2;
    std::size_t Nx_loc = Nxt / dims[1]; 
    std::size_t Ny_loc = Nyt / dims[0];

    std::size_t idx_glob_start = Nx_loc * px;
    std::size_t idy_glob_start = Ny_loc * py;

    std::size_t rest_x = Nxt % dims[1];
    std::size_t rest_y = Nyt % dims[0];

    if (px < rest_x) Nx_loc += 1;
    if (py < rest_y) Ny_loc += 1;
    idx_glob_start += px;
    idy_glob_start += py;

    return std::make_tuple(Nx_loc, Ny_loc, idx_glob_start, idy_glob_start);
}
