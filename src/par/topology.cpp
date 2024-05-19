#include "topology.hpp"

inline bool is_divisible(int number, int by) {
    return number % by == 0;
}

inline bool is_closer(double number, double than, double to) {
    return std::abs(number - to) < std::abs(to - than);
}

/**
 * @brief Initialize dimensions of cartesian topology.
 * 
 * @param dims Dimensions [out].
 * @param global_grid Global UnitSquareGrid.
 */
void initialize_cartesian_topology_dimensions(std::vector<int> &dims, UnitSquareGrid const& global_grid) {
    int nnodes;
    MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
    
    // first let MPI create dims
    std::fill(dims.begin(), dims.end(), 0);
    MPI_Dims_create(nnodes, 2, dims.data());

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
    for (int dimx = dims[1] - 1; dimx > 0 && dimx <= nnodes; dimx += direction) {
        if (is_divisible(nnodes, dimx)) {
            new_dims = {0, dimx};
            MPI_Dims_create(nnodes, 2, new_dims.data());
            new_proc_ratio = new_dims[1] / new_dims[0];
            if (is_closer(new_proc_ratio, proc_ratio, node_ratio)) {
                proc_ratio = new_proc_ratio;
                dims = new_dims;
            }
        }
    }
}
