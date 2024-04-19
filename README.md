# parallel_finite_differences

Solve 2D Laplace equation with Dirichlet boundary conditions with finite differences on unit square using MPI and OpenMP.

### Build and run
* compile using `make build`.
* run using, say, `make run np=8 Nx=500 Ny=500` to run with 8 MPI processes on a 500 x 500 grid

### Options
* change dirichlet boundary conditions by overwriting `boundary_condition` function in [laplace.cpp](https://github.com/thoeschler/parallel_finite_differences/blob/main/laplace.cpp)
* choose MPI communication type by setting `COMMUNICATION_TYPE` in [config.mk](https://github.com/thoeschler/parallel_finite_differences/blob/main/config.mk)
