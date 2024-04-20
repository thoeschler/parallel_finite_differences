# parallel_finite_differences

This code was written as part of a project of a parallel programming university course.

It solves the 2D Laplace equation with Dirichlet boundary conditions with finite differences on a unit square using MPI and OpenMP.

### Build and run
* compile using `make build`.
* run using, say, `make run np=8 Nx=500 Ny=500` to run with 8 MPI processes on a 500 x 500 grid

### Options
* change Dirichlet boundary conditions by redefining `boundary_condition` in [laplace.cpp](https://github.com/thoeschler/parallel_finite_differences/blob/main/laplace.cpp)
* different MPI communication types are possible: choose the type by setting `COMMUNICATION_TYPE` in [config.mk](https://github.com/thoeschler/parallel_finite_differences/blob/main/config.mk)
