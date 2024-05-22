# parallel_finite_differences

This code was written as part of a project for a parallel programming university course.

It solves the 2D Laplace equation with Dirichlet boundary conditions with finite differences (using the 5-point stencil) on a unit square using MPI and OpenMP.

### Build and run
* compile the sequential and parallel versions of the code using `make sequential` and `make parallel`, respectively
* run the sequential version using, say, `make run_seq Nx=500 Ny=500` to run on a 500 x 500 grid
* run the parallel version using, say, `make run_par np=8 Nx=500 Ny=500` to run on a 500 x 500 grid with 8 MPI processes

### Options (described here for parallel version)
* change Dirichlet boundary conditions by redefining `boundary_condition` in [laplace.cpp](https://github.com/thoeschler/parallel_finite_differences/blob/main/src/par/laplace.cpp)
* change analytical solution by redefining `analytical_solution` in [laplace.cpp](https://github.com/thoeschler/parallel_finite_differences/blob/main/src/par/laplace.cpp)
* different MPI communication types are possible: choose the type by setting `COMMUNICATION_TYPE` in [config.mk](https://github.com/thoeschler/parallel_finite_differences/blob/main/src/par/config.mk)
