# parallel_finite_differences

Solve 2D Laplace equation with Dirichlet boundary conditions in parallel using MPI and OpenMP.

### How to use
The main file is laplace.cpp.

Compile using `make build`.

Run using, say, `mpirun np=8 Nx=500 Ny=500` to run with 8 MPI processes on a 500 x 500 grid.
