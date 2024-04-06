CXX=mpicxx
CXXFLAGS=-Wall -O3 -march=native
LIBS=#-fopenmp

build: laplace.o la.o crs.o finite_diff.o topology.o grid.o cg.o
	$(CXX) $(CXXFLAGS) topology.o grid.o finite_diff.o la.o cg.o crs.o laplace.o -o laplace.out $(LIBS)

run:
	np=$1
	Nx=$2
	Ny=$3
	mpirun -np $(np) ./laplace.out $(Nx) $(Ny)

clean:
	rm *.o *.out

.cpp.o:
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LIBS)
