CXX=mpicxx
CXXFLAGS=-Wall -O3 -march=native

build: laplace.o la.o crs.o finite_diff.o topology.o grid.o cg.o
	$(CXX) $(CXXFLAGS) topology.o grid.o finite_diff.o la.o cg.o crs.o laplace.o -o laplace.out

run:
	np=$1;
	mpirun -np $(np) ./laplace.out

clean:
	rm *.o *.out

.cpp.o:
	$(CXX) $(CXXFLAGS) -o $@ -c $<
