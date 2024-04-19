include config.mk

CXXFLAGS=-Wall -O3 -march=native -Dcommunication_type=$(COMMUNICATION_TYPE)
LIBS=#-fopenmp

ALL = laplace.o la.o crs.o finite_diff.o topology.o grid.o cg.o

build: $(ALL)
	$(CXX) $(CXXFLAGS) $(ALL) $(LIBS) -o laplace.out

run:
	np=$1
	Nx=$2
	Ny=$3
	mpirun -np $(np) ./laplace.out $(Nx) $(Ny)

clean:
	rm *.o *.out

.cpp.o:
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LIBS)
