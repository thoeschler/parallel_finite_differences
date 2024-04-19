CXX=g++
CXXFLAGS=-Wall -Wextra -O3 -march=native

build: laplace.o la.o crs.o finite_diff.o
	$(CXX) $(CXXFLAGS) finite_diff.o la.o crs.o laplace.o -o laplace.out

run:
	Nx=$1
	Ny=$2
	./laplace.out $(Nx) $(Ny)

clean:
	rm *.o *.out

.cpp.o:
	$(CXX) $(CXXFLAGS) -o $@ -c $<
