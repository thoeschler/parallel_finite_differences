CXX=mpicxx
CXXFLAGS=-Wall -O3 -march=native

build: main.o la.o crs.o finite_diff.o topology.o
	$(CXX) $(CXXFLAGS) topology.o finite_diff.o la.o crs.o main.o -o main.out

run:
	np=$1;
	mpirun -np $(np) ./main.out

clean:
	rm *.o *.out

.cpp.o:
	$(CXX) $(CXXFLAGS) -o $@ -c $<
