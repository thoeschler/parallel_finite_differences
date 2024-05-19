include config.mk

HOME=../..
INCLUDE=$(HOME)/include

CXXFLAGS=-Wall -O3 -march=native -Dcommunication_type=$(COMMUNICATION_TYPE) -I$(INCLUDE)
LFLAGS=-Wall -O3 -march=native
LIBS=#-fopenmp


ALL=laplace.o la.o crs.o finite_diff.o topology.o grid.o cg.o

laplace: $(ALL)
	$(CXX) $(ALL) $(LFLAGS) $(LIBS) -o $(HOME)/laplace_par.out

run:
	mpirun -np $(np) $(HOME)/laplace_par.out $(Nx) $(Ny)

clean:
	rm *.o *.out

.cpp.o:
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ -c $<
