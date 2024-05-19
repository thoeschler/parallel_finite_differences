HOME=../..
INCLUDE=$(HOME)/include

CXX=mpicxx
CXXFLAGS=-Wall -O3 -march=native -I$(INCLUDE)

ALL=laplace.o cg.o ../la.o ../crs.o finite_diff.o


laplace: $(ALL)
	$(CXX) $(CXXFLAGS) $(ALL) -o ../../laplace_seq.out

run:
	$(HOME)/laplace_seq.out $(Nx) $(Ny)

clean:
	rm *.o *.out

.cpp.o:
	$(CXX) $(CXXFLAGS) -o $@ -c $<
