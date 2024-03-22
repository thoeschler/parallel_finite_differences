CXX=g++
CXXFLAGS=-Wall -Wextra -O3 -march=native

build: main.o la.o crs.o finite_diff.o
	$(CXX) $(CXXFLAGS) finite_diff.o la.o crs.o main.o -o main.out

run:
	./main.out

clean:
	rm *.o *.out

.cpp.o:
	$(CXX) $(CXXFLAGS) -o $@ -c $<
