SRC_SEQ=src/seq
SRC_PAR=src/par

sequential:
	cd $(SRC_SEQ) && make laplace

parallel:
	cd $(SRC_PAR) && make laplace

run_seq:
	cd $(SRC_SEQ) && make run Nx=$(Nx) Ny=$(Ny)

run_par:
	cd $(SRC_PAR) && make run np=$(np) Nx=$(Nx) Ny=$(Ny)

clean:
	rm -f src/par/*.o  src/seq/*.o *.out
