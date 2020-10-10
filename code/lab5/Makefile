IDIR=include
CXX=mpicxx
CXXFLAGS=-I$(IDIR) -std=c++11 -fopenmp -O3

ODIR=src
LDIR =../lib

LIBS=-lm -fopenmp 

_DEPS = deep_core.h vector_ops.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = deep_core.o vector_ops.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< ${CXXFLAGS}

%.o: %.cxx $(DEPS)
	$(CXX) -c -o $@ $< ${CXXFLAGS}

nnetwork_mpi.o:
	$(CXX) -c -o $@ nnetwork_mpi.cxx ${CXXFLAGS} -DUSE_MPI

nnetwork_mpi: $(OBJ) nnetwork_mpi.o
	$(CXX) -o $@ $^ $(LIBS)

run_parallel:
	mpirun -np 4 ./nnetwork_mpi

all: clean nnetwork_mpi
.PHONY: clean

default: clean nnetwork 

.DEFAULT_GOAL := default

clean:
	rm -f $(ODIR)/*.o *.o nnetwork_mpi
