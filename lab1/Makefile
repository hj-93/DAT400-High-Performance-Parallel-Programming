IDIR=include
#Choose compiler
#CXX=
CXXFLAGS=-I$(IDIR) -std=c++11

ODIR=src
LDIR =../lib

LIBS=-lm -fopenmp 

_DEPS = deep_core.h vector_ops.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = deep_core.o vector_ops.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

deep_core.o: deep_core.cpp $(DEPS)
	g++ -c -o $@ $< ${CXXFLAGS}

vector_ops.o: vector_ops.cpp $(DEPS)
	g++ -c -o $@ $< ${CXXFLAGS}


nnetwork_mpi.o:
	g++ -c -o $@ nnetwork.cxx ${CXXFLAGS} -DUSE_MPI

nnetwork.o:
	g++ -c -o $@ nnetwork.cxx ${CXXFLAGS} 

nnetwork_mpi: $(OBJ) nnetwork_mpi.o
	g++ -o $@ $^ $(LIBS)

nnetwork: $(OBJ) nnetwork.o
	g++ -o $@ $^ $(LIBS)

run_serial:
	./nnetwork

all: clean nnetwork_mpi nnetwork
.PHONY: clean
default: clean nnetwork 

.DEFAULT_GOAL := default

clean:
	rm -f $(ODIR)/*.o *.o nnetwork_mpi nnetwork
