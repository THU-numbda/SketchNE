ifeq (, $(shell which jemalloc-config))
JEMALLOC = 
else
JEMALLOCLD = $(shell jemalloc-config --libdir)
JEMALLOC = -L$(JEMALLOCLD) -ljemalloc
endif

# Always compile with LONG (note that timings on small graphs may be a bit
# faster w/o this flag).
INTT = -DLONG

ifdef EDGELONG
INTE = -DEDGELONG
endif

# If you install intel mkl with anaconda, then you need to set something like:
# INCLUDE_DIRS = -I./ligra -I./pbbslib -I./mklfreigs -I"{ANACONDA_PATH}/envs/sketchne/include"
# LINK_DIRS = -L"{ANACONDA_PATH}/envs/sketchne/lib"

INCLUDE_DIRS = -I./ligra -I./pbbslib -I./mklfreigs -I/opt/intel/mkl/include
LINK_DIRS = -L"/opt/intel/mkl/lib/intel64"

OPT = -O3 -DNDEBUG -DMKL_ILP64 -m64 #-g
#OPT = -O0 -g -DMKL_ILP64 -m64

WARNINGS = -Wno-deprecated-declarations -Wno-sign-compare -Wno-unused-variable

CFLAGS = $(INCLUDE_DIRS) -mcx16 -ldl -std=c++17 -march=native -Wall $(WARNINGS) $(OPT) $(INTT) $(INTE) -DAMORTIZEDPD $(CONCEPTS) -DUSEMALLOC $(LINK_DIRS) -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm


OMPFLAGS = -DOPENMP -fopenmp #default we use openmp
CILKFLAGS = -DCILK -fcilkplus
HGFLAGS = -DHOMEGROWN -pthread

ifdef CLANG
CC = clang++
PFLAGS = $(CILKFLAGS)
else ifdef CILK
CC = g++
PFLAGS = $(CILKFLAGS)
else ifdef OPENMP
CC = g++
PFLAGS = $(OMPFLAGS)
else ifdef HOMEGROWN
CC = g++
PFLAGS = $(HGFLAGS)
else ifdef SERIAL
CC = g++
PFLAGS =
else
CC = g++
PFLAGS = $(HGFLAGS)
endif


SRC = $(wildcard *.cpp)

$(info $(SRC) )

TARGETS = sketchne

all:$(TARGETS)

$(TARGETS):%:%.cpp
	$(CC) $< $(CFLAGS) $(OMPFLAGS) -o $@

.PHONY : clean all

clean :
	rm -f *.o $(TARGETS)
