.PHONY:all python
ifndef CXX
CXX=g++
endif
ifndef CC
CC=gcc
endif
FLAGS=-O3 -funroll-loops -pipe -march=native -mavx2 -I. -fpic -Wall -Wextra -Wdisabled-optimization -DNDEBUG -Wno-unused-parameter

ifneq (,$(findstring g++,$(CXX)))
	ifeq ($(shell uname),Darwin)
		ifeq (,$(findstring clang,$(CXX)))
			FLAGS := $(FLAGS) -Wa,-q
		endif
	endif
endif

all: test libhll.a

libhll.a: hll.o
	ar cr $@ $<

INCLUDES=-I`python3-config --includes` -Ipybind11/include
SUF=`python3-config --extension-suffix`
OBJS=$(patsubst %.cpp,%.cpython.so,$(wildcard *.cpp))

python: _hll.cpython.so

%.cpython.so: %.cpp hll.o
	$(CXX) -undefined dynamic_lookup $(INCLUDES) -O3 -Wall $(FLAGS) $(INC) -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` $< -o $*$(SUF) && \
    ln -fs $*$(SUF) $@

%.o: %.cpp
	$(CXX) -c $(FLAGS) -std=c++17	$< -o $@
%.o: %.c
	$(CC) -c $(FLAGS)	$< -o $@

test: test.cpp hll.o kthread.o
	$(CXX) $(FLAGS)	-Wno-unused-parameter hll.o kthread.o -pthread $< -o $@

clean:
	rm -f test.o test hll.o kthread.o libhll.a *hll*cpython*so
