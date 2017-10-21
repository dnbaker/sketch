CXX=g++
CC=gcc
FLAGS=-O3 -funroll-loops -pipe -march=native -I. -fpic -Wall -Wextra -Wdisabled-optimization -DNDEBUG -Wno-unused-parameter

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

%.o: %.cpp
	$(CXX) -c $(FLAGS) -std=c++17	$< -o $@
%.o: %.c
	$(CC) -c $(FLAGS)	$< -o $@

test: test.cpp hll.o kthread.o
	$(CXX) $(FLAGS)	-Wno-unused-parameter hll.o kthread.o -pthread $< -o $@

clean:
	rm -f test.o test hll.o kthread.o libhll.a
