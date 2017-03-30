CXX=g++
CC=gcc
FLAGS=-O3 -funroll-loops -pipe -march=native -I. -fpic -std=c++17 -Wall -Wextra -Wdisabled-optimization -DNDEBUG -DTHREADSAFE

ifeq ($(shell uname),Darwin)
	FLAGS := $(FLAGS) -Wa,-q
endif


all: test libhll.a

libhll.a: hll.o
	ar cr $@ $<

%.o: %.cpp
	$(CXX) -c $(FLAGS)	$< -o $@
%.o: %.c
	$(CXX) -c $(FLAGS)	$< -o $@

test: test.cpp hll.o kthread.o
	$(CXX) $(FLAGS)	-Wno-unused-parameter hll.o kthread.o -pthread $< -o $@

clean:
	rm -f test.o test hll.o kthread.o libhll.a
