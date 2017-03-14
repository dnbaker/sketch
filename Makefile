CXX=g++
FLAGS=-O3 -funroll-loops -pipe -march=native -I. -fpic -std=c++17 -Wall -Wextra -Wdisabled-optimization -DNDEBUG 

ifeq ($(shell uname),Darwin)
	FLAGS := $(FLAGS) -Wa,-q
endif


all: test libhll.a

libhll.a: hll.o
	ar cr $@ $<

hll.o: hll.cpp
	$(CXX) -c $(FLAGS)	$< -o $@

test: test.cpp hll.o
	$(CXX) $(FLAGS)	hll.o $< -o $@

clean:
	rm -f test.o test hll.o libhll.a
