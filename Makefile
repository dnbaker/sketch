CXX=g++
FLAGS=-O3 -funroll-loops -DNDEBUG -pipe -march=native -I. -Wa,-q -mavx2

test: test.cpp
	$(CXX) $(FLAGS)	$< -o $@

clean:
	rm -f test.o test
