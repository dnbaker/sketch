.PHONY:all python clean mostlyclean
CXX?=g++
CC?=gcc
ifndef DBG
DBG=-DNDEBUG
else
DBG=
endif
WARNINGS=-Wall -Wextra -Wno-char-subscripts \
		 -Wpointer-arith -Wwrite-strings -Wdisabled-optimization \
		 -Wformat -Wcast-align -Wno-unused-function -Wno-unused-parameter \
		 -pedantic -Wunused-variable -Wno-attributes -Wno-ignored-attributes

FLAGS=-O1 -funroll-loops -pipe -march=native -Ivec/blaze -Ivec -Ipybind11/include -I. -fpic -Wall $(WARNINGS) \
     -fno-strict-aliasing \
      -DXXH_INLINE_ALL  \
	  -Wno-attributes -Wno-pragmas -Wno-ignored-qualifiers

CXXFLAGS=$(FLAGS) -Wreorder

PYCONF?=python3-config

GPUFLAGS= -O3 -std=c++14

ifeq ($(shell uname),Darwin)
    UNDEFSTR=-undefined dynamic_lookup
else
    UNDEFSTR=
endif


NVCC?=nvcc
EX=$(patsubst src/%.cpp,%,$(wildcard src/*.cpp))
all: $(EX)
run_tests: $(EX) lztest
	for i in $(EX) lztest; do ./$$i; done

STD?=-std=c++14

INCLUDES=-I`$(PYCONF) --includes` -Ipybind11/include
SUF=`$(PYCONF) --extension-suffix`
OBJS=$(patsubst %.cpp,%$(SUF),$(wildcard *.cpp))
HEADERS=$(wildcard *.h)

SAN=-fsanitize=undefined -fsanitize=address

sleef.h:
	+cd vec/sleef && mkdir -p build && cd build && cmake .. && $(MAKE) && cd ../../../ && ln -s vec/sleef//build/include/sleef.h sleef.h

python: hll.cpython.so
	+python -c "import subprocess;import site; subprocess.check_call('cp hll.py "*`$(PYCONF) --extension-suffix`" %s' % site.getsitepackages()[0], shell=True)"

%.cpython.so: %.cpp
	$(CXX) $(UNDEFSTR) $(INCLUDES) -fopenmp -O3 -Wall $(CXXFLAGS) -shared $(STD) -fPIC `python3 -m pybind11 --includes` $< -o $*$(SUF) -lz && \
    ln -fs $*$(SUF) $@

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(STD)	$< -o $@

%.o: %.c
	$(CC) -c $(FLAGS)	$< -o $@

%: src/%.cpp kthread.o $(HEADERS) sleef.h
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz  $(SAN)

heaptest: src/heaptest.cpp kthread.o $(HEADERS) sleef.h
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz # $(SAN)

divtest: src/divtest.cpp kthread.o $(HEADERS) sleef.h
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz # $(SAN)

%: src/%.cu
	$(NVCC) $< -o $@ $(GPUFLAGS)

%.o: %.cu
	$(NVCC) $< -c -o $@ $(GPUFLAGS)

%: benchmark/%.cpp kthread.o $(HEADERS)
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o -DNDEBUG=1 $< -o $@ -lz

%_d: src/%.cpp kthread.o $(HEADERS)
	$(CXX) $(CXXFLAGS)	$(STD) -fsanitize=leak -fsanitize=undefined -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz

lztest: src/test.cpp kthread.o $(HEADERS)
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o -DLZ_COUNTER $< -o $@ -lz

dev_test_p: dev_test.cpp kthread.o hll.h
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o -static-libstdc++ -static-libgcc $< -o $@ -lz

clean:
	rm -f test.o test hll.o kthread.o *hll*cpython*so $(EX)

#mctest: mctest.cpp ccm.h
#mctest_d: mctest.cpp ccm.h

mostlyclean: clean
