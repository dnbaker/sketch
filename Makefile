.PHONY:all python clean mostlyclean test
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
		 -pedantic -Wunused-variable\
        -Wno-cast-align

FLAGS=-O3 -funroll-loops -pipe -march=native -Iinclude/sketch -I. -Ivec/blaze -Ivec -Ipybind11/include -Iinclude -fpic -Wall $(WARNINGS) \
     -fno-strict-aliasing \
      -DXXH_INLINE_ALL

CXXFLAGS=$(FLAGS) -Wreorder  \

CXXFLAGS+= $(EXTRA)

PYCONF?=python3-config

ifeq ($(shell uname),Darwin)
    UNDEFSTR=-undefined dynamic_lookup
    SLEEF_COMPILER=clang
else
    UNDEFSTR=
    SLEEF_COMPILER=$(CC)
endif


NVCC?=nvcc
EX=$(patsubst testsrc/%.cpp,%,$(wildcard testsrc/*.cpp))
all: $(EX)
setup_tests: $(EX) lztest
	echo $(EX) lztest > tmpfiles.txt

STD?= -std=c++14

#CCBIN?=-ccbin=clang++

GPUFLAGS= $(CCBIN) -O3 -std=c++14 -Iinclude -I. -Xcompiler -march=native -Xcompiler -fopenmp -Iinclude/sketch \
		-lz

INCLUDES=-I`$(PYCONF) --includes` -Ipybind11/include
SUF=`$(PYCONF) --extension-suffix`
OBJS=$(patsubst %.cpp,%$(SUF),$(wildcard *.cpp))
HEADERS=$(wildcard include/sketch/*.h)

SAN=-fsanitize=undefined -fsanitize=address
PYTHON?=python3

python: $(HEADERS) python/hll.cpp python/setup.py
	cd python && $(PYTHON) setup.py install

mpython: python/hll.cpp hll.cpython.so
	$(PYTHON) -c "import subprocess;import site; subprocess.check_call('cp "hll*`$(PYCONF) --extension-suffix`" %s' % site.getsitepackages()[0], shell=True)"

hpython: pybbmh.cpython.so
	$(PYTHON) -c "import subprocess;import site; subprocess.check_call('cp pybbmh.py "*`$(PYCONF) --extension-suffix`" %s' % site.getsitepackages()[0], shell=True)"

%.cpython.so: %.cpp
	$(CXX) $(UNDEFSTR) $(INCLUDES) -fopenmp -O3 -Wall $(CXXFLAGS) -shared $(STD) -fPIC `python3 -m pybind11 --includes` $< -o $*$(SUF) -lz && \
    ln -fs $*$(SUF) $@

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(STD)	$< -o $@

%.o: %.c
	$(CC) -c $(FLAGS)	$< -o $@

%: examples/%.cpp kthread.o $(HEADERS)
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz # $(SAN)

%: testsrc/%.cpp kthread.o $(HEADERS)
	$(CXX) $(STD) $(CXXFLAGS) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz # $(SAN)

heaptest: testsrc/heaptest.cpp kthread.o $(HEADERS)
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz # $(SAN)

divtest: testsrc/divtest.cpp kthread.o $(HEADERS)
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz # $(SAN)

%: src/%.cu $(wildcard include/sketch/cuda/*h)
	$(NVCC) $< -o $@ $(GPUFLAGS)

#%.o: %.cu
#	$(NVCC) $< -c -o $@ $(GPUFLAGS)

%: benchmark/%.cpp kthread.o $(HEADERS)
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o -DNDEBUG=1 $< -o $@ -lz

%_d: src/%.cpp kthread.o $(HEADERS)
	$(CXX) $(CXXFLAGS)	$(STD) -fsanitize=leak -fsanitize=undefined -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz

lztest: testsrc/hlltest.cpp kthread.o $(HEADERS)
	$(CXX) $(CXXFLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o -DLZ_COUNTER $< -o $@ -lz

clean:
	rm -f test.o test hll.o kthread.o *hll*cpython*so $(EX)

test:
	bash ./run_tests.sh


PREFIX?=/usr/local

install: $(HEADERS) compact_vector/include/compact_vector.hpp
	install -d $(DESTDIR)$(PREFIX)/include/sketch/vec && \
	install -d $(DESTDIR)$(PREFIX)/include/compact_vector/include && \
	install -d $(DESTDIR)$(PREFIX)/include/aesctr && \
    install -m 644 $(HEADERS) $(DESTDIR)$(PREFIX)/include/sketch && \
    install -m 644 $(wildcard compact_vector/include/*.hpp) $(DESTDIR)$(PREFIX)/include/compact_vector/include && \
    install -m 644 $(wildcard aesctr/*.h) $(DESTDIR)$(PREFIX)/include/aesctr && \
    install -m 644 $(wildcard vec/*.h) $(DESTDIR)$(PREFIX)/include/sketch/vec

#mctest: mctest.cpp ccm.h
#mctest_d: mctest.cpp ccm.h

mostlyclean: clean
