.PHONY:all python clean mostlyclean
ifndef CXX
CXX=g++
endif
ifndef CC
CC=gcc
endif
ifndef DBG
DBG=-DNDEBUG
else
DBG=
endif
FLAGS=-O3 -funroll-loops -pipe -march=native -mavx2 -I. -fpic -Wall -Wextra -Wdisabled-optimization -Wno-unused-parameter

ifeq ($(shell uname),Darwin)
    UNDEFSTR=-undefined dynamic_lookup
else
    UNDEFSTR=
endif

all: test libhll.a

libhll.a: hll.o
	ar cr $@ $<

INCLUDES=-I`python3-config --includes` -Ipybind11/include
SUF=`python3-config --extension-suffix`
OBJS=$(patsubst %.cpp,%$(SUF),$(wildcard *.cpp))

python: _hll.cpython.so
	python -c "import subprocess;import site; subprocess.check_call('cp hll.py "*`python3-config --extension-suffix`" %s' % site.getsitepackages()[0], shell=True)"

%.cpython.so: %.cpp hll.o
	$(CXX) $(UNDEFSTR) $(INCLUDES) -O3 -Wall $(FLAGS) $(INC) -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` $< -o $*$(SUF) && \
    ln -fs $*$(SUF) $@

%.o: %.cpp
	$(CXX) -c $(FLAGS) -std=c++17	$< -o $@
%.o: %.c
	$(CC) -c $(FLAGS)	$< -o $@

test: test.cpp hll.o kthread.o
	$(CXX) $(FLAGS)	-std=c++17 -Wno-unused-parameter hll.o kthread.o -pthread $< -o $@

serial_test: serial_test.cpp hll.o kthread.o
	$(CXX) $(FLAGS)	-std=c++17 -Wno-unused-parameter -pthread $< -o $@

dev_test: dev_test.cpp kthread.o
	$(CXX) $(FLAGS)	-std=c++17 -Wno-unused-parameter -pthread -DENABLE_HLL_DEVELOP -DHLL_HEADER_ONLY kthread.o $< -o $@

clean:
	rm -f test.o test hll.o kthread.o libhll.a *hll*cpython*so

mostlyclean: clean
