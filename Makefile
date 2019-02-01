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
FLAGS=-O3 -funroll-loops -pipe -march=native -msse2 -mavx2 -Ivec -I. -fpic -Wall -Wextra \
	-Wdisabled-optimization -Wno-unused-parameter -pedantic -fno-strict-aliasing \
	-Wno-attributes -Wno-pragmas # -Wsuggest-attribute=malloc

ifeq ($(shell uname),Darwin)
    UNDEFSTR=-undefined dynamic_lookup
else
    UNDEFSTR=
endif


EX=$(patsubst %.cpp,%,$(wildcard *test.cpp))
all: $(EX)

STD?=-std=c++14

INCLUDES=-I`python3-config --includes` -Ipybind11/include
SUF=`python3-config --extension-suffix`
OBJS=$(patsubst %.cpp,%$(SUF),$(wildcard *.cpp))
HEADERS=$(wildcard *.h)

python: _hll.cpython.so
	python -c "import subprocess;import site; subprocess.check_call('cp hll.py "*`python3-config --extension-suffix`" %s' % site.getsitepackages()[0], shell=True)"

%.cpython.so: %.cpp hll.o
	$(CXX) $(UNDEFSTR) $(INCLUDES) -O3 -Wall $(FLAGS) $(INC) -shared $(STD) -fPIC `python3 -m pybind11 --includes` $< -o $*$(SUF) -lz && \
    ln -fs $*$(SUF) $@

%.o: %.cpp
	$(CXX) -c $(FLAGS) $(STD)	$< -o $@

%.o: %.c
	$(CC) -c $(FLAGS)	$< -o $@

%: src/%.cpp kthread.o $(HEADERS)
	$(CXX) $(FLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz

test: src/test.cpp kthread.o $(HEADERS)
	$(CXX) $(FLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz

mctest: src/mctest.cpp kthread.o $(HEADERS)
	$(CXX) $(FLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o -O1 $< -o $@ -lz

%_d: %.cpp kthread.o $(HEADERS)
	$(CXX) $(FLAGS)	$(STD) -fsanitize=leak -fsanitize=undefined -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz

lztest: src/test.cpp kthread.o $(HEADERS)
	$(CXX) $(FLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o -DLZ_COUNTER $< -o $@ -lz

serial_test: serial_test.cpp hll.h
	$(CXX) $(FLAGS)	$(STD) -Wno-unused-parameter -pthread -DNOT_THREADSAFE $< -o $@ -lz

dev_test: dev_test.cpp kthread.o hll.h
	$(CXX) $(FLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o $< -o $@ -lz

dev_test_p: dev_test.cpp kthread.o hll.h
	$(CXX) $(FLAGS)	$(STD) -Wno-unused-parameter -pthread kthread.o -static-libstdc++ -static-libgcc $< -o $@ -lz

clean:
	rm -f test.o test hll.o kthread.o *hll*cpython*so $(EX)

#mctest: mctest.cpp ccm.h
#mctest_d: mctest.cpp ccm.h

mostlyclean: clean
