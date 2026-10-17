# Makefile for the sketch GPU layer (CUDA; hipify-cl portable to HIP/AMD).
# This directory is NOT part of the CPU build; everything here is opt-in.
#
#   make            build the device-bit-identity test binary
#   make test       build + run it
#   make hostcheck  build + run the no-CUDA scalar-semantics checks (any g++)
#   make clean
#
# Hard rule (learned on a CUDA 12.8.1 box): nvcc must never see the CPU
# sketch headers — frontend ICEs and include-order-dependent constexpr
# errors. So the test binary links three TUs:
#   sketch.cu                    kernels + gpu/*.cuh only (nvcc)
#   tests/test_gpu_sketches.cu   test driver, gpu/*.cuh + reference.h (nvcc)
#   tests/reference.o            ALL CPU-header reference code (plain g++)
#
# Set ARCH to match the target GPU, e.g.:
#   make ARCH="-gencode arch=compute_86,code=sm_86"     # RTX 3090
#   make ARCH="-gencode arch=compute_89,code=sm_89"     # RTX 4090
#   make ARCH="-gencode arch=compute_80,code=sm_80"     # A5000/A100
NVCC ?= nvcc
CXX  ?= g++
# NB: if the environment exports CXX to a broken clang (this Mac does:
# Homebrew libc++), pin the compiler, e.g. make CXX=/usr/local/bin/g++-15 hostcheck
# Volta is the oldest arch with the u64 atomicMin used by the bbmh kernel.
ARCH ?= -gencode arch=compute_70,code=sm_70
STD  := -std=c++17
# -march=native is for the CPU reference side (hll.h requires at least
# SSE4.1); drop or adjust for non-x86 hosts. NO_BLAZE keeps blaze out of both
# host TUs; the reference code does not need it.
NVCCFLAGS := -O3 $(STD) --expt-relaxed-constexpr -DNDEBUG -DNO_BLAZE \
             -I$(CURDIR)/../include -I$(CURDIR)/../include/sketch -I$(CURDIR)/.. \
             -Xcompiler -march=native -Xcompiler -O3 -Xcompiler -pthread
CXXFLAGS := $(STD) -O2 -DNDEBUG -DNO_BLAZE -march=native -pthread \
            -I$(CURDIR)/../include -I$(CURDIR)/../include/sketch -I$(CURDIR)/.. \
            -I$(CURDIR)/../include/blaze -I$(CURDIR)/../include/vec
LDFLAGS := -lz -Xcompiler -pthread

TEST := tests/test_gpu_sketches
HOSTCHECK := tests/host_math_check

all: $(TEST)

sketch.o: sketch.cu $(wildcard ../include/sketch/gpu/*.cuh)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c sketch.cu -o $@

reference.o: tests/reference.cpp tests/reference.h
	$(CXX) $(CXXFLAGS) -c tests/reference.cpp -o $@

$(TEST): tests/test_gpu_sketches.cu sketch.o reference.o
	$(NVCC) $(ARCH) $(NVCCFLAGS) tests/test_gpu_sketches.cu sketch.o reference.o $(LDFLAGS) -o $@

test: $(TEST)
	./$(TEST)

$(HOSTCHECK): tests/host_math_check.cpp $(wildcard ../include/sketch/gpu/*.cuh)
	$(CXX) $(CXXFLAGS) tests/host_math_check.cpp -o $(HOSTCHECK)

hostcheck: $(HOSTCHECK)
	./$(HOSTCHECK)

clean:
	rm -f sketch.o reference.o $(TEST) $(HOSTCHECK)

.PHONY: all test hostcheck clean
