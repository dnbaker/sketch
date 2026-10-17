# Makefile for the sketch GPU layer (CUDA; hipify-cl portable to HIP/AMD).
# This directory is NOT part of the CPU build; everything here is opt-in.
#
#   make            build the device-bit-identity test binary
#   make test       build + run it
#   make hostcheck  build + run the no-CUDA scalar-semantics checks (any g++)
#   make clean
#
# Set ARCH to match the target GPU, e.g.:
#   make ARCH="-gencode arch=compute_86,code=sm_86"     # RTX 3090
#   make ARCH="-gencode arch=compute_89,code=sm_89"     # RTX 4090
#   make ARCH="-gencode arch=compute_80,code=sm_80"     # A5000/A100
NVCC ?= nvcc
CXX  ?= g++
# Volta is the oldest arch with the u64 atomicMin used by the bbmh kernel.
ARCH ?= -gencode arch=compute_70,code=sm_70
STD  := -std=c++17
# NO_BLAZE keeps blaze out of nvcc's host pass; the reference code in the
# test TU does not need it. -march=native is for the CPU reference side
# (hll.h requires at least SSE4.1); drop or adjust for non-x86 hosts.
NVCCFLAGS := -O3 $(STD) --expt-relaxed-constexpr -DNDEBUG -DNO_BLAZE \
             -I$(CURDIR)/../include -I$(CURDIR)/../include/sketch -I$(CURDIR)/.. \
             -Xcompiler -march=native -Xcompiler -O3 -Xcompiler -pthread
LDFLAGS := -lz -pthread

TEST := tests/test_gpu_sketches
HOSTCHECK := tests/host_math_check

all: $(TEST)

sketch.o: sketch.cu $(wildcard ../include/sketch/gpu/*.cuh)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c sketch.cu -o $@

$(TEST): tests/test_gpu_sketches.cu sketch.o
	$(NVCC) $(ARCH) $(NVCCFLAGS) tests/test_gpu_sketches.cu sketch.o $(LDFLAGS) -o $@

test: $(TEST)
	./$(TEST)

$(HOSTCHECK): tests/host_math_check.cpp $(wildcard ../include/sketch/gpu/*.cuh)
	$(CXX) $(STD) -O2 -DNDEBUG -march=native -DNO_BLAZE -pthread \
	    -I$(CURDIR)/../include -I$(CURDIR)/../include/sketch -I$(CURDIR)/.. -I$(CURDIR)/../include/blaze -I$(CURDIR)/../include/vec \
	    tests/host_math_check.cpp -o $(HOSTCHECK)

hostcheck: $(HOSTCHECK)
	./$(HOSTCHECK)

clean:
	rm -f sketch.o $(TEST) $(HOSTCHECK)

.PHONY: all test hostcheck clean
