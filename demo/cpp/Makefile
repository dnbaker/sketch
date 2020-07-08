

all: demo

INCLUDE = -I../.. -I../../include

CXXFLAGS += $(LDFLAGS) -O3 -march=native $(INCLUDE) -lz


%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
