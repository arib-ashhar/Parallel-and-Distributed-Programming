# ---------------------------------------------------------------------------
# On HPC:    make          (USE_CUDA=1 by default)
# On macOS:  make USE_CUDA=0
# ---------------------------------------------------------------------------
USE_CUDA ?= 1

ifeq ($(USE_CUDA), 1)
    CXX      = nvcc
    CXXFLAGS = -std=c++17 -O2 -DUSE_CUDA -arch=sm_35 -Wno-deprecated-gpu-targets -Xcompiler -fopenmp,-std=c++17
    LDFLAGS  = -lgomp
    ALL_SRCS = PointCloud.cpp knnKernel.cu approxKnnKernel.cu kmeansKernel.cu
else
    CXX        = clang++
    OMP_PREFIX = $(shell brew --prefix libomp)
    CXXFLAGS   = -std=c++17 -O2 -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include -Wall -Wextra
    LDFLAGS    = -L$(OMP_PREFIX)/lib -lomp
    ALL_SRCS   = PointCloud.cpp
endif

SOLUTION = a2
TEST     = test_runner

all: $(SOLUTION) $(TEST)

$(SOLUTION): main.cpp $(ALL_SRCS) PointCloud.h
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(ALL_SRCS) $(LDFLAGS)

$(TEST): test.cpp $(ALL_SRCS) PointCloud.h
	$(CXX) $(CXXFLAGS) -o $@ test.cpp $(ALL_SRCS) $(LDFLAGS)

test: $(TEST)
	./$(TEST)

clean:
	rm -f $(SOLUTION) $(TEST) *.txt *.o

.PHONY: all test clean