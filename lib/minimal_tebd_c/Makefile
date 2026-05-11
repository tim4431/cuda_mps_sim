# Unified Makefile for the minimal_tebd_c library.
#
# Layout:
#   core/        CPU library  (.cpp/.h)
#   gpu/         CUDA backend (.cu/.h)
#   apps/        driver/bench binaries
#   tests/       Google Test binaries
#   scripts/     plotting / analysis (not built)
#   third_party/googletest/
#
# Targets:
#   make            build CPU sim (+ CUDA apps if nvcc is present)
#   make tests      build Google Test binaries
#   make run_tests  build and run the test suite
#   make bench      build all GPU benchmark binaries
#   make clean
#
# Header search uses -Icore -Igpu, so source files use plain
# `#include "tensor.h"` and `#include "gpu_tebd.h"` regardless of subdir.
# The app list is driven by $(wildcard), so the build naturally adapts to
# whichever apps exist in apps/.

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------

CXX      ?= g++
CXXFLAGS  = -O2 -std=c++14 -Icore -Igpu

# CUDA auto-detection.  Override CUDA_HOME on the make command line if needed.
CUDA_HOME ?= /data/yihuai/miniforge3/envs/imitation
NVCC      ?= $(CUDA_HOME)/bin/nvcc
NVCC_FOUND := $(shell test -x $(NVCC) && echo yes)

ifeq ($(NVCC_FOUND),yes)
  CUDA_ARCH     ?= sm_120
  CUDA_CCBIN    ?= /usr/bin/g++-13
  CUDA_INC       = -I$(CUDA_HOME)/targets/x86_64-linux/include
  NVCCFLAGS      = -O2 -std=c++14 -arch=$(CUDA_ARCH) -ccbin $(CUDA_CCBIN) \
                   -Xcompiler -O2 -Icore -Igpu $(CUDA_INC)
  CUDA_LDFLAGS   = -L$(CUDA_HOME)/targets/x86_64-linux/lib \
                   -lcublas -lcusolver -lcudart \
                   -Wl,-rpath,$(CUDA_HOME)/targets/x86_64-linux/lib
endif

# ---------------------------------------------------------------------------
# Source listings (wildcard-driven so the build adapts as files are added)
# ---------------------------------------------------------------------------

CORE_SRCS := $(wildcard core/*.cpp)
CORE_OBJS := $(CORE_SRCS:.cpp=.o)

GPU_SRCS  := $(wildcard gpu/*.cu)
GPU_OBJS  := $(GPU_SRCS:.cu=.o)

# CPU app: apps/sim (no CUDA dep).
CPU_APP_SRCS := $(wildcard apps/sim.cpp)
CPU_APPS     := $(CPU_APP_SRCS:.cpp=)

# GPU apps: every other app source in apps/.
GPU_APP_SRCS := $(filter-out apps/sim.cpp, $(wildcard apps/*.cpp))
GPU_APPS     := $(GPU_APP_SRCS:.cpp=)

ifeq ($(NVCC_FOUND),yes)
  ALL_APPS := $(CPU_APPS) $(GPU_APPS)
else
  ALL_APPS := $(CPU_APPS)
endif

# ---------------------------------------------------------------------------
# Build rules
# ---------------------------------------------------------------------------

default: $(ALL_APPS)

bench: $(GPU_APPS)

# Generic .cpp -> .o (picked up for core/, apps/, tests/).
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA .cu -> .o.
gpu/%.o: gpu/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# CPU-only sim.
apps/sim: apps/sim.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Static pattern rule covering every GPU app.  When GPU_APPS is empty
# (no gpu/ files yet) the rule simply matches nothing.
$(GPU_APPS): apps/%: apps/%.o $(CORE_OBJS) $(GPU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CUDA_LDFLAGS)

# ---------------------------------------------------------------------------
# Google Test (vendored).
# ---------------------------------------------------------------------------

GTEST_DIR    = third_party/googletest/googletest
GTEST_INC    = $(GTEST_DIR)/include
GTEST_FLAGS  = -isystem $(GTEST_INC) -O2

gtest-all.o: $(GTEST_DIR)/src/gtest-all.cc
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $< -o $@

gtest_main.o: $(GTEST_DIR)/src/gtest_main.cc
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $< -o $@

gtest_main.a: gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# Test .cpp -> .o needs the gtest include path on top of CXXFLAGS.
tests/%.o: tests/%.cpp
	$(CXX) $(CXXFLAGS) -isystem $(GTEST_INC) -c $< -o $@

# Test binaries.
tests/test_tensor: tests/test_tensor.o core/tensor.o gtest_main.a
	$(CXX) $^ -o $@
tests/test_linalg: tests/test_linalg.o core/tensor.o core/linalg.o gtest_main.a
	$(CXX) $^ -o $@
tests/test_mps: tests/test_mps.o $(CORE_OBJS) gtest_main.a
	$(CXX) $^ -o $@
tests/test_model: tests/test_model.o $(CORE_OBJS) gtest_main.a
	$(CXX) $^ -o $@
tests/test_tebd: tests/test_tebd.o $(CORE_OBJS) gtest_main.a
	$(CXX) $^ -o $@
tests/test_validate: tests/test_validate.o $(CORE_OBJS) gtest_main.a
	$(CXX) $^ -o $@

TEST_BINS := tests/test_tensor tests/test_linalg tests/test_mps \
             tests/test_model tests/test_tebd tests/test_validate

tests: $(TEST_BINS)

run_tests: $(TEST_BINS)
	@set -e; for t in $(TEST_BINS); do echo "=== $$t ==="; $$t; done

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:
	rm -f core/*.o gpu/*.o apps/*.o tests/*.o
	rm -f gtest-all.o gtest_main.o gtest_main.a
	rm -f $(CPU_APPS) $(GPU_APPS) $(TEST_BINS)

.PHONY: default bench tests run_tests clean
