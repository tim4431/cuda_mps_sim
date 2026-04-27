# Makefile for minimal_tebd_c
# Follows HW2 conventions: nvc++ with Google Test

# Compiler
CXX ?= g++
CXXFLAGS = -O2 -std=c++14

# Google Test
GTEST_DIR = ./googletest/googletest
GTEST_INC = $(GTEST_DIR)/include
GTEST_FLAGS = -isystem $(GTEST_INC) -O2

# ---------------------------------------------------------------------------
# Production target
# ---------------------------------------------------------------------------

LIB_SRCS = tensor.cpp linalg.cpp mps.cpp model.cpp tebd.cpp exact_diag.cpp validate.cpp
LIB_OBJS = $(LIB_SRCS:.cpp=.o)
TARGET = tebd_sim

default: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(LIB_OBJS) main.o
	$(CXX) $(CXXFLAGS) $^ -o $@

validate_driver: $(LIB_OBJS) validate_driver.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# ---------------------------------------------------------------------------
# Google Test library
# ---------------------------------------------------------------------------

gtest-all.o: $(GTEST_DIR)/src/gtest-all.cc
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $< -o $@

gtest_main.o: $(GTEST_DIR)/src/gtest_main.cc
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $< -o $@

gtest_main.a: gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

TEST_CXXFLAGS = $(CXXFLAGS) -isystem $(GTEST_INC)

test_tensor.o: test_tensor.cpp
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@
test_tensor: test_tensor.o tensor.o gtest_main.a
	$(CXX) $^ -o $@

test_linalg.o: test_linalg.cpp
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@
test_linalg: test_linalg.o tensor.o linalg.o gtest_main.a
	$(CXX) $^ -o $@

test_mps.o: test_mps.cpp
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@
test_mps: test_mps.o $(LIB_OBJS) gtest_main.a
	$(CXX) $^ -o $@

test_model.o: test_model.cpp
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@
test_model: test_model.o $(LIB_OBJS) gtest_main.a
	$(CXX) $^ -o $@

test_tebd.o: test_tebd.cpp
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@
test_tebd: test_tebd.o $(LIB_OBJS) gtest_main.a
	$(CXX) $^ -o $@

test_validate.o: test_validate.cpp
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@
test_validate: test_validate.o $(LIB_OBJS) gtest_main.a
	$(CXX) $^ -o $@

tests: test_tensor test_linalg test_mps test_model test_tebd test_validate

run_tests: tests
	@echo "=== test_tensor ===" && ./test_tensor && \
	echo "=== test_linalg ===" && ./test_linalg && \
	echo "=== test_mps ===" && ./test_mps && \
	echo "=== test_model ===" && ./test_model && \
	echo "=== test_tebd ===" && ./test_tebd && \
	echo "=== test_validate ===" && ./test_validate

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean:
	rm -f *.o $(TARGET) validate_driver test_tensor test_linalg test_mps test_model test_tebd test_validate gtest_main.a

.PHONY: default tests run_tests clean
