# ==========================================
# Project settings
# ==========================================
TARGET      := hgraph_cpu.exe

SRC_DIRS    := sources kernels
HDR_DIR     := headers
INC_DIR     := includes
BUILD_DIR   := build

# shared with the CUDA implementation (one level up)
EXT_INC_DIR := ../includes
EXT_SRCS    := ../sources/constr.cpp

CXX         := g++

# NOTE: '-ffp-contract=off' keeps g++ from fusing multiply-adds that nvcc did not fuse (and vice versa, those that nvcc
#       did fuse are written as explicit 'fmaf'), so that float arithmetic follows the CUDA kernels operation by operation
CXXFLAGS    := -O3 --std=c++20 -Wall -Wextra -Wno-unused-function -Wno-parentheses -Wno-dangling-else -fopenmp -march=native -ffp-contract=off \
               -I$(HDR_DIR) -I$(INC_DIR) -I$(EXT_INC_DIR)
LINKFLAGS   := -fopenmp

# extra compile-time flags (e.g. make EXTRA=-DABLATE_ONE_MOVE_PER_ROUND BUILD_DIR=build_abl TARGET=hgraph_cpu_abl.exe)
EXTRA       :=
CXXFLAGS    += $(EXTRA)

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
MAKEFLAGS += -j$(NPROC) # run with as many jobs as there are CPU cores

# ==========================================
# Source discovery
# ==========================================
SRC_CPPS    := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cpp))

OBJS        := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC_CPPS))
OBJS        += $(patsubst ../%.cpp,$(BUILD_DIR)/ext/%.o,$(EXT_SRCS))

DEPS        := $(OBJS:.o=.d)

# ==========================================
# Build rules
# ==========================================
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LINKFLAGS) -o $@ $^

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# own sources
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# shared sources
$(BUILD_DIR)/ext/%.o: ../%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

# ==========================================
# Argument extraction (for run)
# ==========================================
RUN_ARG_1     := $(word 2, $(MAKECMDGOALS))
RUN_ARG_2     := $(word 3, $(MAKECMDGOALS))

# Dummy rule to avoid "No rule to make target ..."
$(RUN_ARG_1) $(RUN_ARG_2):
	@:

# ==========================================
# Commands
# ==========================================
# Run:
#   make run
#   make run <input_file>
#   make run <input_file> <config_name>
run: $(TARGET)
	@if [ -z "$(RUN_ARG_1)" ]; then \
		./$(TARGET); \
	elif [ -z "$(RUN_ARG_2)" ]; then \
		./$(TARGET) -r $(RUN_ARG_1); \
	else \
		./$(TARGET) -r $(RUN_ARG_1) -c $(RUN_ARG_2); \
	fi

# ==========================================
clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)

.PHONY: all run clean
