# ==========================================
# Project settings
# ==========================================
TARGET      := hgraph_gpu.exe
CXX         := g++
NVCC        := nvcc

# TODO: change to -O3 for the final version, keep -O1 for fast compilation during development
CXXFLAGS    := -O1 --std=c++20 -Wall -Wextra -I.
NVCCFLAGS   := -O1 --std=c++20 -arch=native -dc -allow-unsupported-compiler --extended-lambda -Xcompiler "-Wall -Wextra -Wno-maybe-uninitialized" -I .
LINKFLAGS   := --std=c++20 -arch=native -allow-unsupported-compiler --extended-lambda

SRC_CPPS    := #main.cpp
SRC_CUS     := main.cu kernel.cu

OBJ_CPPS    := $(SRC_CPPS:.cpp=.o)
OBJ_CUS     := $(SRC_CUS:.cu=.o)

OBJS        := $(OBJ_CPPS) $(OBJ_CUS)

# ==========================================
# Build rules
# ==========================================
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LINKFLAGS) -o $@ $^

%.o: %.cpp hgraph.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu hgraph.hpp utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ==========================================
# Argument extraction (for run and profile)
# ==========================================
RUN_ARG     := $(word 2, $(MAKECMDGOALS))
PROFILE_ARG := $(word 2, $(MAKECMDGOALS))

# Dummy rule to avoid "No rule to make target 'filename'"
$(RUN_ARG) $(PROFILE_ARG):
	@:

# ==========================================
# Commands
# ==========================================
# Run:
#   make run <input_file>
run: $(TARGET)
	@if [ -z "$(RUN_ARG)" ]; then \
		./$(TARGET); \
	else \
		./$(TARGET) -r $(RUN_ARG); \
	fi

#   make run8k
run8k: $(TARGET)
	./$(TARGET) -r hgraphs/8k_model_ordered_processed

# Profile:
#   make profile <input_file>
profile: $(TARGET)
	@if [ -z "$(PROFILE_ARG)" ]; then \
		echo "Usage: make profile <input_file>"; \
		exit 1; \
	fi
	nsys profile --stats=true --force-overwrite=true \
		--output=$(TARGET)_profile \
		./$(TARGET) -r $(PROFILE_ARG)

# ==========================================
clean:
	rm -f $(TARGET) *.o *.qdrep *.nsys-rep *.sqlite

.PHONY: all run profile clean
