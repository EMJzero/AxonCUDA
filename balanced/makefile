# ==========================================
# Project settings
# ==========================================
TARGET      := balance_gpu.exe
CXX         := g++
NVCC        := nvcc

# TODO: change to -O3 for the final version, keep -O1 for fast compilation during development
CXXFLAGS    := -O1 --std=c++20 -Wall -Wextra -I.
NVCCFLAGS   := -O1 --std=c++20 -arch=native -dc -allow-unsupported-compiler --extended-lambda -Xcompiler "-Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-function" -I .
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

%.o: %.cpp ../hgraph.hpp ../nmhardware.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu ../hgraph.hpp ../nmhardware.hpp ../utils.cuh utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ==========================================
# Argument extraction (for run and profile)
# ==========================================
RUN_ARG_1     := $(word 2, $(MAKECMDGOALS))
RUN_ARG_2     := $(word 3, $(MAKECMDGOALS))
PROFILE_ARG_1 := $(word 2, $(MAKECMDGOALS))
PROFILE_ARG_2 := $(word 3, $(MAKECMDGOALS))

# Dummy rule to avoid "No rule to make target 'filename'"
$(RUN_ARG1) $(RUN_ARG2) $(PROFILE_ARG1) $(PROFILE_ARG2):
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

#   make run8k
run8k: $(TARGET)
	./$(TARGET) -r ../hgraphs/8k_model_ordered_processed

# Profile:
#   make profile <input_file>
#   make profile <input_file> <config_name>
profile: $(TARGET)
	@if [ -z "$(PROFILE_ARG_1)" ]; then \
		echo "Usage: make profile <input_file> [cache_string]"; \
		exit 1; \
	elif [ -z "$(PROFILE_ARG_2)" ]; then \
		nsys profile --stats=true --force-overwrite=true \
			--output=$(TARGET)_profile \
			./$(TARGET) -r $(PROFILE_ARG_1); \
	else \
		nsys profile --stats=true --force-overwrite=true \
			--output=$(TARGET)_profile \
			./$(TARGET) -r $(PROFILE_ARG_1) -c $(PROFILE_ARG_2); \
	fi

# ==========================================
clean:
	rm -f $(TARGET) *.o *.qdrep *.nsys-rep *.sqlite

.PHONY: all run run8k profile clean
