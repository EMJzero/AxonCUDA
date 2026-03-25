# ==========================================
# Project settings
# ==========================================
TARGET      := hgraph_gpu.exe

SRC_DIR     := sources
HDR_DIR     := headers
INC_DIR     := includes
BUILD_DIR   := build

CXX         := g++
NVCC        := nvcc
ARCH        := native

CXXFLAGS    := -O3 --std=c++20 -Wall -Wextra -fopenmp -I$(HDR_DIR) -I$(INC_DIR)
NVCCFLAGS   := -O3 --std=c++20 -arch=$(ARCH) -dc -allow-unsupported-compiler --extended-lambda \
               -Xcompiler "-Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-function -fopenmp" \
               -I $(HDR_DIR) -I $(INC_DIR)
LINKFLAGS   := --std=c++20 -arch=$(ARCH) -allow-unsupported-compiler --extended-lambda -lgomp

# ==========================================
# Source discovery
# ==========================================
SRC_CPPS    := $(wildcard $(SRC_DIR)/*.cpp)
SRC_CUS     := $(wildcard $(SRC_DIR)/*.cu)

OBJ_CPPS    := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_CPPS))
OBJ_CUS     := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC_CUS))
OBJS        := $(OBJ_CPPS) $(OBJ_CUS)

DEPS        := $(OBJS:.o=.d)

# ==========================================
# Build rules
# ==========================================
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LINKFLAGS) -o $@ $^

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

# ==========================================
# Argument extraction (for run, profile, memcheck)
# ==========================================
RUN_ARG_1     := $(word 2, $(MAKECMDGOALS))
RUN_ARG_2     := $(word 3, $(MAKECMDGOALS))
PROFILE_ARG_1 := $(word 2, $(MAKECMDGOALS))
PROFILE_ARG_2 := $(word 3, $(MAKECMDGOALS))

# Dummy rule to avoid "No rule to make target ..."
$(RUN_ARG_1) $(RUN_ARG_2) $(PROFILE_ARG_1) $(PROFILE_ARG_2):
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
	./$(TARGET) -r hgraphs/8k_model_ordered_processed.snn

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

# Memcheck:
#   make memcheck <input_file>
#   make memcheck <input_file> <config_name>
memcheck: $(TARGET)
	@if [ -z "$(PROFILE_ARG_1)" ]; then \
		echo "Usage: make memcheck <input_file> [cache_string]"; \
		exit 1; \
	elif [ -z "$(PROFILE_ARG_2)" ]; then \
		compute-sanitizer --tool memcheck \
			./$(TARGET) -r $(PROFILE_ARG_1); \
	else \
		compute-sanitizer --tool memcheck \
			./$(TARGET) -r $(PROFILE_ARG_1) -c $(PROFILE_ARG_2); \
	fi

# ==========================================
clean:
	rm -f $(TARGET) *.qdrep *.nsys-rep *.sqlite
	rm -rf $(BUILD_DIR)

.PHONY: all run run8k profile memcheck clean