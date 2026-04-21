TARGET      := hgrade.exe
BUILD_DIR   := build
ROOT_DIR    := ..

CXX         := g++
CXXFLAGS    := -O3 --std=c++20 -Wall -Wextra -fopenmp \
               -I$(ROOT_DIR)/headers -I$(ROOT_DIR)/includes -I$(ROOT_DIR)/placement/includes

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
MAKEFLAGS += -j$(NPROC) # run with as many jobs as there are CPU cores

LOCAL_SRCS  := grade.cpp
ROOT_SRCS   := sources/constr.cpp placement/sources/nmhardware.cpp
SRCS        := $(LOCAL_SRCS) $(addprefix $(ROOT_DIR)/,$(ROOT_SRCS))

OBJS        := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(LOCAL_SRCS))
OBJS        += $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(ROOT_SRCS))
DEPS        := $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: $(ROOT_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)

.PHONY: all clean
