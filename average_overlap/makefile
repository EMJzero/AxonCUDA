# ==========================================
# Project settings
# ==========================================
TARGET      := hgraph_gpu.exe
CXX         := g++
NVCC        := nvcc

CXXFLAGS    := -O3 --std=c++20 -Wall -Wextra -I.
NVCCFLAGS   := -O3 --std=c++20 -arch=native -Xcompiler "-Wall -Wextra" -I.

SRC_CPPS    := main.cpp
SRC_CUS     := kernel.cu

OBJ_CPPS    := $(SRC_CPPS:.cpp=.o)
OBJ_CUS     := $(SRC_CUS:.cu=.o)

OBJS        := $(OBJ_CPPS) $(OBJ_CUS)

# ==========================================
# Build rules
# ==========================================
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cpp hgraph.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu hgraph.hpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ==========================================
# Shorthands
# ==========================================
run: $(TARGET)
	./$(TARGET)

run8k: $(TARGET)
	./$(TARGET) -r hgraphs/8k_model_ordered_processed

clean:
	rm -f $(TARGET) *.o

# avoid conflicts with files named "clean" or "all"
.PHONY: all run clean
