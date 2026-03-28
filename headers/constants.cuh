#pragma once
#include <cstdint>

#include <cuda_runtime.h>

// DEVICE CONSTANTS:
extern __constant__ uint32_t max_nodes_per_part;
extern __constant__ uint32_t max_inbound_per_part;