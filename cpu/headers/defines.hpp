#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// NOTE: program-wide constants are shared with the CUDA implementation, so that both run the very same algorithm
// => put here CPU-only constants, and into each header kernel-specific constants (copied from the CUDA headers, keep them in sync!)
#include "../../headers/defines.cuh"


// USED BY: everyone (CPU only)

#define PARALLEL_GRAIN 16384u // below this many items, loops and primitives run sequentially rather than pay the fork-join overhead
#define DYNAMIC_CHUNK 64 // chunk size of dynamically scheduled loops over nodes, groups, and hedges (their degrees are skewed)


// USED BY: refinement (CPU only)

#define DENSE_PPP_RAM_FRACTION 0.5f // fraction of the available RAM the dense pins per partition matrix may take before switching to the sparse one
