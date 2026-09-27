#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// NOTE: constants copied from 'placement/headers/defines_plc.cuh', keep them in sync!
// => program-wide and CPU-only constants come from '../headers/defines.hpp'

#include "defines.hpp"

// USED BY: everyone

#define SEED 86u


// TODO: infer this at runtime, make it a global set once
//       => infer it especially from the hardware width/height, that determine the manhattan distance range
#define FORCE_FIXED_POINT_SCALE 131072u

#define MULTISTART_ATTEMPTS -1u // -1 -> decide at runtime based on parallel resource
#define MULTISTART_BATCH_SIZE -1u // -1 -> refine every multi-start attempt in one single batch


// USED BY: recursive bipartitioning

#define LABELPROP_REPEATS 8


// USED BY: candidate moves kernel

#define CANDIDATE_MOVES 4 // must be between 1 and 4


// USED BY: force-directed refinement

#define FD_ITERATIONS 64 // 1024
#define FD_MIN_GAIN 0.001f // below this, a multi-start's best improving prefix is not worth applying
#define PREFIX_GAIN_THREADS 256u // entries per tile of 'prefix_gain_kernel', the block size of its CUDA version (must match)
#define FD_ACTIVE_CHECK_PERIOD 16u // iterations between two checks of whether the whole batch converged
