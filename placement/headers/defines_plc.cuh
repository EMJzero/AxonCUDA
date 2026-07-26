#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// USED BY: main

#define DEVICE_ID 0

#define VERBOSE_LENGTH 20
#define VERBOSE_LAUNCHES true
#define VERBOSE_INFO true
#define VERBOSE_LOGS false
#define VERBOSE_ERRS true
#define DEBUG_ON false


// USED BY: everyone

#define SEED 86u


// TODO: infer this at runtime, make it a device-side constant
//       => infer it especially from the hardware width/height, that determine the manhattan distance range
#define FORCE_FIXED_POINT_SCALE 131072u

#define MULTISTART_ATTEMPTS -1u // -1 -> decide at runtime based on parallel resource
#define NUM_HOST_THREADS -1u // -1 -> decide at runtime based cores count


// USED BY: recursive bipartitioning

#define LABELPROP_REPEATS 8


// USED BY: candidate moves kernel

#define CANDIDATE_MOVES 4 // must be between 1 and 4


// USED BY: force-directed refinement

#define FD_ITERATIONS 64 // 1024