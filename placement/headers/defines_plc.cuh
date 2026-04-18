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


// USED BY: everyone

#define SEED 86u

#define LEFT 0
#define RIGHT 1
#define UP 2
#define DOWN 3

// TODO: infer this at runtime, make it a device-side constant
//       => infer it especially from the hardware width/height, that determine the manhattan distance range
#define FORCE_FIXED_POINT_SCALE 131072u

#define MULTISTART_ATTEMPTS -1u // -1 -> decide at runtime based on parallel resource
#define NUM_HOST_THREADS -1u // -1 -> decide at runtime based cores count


// USED BY: recursive bipartitioning

#define LABELPROP_REPEATS 16


// USED BY: candidate moves kernel

// TODO: go exotic with the stencil, not just a "+", but extend this beyond 4 with the 8-point stencil for instance!!
#define MAX_CANDIDATE_MOVES 4 // must be between 1 and 4


// USED BY: force-directed refinement

#define FD_ITERATIONS 32 // 1024