#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

#include <cuda_runtime.h>
#include <curand.h>

// USED BY: everyone

#define CURAND_CHECK(ans) { curandAssert((ans), #ans, __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char* expr, const char* file, int line, bool abort = true) {
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRANDassert:\n  Error: %d, Expr.: %s\n  File: %s, Line: %d\n", static_cast<int>(code), expr, file, line);
        if (abort) exit(code);
    }
}

#define MAX_ITERATIONS 32 // 1024

#define LEFT 0
#define RIGHT 1
#define UP 2
#define DOWN 3

// TODO: infer this at runtime, make it a device-side constant
//       => infer it especially from the hardware width/height, that determine the manhattan distance range
#define FORCE_FIXED_POINT_SCALE 262144u

typedef struct __align__(8) {
    int32_t x;
    int32_t y;
} coords;

__forceinline__ __device__ uint32_t manhattan(coords c1, coords c2) {
    const uint32_t x_dst = c1.x > c2.x ? c1.x - c2.x : c2.x - c1.x;
    const uint32_t y_dst = c1.y > c2.y ? c1.y - c2.y : c2.y - c1.y;
    return x_dst + y_dst;
}


// USED BY: candidate moves kernel

#define MAX_CANDIDATE_MOVES 4 // must be between 1 and 4


// USED BY: exclusive swaps kernel

#define SWAPS_PATH_SIZE 224u // initial slots for places to see while traversing the swaps tree
#define MAX_SWAPS_MATCHING_REPEATS 64u // number of places that can be handled by the same thread in case of limited space for the cooperative kernel launch


// USED BY: event kernels

typedef struct __align__(8) {
    uint32_t lo;
    uint32_t hi;
} swap;