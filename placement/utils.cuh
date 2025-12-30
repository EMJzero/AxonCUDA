#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// USED BY: everyone

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


// USED BY: event kernels

typedef struct __align__(8) {
    uint32_t lo;
    uint32_t hi;
} swap;