#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// USED BY: everyone

typedef struct __align__(8) {
    int32_t x;
    int32_t y;
} coords;


// USED BY: event kernels

typedef struct __align__(8) {
    uint32_t lo;
    uint32_t hi;
} swap;