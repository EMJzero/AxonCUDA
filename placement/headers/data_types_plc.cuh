#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// USED BY: everyone
// nothing for now :)


// USED BY: event kernels

typedef struct __align__(8) {
    uint32_t lo;
    uint32_t hi;
} swap;