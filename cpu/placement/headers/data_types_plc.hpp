#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// USED BY: everyone
// nothing for now :)


// USED BY: event kernels

struct alignas(8) swap {
    uint32_t lo;
    uint32_t hi;
};
