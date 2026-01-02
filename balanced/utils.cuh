#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// USED BY: refinement constraints checks

// valid values filtering functor
struct masked_value_functor_small {
    const float* value;
    const uint32_t* valid;
    __host__ __device__ float operator()(uint32_t i) const { return valid[i] == 0 ? value[i] : -FLT_MAX; }
};