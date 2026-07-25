#pragma once
#include <cfloat>
#include <cstdio>
#include <cstdint>
#include <stdint.h>

#include <cuda_runtime.h>
#include <curand.h>

#include "data_types.cuh"
#include "data_types_plc.cuh"
#include "defines_plc.cuh"

// USED BY: everyone

#define CURAND_CHECK(ans) { curandAssert((ans), #ans, __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char* expr, const char* file, int line, bool abort = true) {
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRANDassert:\n  Error: %d, Expr.: %s\n  File: %s, Line: %d\n", static_cast<int>(code), expr, file, line);
        if (abort) exit(code);
    }
}


// USED BY: main
// nothing for now :)