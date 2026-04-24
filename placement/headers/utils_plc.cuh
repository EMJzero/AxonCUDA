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

__forceinline__ __device__ uint32_t manhattan(coords c1, coords c2) {
    const uint32_t x_dst = c1.x > c2.x ? c1.x - c2.x : c2.x - c1.x;
    const uint32_t y_dst = c1.y > c2.y ? c1.y - c2.y : c2.y - c1.y;
    return x_dst + y_dst;
}


// USED BY: main

inline void printMatrixHex16(const uint32_t* matrix, dim_t width, dim_t height, dim_t maxRows, dim_t maxCols) {
    const dim_t rowsToPrint = std::min(height, maxRows);
    const dim_t colsToPrint = std::min(width, maxCols);
    for (dim_t y = 0; y < rowsToPrint; y++) {
        for (dim_t x = 0; x < colsToPrint; x++) {
            uint32_t value = matrix[y * width + x];
            uint16_t low16 = static_cast<uint16_t>(value & 0xFFFF);
            std::printf("%04X", low16);
            if (x + 1 < colsToPrint)
                std::printf(" ");
        }
        std::printf("\n");
    }
}
