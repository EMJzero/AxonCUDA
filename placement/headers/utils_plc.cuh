#pragma once
#include <cfloat>
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

inline std::vector<coords> hilbertPlacement(uint32_t nodes, uint32_t width, uint32_t height) {
    if (nodes > width * height)
        throw std::runtime_error("Grid too small to hold all nodes.");
    auto sgn = [](int x) -> int { return (x > 0) - (x < 0); };
    std::vector<coords> result;
    result.reserve(nodes);
    // recursive generator
    auto generate = [&](auto&& self, int x, int y, int ax, int ay, int bx, int by) -> void {
        if (result.size() >= nodes) return;
        int w = std::abs(ax + ay);
        int h = std::abs(bx + by);

        int dax = sgn(ax);
        int day = sgn(ay);
        int dbx = sgn(bx);
        int dby = sgn(by);

        // trivial row fill
        if (h == 1) {
            for (int i = 0; i < w && result.size() < nodes; ++i) {
                result.push_back((coords){ x, y });
                x += dax;
                y += day;
            }
            return;
        }

        // trivial column fill
        if (w == 1) {
            for (int i = 0; i < h && result.size() < nodes; ++i) {
                result.push_back((coords){ x, y });
                x += dbx;
                y += dby;
            }
            return;
        }

        int ax2 = ax / 2;
        int ay2 = ay / 2;
        int bx2 = bx / 2;
        int by2 = by / 2;

        int w2 = std::abs(ax2 + ay2);
        int h2 = std::abs(bx2 + by2);

        if (2 * w > 3 * h) {
            if ((w2 & 1) && w > 2) {
                ax2 += dax;
                ay2 += day;
            }

            // long case
            self(self, x, y, ax2, ay2, bx, by);
            self(self, x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by);
        } else {
            if ((h2 & 1) && h > 2) {
                bx2 += dbx;
                by2 += dby;
            }

            // standard case
            self(self, x, y, bx2, by2, ax2, ay2);
            self(self, x + bx2, y + by2, ax, ay, bx - bx2, by - by2);
            self(self, x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby), -bx2, -by2, -(ax - ax2), -(ay - ay2));
        }
    };

    // Reduce width/height to smallest even values fitting nodes
    uint32_t cw = width  - ((width  % 2) ? 1 : 2);
    uint32_t ch = height - ((height % 2) ? 1 : 2);

    while (cw * ch >= nodes) {
        width = cw;
        height = ch;
        if (width > height) cw -= 2;
        else ch -= 2;
    }

    if (width >= height)
        generate(generate, 0, 0, (int)width, 0, 0, (int)height);
    else
        generate(generate, 0, 0, 0, (int)height, (int)width, 0);

    return result;
}

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
