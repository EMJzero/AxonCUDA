#pragma once
#include <cfloat>
#include <cstdio>
#include <cstdint>
#include <stdint.h>

#include <cuda_runtime.h>
#include <curand.h>

#include "utils.cuh"
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

// warp-cooperatively visits every pin across a node's 'touching_count' touching hyperedges, flattening pin
// iteration across hedge boundaries instead of giving each hedge the whole warp: this keeps every lane busy
// regardless of individual hedge size, as a hedge that runs out mid-stride simply lets its lanes continue
// straight into the next one, rather than everyone idling on small hedges or stalling on an oversized one
// -> calls 'fn(pin, hedge_weight)' once per visited pin
// NOTE: 'sm_hedge_idx/cum/weight' are per-warp scratch, WARP_SIZE entries each, exclusive to the calling warp
template<typename Fn>
__device__ __forceinline__
void warpForEachTouchingPin(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ my_touching,
    const uint32_t touching_count,
    const uint32_t lane_id,
    uint32_t* __restrict__ sm_hedge_idx,
    uint32_t* __restrict__ sm_hedge_cum,
    float* __restrict__ sm_hedge_weight,
    Fn&& fn
) {
    for (uint32_t group_start = 0; group_start < touching_count; group_start += WARP_SIZE) {
        const uint32_t group_size = min(WARP_SIZE, touching_count - group_start);

        // stage (up to) one hedge per lane, then prefix-sum their sizes to place them in a flat pin sequence
        uint32_t my_hedge_size = 0u;
        if (lane_id < group_size) {
            const uint32_t hedge_idx = my_touching[group_start + lane_id];
            sm_hedge_idx[lane_id] = hedge_idx;
            my_hedge_size = static_cast<uint32_t>(hedges_offsets[hedge_idx + 1] - hedges_offsets[hedge_idx]);
            sm_hedge_weight[lane_id] = hedge_weights[hedge_idx];
        }
        const uint32_t my_cum = warpExclusiveScan<uint32_t>(my_hedge_size); // my hedge's start position in the flat sequence
        if (lane_id < group_size)
            sm_hedge_cum[lane_id] = my_cum;
        const uint32_t group_pins = __shfl_sync(0xFFFFFFFFu, my_cum + my_hedge_size, group_size - 1);
        __syncwarp();

        // stride WARP_SIZE flat positions at a time over the group's pins, binary-searching which hedge each flat position belongs to
        for (uint32_t flat_pos = lane_id; flat_pos < group_pins; flat_pos += WARP_SIZE) {
            uint32_t lo = 0u, hi = group_size - 1u;
            while (lo < hi) {
                const uint32_t mid = (lo + hi + 1u) >> 1;
                if (sm_hedge_cum[mid] <= flat_pos) lo = mid; else hi = mid - 1u;
            }
            const uint32_t pin = hedges[hedges_offsets[sm_hedge_idx[lo]] + (flat_pos - sm_hedge_cum[lo])];
            fn(pin, sm_hedge_weight[lo]);
        }
        __syncwarp(); // every lane must be done reading this group's scratch before the next group overwrites it
    }
}


// USED BY: main
// nothing for now :)