#pragma once
#include <stdint.h>

// USED BY: everyone

#define WARP_SIZE 32u


// USED BY: candidates kernel

#define HIST_SIZE 64u // must be a multiple of WARP_SIZE (for the histogram max reduction)

typedef struct {
    uint32_t node;
    float score;
} bin;

__forceinline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__forceinline__ __device__ bin warpReduceMax(float val, uint32_t payload) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        uint32_t other_payload = __shfl_down_sync(0xffffffff, payload, offset);
        if (other_val > val) {
            val = other_val;
            payload = other_payload;
        }
    }
    return {.node = payload, .score = val};
}


// USED BY: grouping kernel

#define MAX_GROUP_SIZE 4 // => MAX_GROUP_SIZE - 1 slots per node
// TODO: determine this at runtime w.r.t. the mean and variance of the spike frequency!
#define FIXED_POINT_SCALE 1024 // used to convert scores to fixed point
#define PATH_SIZE 128u // initial slots for nodes to see while traversing the pairs three, TODO: automatically extend if needed (costly...)

typedef struct __align__(8) {
    uint32_t id; // lower 32 bits (Nvidia GPUs are little-endian)
    uint32_t score; // converted from float to fixed point! higher 32 bits
} slot;

__device__ __forceinline__ unsigned long long pack_slot(uint32_t score, uint32_t node) {
    // high 32 bits = score, low 32 bits = node
    return ( (unsigned long long)score << 32 ) | ( (unsigned long long)node  & 0xffffffffull );
}

__device__ __forceinline__ void unpack_slot(unsigned long long v, uint32_t &score, uint32_t &node) {
    score = (uint32_t)(v >> 32);
    node = (uint32_t)(v & 0xffffffffu);
}

__device__ __forceinline__ void atomic_max_on_slot(slot* s, uint32_t idx, uint32_t new_node, uint32_t new_score) {
    unsigned long long* s64 = reinterpret_cast<unsigned long long*>(s);
    unsigned long long new_val = pack_slot(new_score, new_node);
    atomicMax(&s64[idx], new_val);
}

__device__ __forceinline__ slot atomic_max_on_slot_ret(slot* s, uint32_t idx, uint32_t new_node, uint32_t new_score) {
    unsigned long long* s64 = reinterpret_cast<unsigned long long*>(s);
    unsigned long long new_val = pack_slot(new_score, new_node);
    slot ret;
    unpack_slot(atomicMax(&s64[idx], new_val), ret.score, ret.id);
    return ret;
}


// USED BY: coarsening routines

#define MAX_DEDUPE_BUFFER_SIZE 16384