#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// TODO: upgrade "uint_32_t"s to "uint64_t"s to handle >4M nodes, >4M touching per node, >4M pins per hedge, >4M neighbors per node!

// USED BY: everyone

// absolute replacement for "size_t"
using dim_t = unsigned long long; // aka uint64_t


// USED BY: candidates kernel

template <typename T>
struct __align__(8) bin {
    uint32_t payload;
    T val;
};


// USED BY: grouping kernel

typedef struct __align__(8) {
    uint32_t id; // lower 32 bits (Nvidia GPUs are little-endian)
    uint32_t score; // converted from float to fixed point! higher 32 bits
} slot;

typedef struct __align__(8) {
    uint32_t with; // total score in the traversed subtree assuming the current node will be paired with its targed
    uint32_t wout; // total score in the traversed subtree assuming the current node will NOT be paired with its targed
} dp_score;


// USED BY: refinement constraints checks

// valid values filtering functor
struct masked_twin_value_functor {
    const float* value;
    const int32_t* valid_1;
    const int32_t* valid_2;
    __host__ __device__ float operator()(uint32_t i) const { return valid_1[i] == 0 && valid_2[i] == 0 ? value[i] : -FLT_MAX; }
};

// valid values filtering functor - one validity entry only
struct masked_value_functor {
    const float* value;
    const int32_t* valid;
    __host__ __device__ float operator()(uint32_t i) const { return valid[i] == 0 ? value[i] : -FLT_MAX; }
};

// custom comparison logic between size and inbound events
struct best_move_functor {
    const float* gain;
    const int32_t* valid_moves;
    const int32_t* inbound_valid_moves;
    __host__ __device__ bool operator()(uint32_t a, uint32_t b) const { // return true -> choose b
        // satisfying the inbound constraint is mandatory
        const bool a_inbound_ok = (inbound_valid_moves[a] == 0);
        const bool b_inbound_ok = (inbound_valid_moves[b] == 0);
        // if only one is valid, choose it
        if (a_inbound_ok != b_inbound_ok) return b_inbound_ok;
        // if neither is valid, keep the earlier one
        if (!a_inbound_ok && !b_inbound_ok) return a > b;
        // minimize size constraint violations
        const int32_t va = valid_moves[a];
        const int32_t vb = valid_moves[b];
        if (va != vb) return va > vb; // fewer violations wins
        // maximize gain
        const float sa = gain[a];
        const float sb = gain[b];
        if (sa != sb) return sa < sb;
        // tie, earlier index wins
        return a > b;
    }
};


// USED BY: pins per partition (sparse bitmap matrix)

#define BITMAP_CAPACITY 64u // number of bits/elements flagged by a bitmap instance
#define BITMAP_CAPLOG 6u // log_2(BITMASK_CAPACITY) -> how many bits are needed to index inside "flg"

struct bitmap {
    uint64_t cnt; // counter of how many entries exist before mines
    uint64_t flg; // i-th bit set to 1 if the cnt+i element is present
};


// USED BY: final small partitions merging

struct constraints_state {
    dim_t s, i; // size and inbound count
    uint32_t g; // group id
};


//  USED BY: hash-sets, hash-maps

typedef struct {
    uint32_t key;
    uint32_t value;
} hashmap_entry;