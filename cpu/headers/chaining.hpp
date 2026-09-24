#pragma once
#include <cstdint>

#include "defines.hpp"
#include "data_types.hpp"

namespace config {
    struct runconfig;
}

using namespace config;


// USED BY: chaining

// NOTE: must match 'headers/chaining.cuh'
#define CHAIN_ITERS 4 // multi-iteration greedy chaining
#define CHAIN_WINDOW 256 // candidates scanned per node per iteration
#define CHAIN_ALPHA 1e-6f // node size penalty scale (adjust based on size magnitude)

#define CHAIN_MAX_STEPS 256 // maximum nodes explored to form a chain, increase if the typical chain length could exceeds it


// STEPS

void chaining(
    const runconfig &cfg,
    const uint32_t *srcs,
    const uint32_t *dsts,
    const uint32_t *size,
    const float *weight,
    const uint32_t num_edges,
    uint32_t *sequence_idx
);

void build_orphan_pairs(
    const runconfig &cfg,
    const uint32_t *nodes_sizes,
    const uint32_t *nodes_pins,
    const uint32_t *inbound_count,
    const uint32_t *pairs,
    const uint32_t curr_num_nodes,
    const uint32_t candidates_count,
    uint32_t* groups
);


// KERNELS

void propose_successor(
    const uint32_t num_edges,
    const uint32_t* dst,
    const uint32_t* size,
    const float* w,
    const int* out_begin,
    const int* out_end,
    const uint32_t* prev,
    const uint32_t* next,
    int window,
    float alpha,
    uint32_t* succ_choice,
    float* succ_score
);

void resolve_successor_conflicts(
    const uint32_t num_edges,
    const uint32_t* succ_choice,
    const float* succ_score,
    const uint32_t* prev,
    uint64_t* best_claim_for_succ
);

void commit_links(
    const uint32_t num_edges,
    uint32_t* next,
    uint32_t* prev,
    const uint64_t* best_claim_for_succ
);

void compute_comp_and_pos(
    const uint32_t num_edges,
    const uint32_t* prev,
    uint32_t* comp,
    uint32_t* pos
);

void pair_kth_smallest_with_kth_largest(
    const uint32_t* __restrict__ sorted_indices,
    const uint32_t num_free,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ nodes_pins,
    const uint32_t* __restrict__ inbound_count,
    uint32_t* __restrict__ groups
);


// HELPERS

// monotonic mapping of floats to unsigned integers (as in 'kernels/chaining.cu')
inline uint32_t float_to_ordered_uint(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(uint32_t));
    return bits ^ ((bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}
