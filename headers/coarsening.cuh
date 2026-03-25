#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "utils.cuh"
#include "constants.cuh"

namespace cg = cooperative_groups;

__global__
void candidates_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* srcs_count,
    const uint32_t* neighbors,
    const dim_t* neighbors_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* inbound_count,
    const float* hedge_weights,
    const uint32_t* nodes_sizes,
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    uint32_t* pairs,
    uint32_t* scores
);

__global__
void grouping_kernel(
    const uint32_t* pairs,
    const uint32_t* scores,
    const uint32_t* nodes_sizes,
    const uint32_t num_nodes,
    const uint32_t num_repeats,
    const uint32_t candidates_count,
    slot* group_slots,
    dp_score* d_dp_scores,
    uint32_t* groups
);