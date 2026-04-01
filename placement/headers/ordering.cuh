#pragma once
#include <cstdint>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand.h>

#include "data_types.cuh"
#include "data_types_plc.cuh"
#include "defines_plc.cuh"

namespace cg = cooperative_groups;


// USED BY: recursive bipartitioning

#define LABELPROP_REPEATS 16


// STEPS

uint32_t* locality_ordering(
    const uint32_t num_nodes,
    const uint32_t num_hedges,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const float* d_hedge_weights,
    const uint32_t* d_touching,
    const dim_t* d_touching_offsets,
    const uint64_t seed
);

void split_partitions_rand(
    uint32_t* d_partitions,
    uint32_t num_nodes,
    uint32_t num_parts,
    curandGenerator_t gen
);


// KERNELS

__global__
void split_partitions_kernel(
    const uint32_t* __restrict__ part_offsets,
    const uint32_t num_nodes,
    uint32_t* __restrict__ partitions
);

__global__
void label_propagation_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_nodes,
    bool* __restrict__ moves,
    uint32_t* __restrict__ even_event_idx,
    uint32_t* __restrict__ odd_event_idx,
    float* __restrict__ scores
);

__global__
void label_move_events_kernel(
    const bool* __restrict__ moves,
    const float* __restrict__ scores,
    const uint32_t* __restrict__ even_ev_idx,
    const uint32_t* __restrict__ odd_ev_idx,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_nodes,
    uint32_t* __restrict__ even_ev_partition,
    float* __restrict__ even_ev_score,
    uint32_t* __restrict__ even_ev_node,
    uint32_t* __restrict__ odd_ev_partition,
    float* __restrict__ odd_ev_score,
    uint32_t* __restrict__ odd_ev_node
);

__global__
void label_cascade_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions,
    const uint32_t* __restrict__ part_even_event_offsets,
    const uint32_t* __restrict__ part_odd_event_offsets,
    const uint32_t* __restrict__ even_ranks,
    const uint32_t* __restrict__ odd_ranks,
    const uint32_t* __restrict__ even_event_node,
    const uint32_t* __restrict__ odd_event_node,
    const uint32_t even_events_count,
    const uint32_t odd_events_count,
    float* __restrict__ even_event_score
);

__global__
void apply_move_events_kernel(
    const uint32_t* __restrict__ apply_up_to,
    const uint32_t* __restrict__ even_event_part,
    const uint32_t* __restrict__ even_event_node,
    const uint32_t* __restrict__ part_even_event_offsets,
    const uint32_t* __restrict__ part_odd_event_offsets,
    const uint32_t* __restrict__ odd_event_node,
    const uint32_t even_events_count,
    uint32_t* __restrict__ partitions
);

__global__
void sibling_tree_connection_strength_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ order,
    const uint32_t* __restrict__ ord_part,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_nodes,
    float* __restrict__ scores
);

__global__
void flag_reversals_kernel(
    const float* __restrict__ sibling_score,
    const uint32_t num_parts,
    bool* __restrict__ reverse
);

__global__
void apply_reversals_kernel(
    const uint32_t* segment,
    const uint32_t* offsets,
    const bool* flag,
    const uint32_t size,
    uint32_t* data
);