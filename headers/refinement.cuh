#pragma once
#include <cstdint>

#include <cuda_runtime.h>

#include "defines.cuh"
#include "data_types.cuh"

struct runconfig;


// USED BY: fm refinement kernel

#define PART_HIST_SIZE 64u // best if it is a multiple of WARP_SIZE, best if partitions_per_thread * WARP_SIZE <= num_partitions


// STEPS

void refinementRepeats(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_srcs_count,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t *d_inbound_count,
    const float *d_hedge_weights,
    const uint32_t *d_nodes_sizes,
    const uint32_t level_idx,
    const uint32_t curr_num_nodes,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    const dim_t touching_size,
    uint32_t *d_pairs,
    float *d_f_scores,
    uint32_t *d_partitions,
    uint32_t *d_partitions_sizes,
    uint32_t *d_pins_per_partitions,
    uint32_t *d_partitions_inbound_sizes
);

void logPartitions(
    const uint32_t *d_partitions,
    const uint32_t *d_partitions_sizes,
    const uint32_t curr_num_nodes,
    const uint32_t num_partitions,
    const uint32_t h_max_nodes_per_part
);


// KERNELS

__global__
void pins_per_partition_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* pins_per_partitions,
    uint32_t* partitions_inbound_sizes
);

__global__
void inbound_pins_per_partition_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* srcs_count,
    const uint32_t* partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* inbound_pins_per_partitions,
    uint32_t* partitions_inbound_sizes
);

__global__
void fm_refinement_gains_kernel(
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t* partitions,
    const uint32_t* pins_per_partitions,
    const uint32_t* nodes_sizes,
    const uint32_t* partitions_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t randomizer,
    const uint32_t discount,
    const bool encourage_all_moves,
    uint32_t* moves,
    float* scores
);

__global__
void fm_refinement_cascade_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t* move_ranks,
    const uint32_t* moves,
    const uint32_t* partitions,
    const uint32_t* pins_per_partitions,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const bool encourage_all_moves,
    float* scores
);

__global__
void fm_refinement_apply_kernel(
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* moves,
    const uint32_t* move_ranks,
    const uint32_t* nodes_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t num_good_moves,
    uint32_t* partitions,
    uint32_t* partitions_sizes
    //uint32_t* pins_per_partitions
);

__global__
void build_size_events_kernel(
    const uint32_t* moves,
    const uint32_t* ranks,
    const uint32_t* partitions,
    const uint32_t* nodes_sizes,
    const uint32_t num_nodes,
    uint32_t* ev_partition,
    uint32_t* ev_index,
    int32_t* ev_delta
);

__global__
void flag_size_events_kernel(
    const uint32_t* ev_partition,
    const uint32_t* ev_index,
    const int32_t* ev_delta,
    const uint32_t* partitions_sizes,
    const uint32_t num_events,
    int32_t* valid_moves
);

__global__
void build_hedge_events_kernel(
    const uint32_t* moves,
    const uint32_t* ranks,
    const uint32_t* partitions,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* inbound_count,
    const uint32_t num_nodes,
    uint32_t* ev_partition,
    uint32_t* ev_index,
    uint32_t* ev_hedge,
    int32_t* ev_delta
);

__global__
void count_inbound_size_events_kernel(
    const uint32_t* partitions_inbound_counts,
    const uint32_t* ev_partition,
    const uint32_t* ev_index,
    const uint32_t* ev_hedge,
    const int32_t* ev_delta,
    uint32_t num_events,
    uint32_t num_partitions,
    uint32_t* inbound_size_events_offsets
);

__global__
void build_inbound_size_events_kernel(
    const uint32_t* partitions_inbound_counts,
    const uint32_t* ev_partition,
    const uint32_t* ev_index,
    const uint32_t* ev_hedge,
    const int32_t* ev_delta,
    const uint32_t* inbound_size_events_offsets,
    uint32_t num_events,
    uint32_t num_partitions,
    uint32_t* new_ev_partition,
    uint32_t* new_ev_index,
    int32_t* new_ev_delta
);

__global__
void flag_inbound_events_kernel(
    const uint32_t* ev_partition,
    const uint32_t* ev_index,
    const int32_t* ev_delta,
    const uint32_t* partitions_inbound_sizes,
    const uint32_t num_events,
    int32_t* valid_moves
);
