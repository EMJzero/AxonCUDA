#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "utils.cuh"
#include "constants.cuh"

__global__
void neighborhoods_count_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t num_nodes,
    const dim_t max_neighbors,
    const bool discharge,
    uint32_t* neighbors,
    dim_t* neighbors_offsets
);

__global__
void neighborhoods_scatter_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t num_nodes,
    const dim_t* neighbors_offsets,
    uint32_t* neighbors
);

__global__
void apply_coarsening_hedges_count(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* srcs_count,
    const uint32_t* groups,
    const uint32_t num_hedges,
    const uint32_t max_hedge_size,
    uint32_t *coarse_oversized_hedges,
    dim_t* coarse_hedges_offsets,
    uint32_t* coarse_srcs_count
);

__global__
void apply_coarsening_hedges_scatter_dsts(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* srcs_count,
    const uint32_t* groups,
    const uint32_t num_hedges,
    const dim_t* coarse_hedges_offsets,
    uint32_t* coarse_hedges
);

__global__
void apply_coarsening_hedges_scatter_srcs(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* srcs_count,
    const uint32_t* groups,
    const uint32_t num_hedges,
    const dim_t* coarse_hedges_offsets,
    const uint32_t* coarse_srcs_count,
    uint32_t* coarse_hedges
);

__global__
void apply_coarsening_neighbors_count(
    const uint32_t* neighbors,
    const dim_t* neighbors_offsets,
    const uint32_t* groups,
    const uint32_t* ungroups,
    const dim_t* ungroups_offsets,
    const uint32_t num_groups,
    const dim_t max_neighbors,
    const bool discharge,
    uint32_t* oversized_coarse_neighbors,
    dim_t* coarse_neighbors_offsets
);

__global__
void apply_coarsening_neighbors_scatter(
    const uint32_t* neighbors,
    const dim_t* neighbors_offsets,
    const uint32_t* groups,
    const uint32_t* ungroups,
    const dim_t* ungroups_offsets,
    const uint32_t num_groups,
    const dim_t* coarse_neighbors_offsets,
    uint32_t* coarse_neighbors
);

__global__
void apply_coarsening_touching_count(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t num_hedges,
    dim_t* coarse_touching_offsets
) ;

__global__
void apply_coarsening_touching_scatter_inbound(
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* inbound_count,
    const uint32_t* ungroups,
    const dim_t* ungroups_offsets,
    const uint32_t num_groups,
    const dim_t* coarse_touching_offsets,
    uint32_t* coarse_touching,
    uint32_t* coarse_inbound_count
);

__global__
void apply_coarsening_touching_scatter_outbound(
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* inbound_count,
    const uint32_t* ungroups,
    const dim_t* ungroups_offsets,
    const uint32_t num_groups,
    const dim_t* coarse_touching_offsets,
    const uint32_t* coarse_inbound_count,
    uint32_t* coarse_touching
);

__global__
void apply_uncoarsening_partitions(
    const uint32_t* groups,
    const uint32_t* coarse_partitions,
    const uint32_t num_nodes,
    uint32_t* partitions
);

__global__
void pack_segments(
    const uint32_t* oversized,
    const dim_t* offsets,
    const uint32_t num_subs,
    const dim_t sub_size,
    uint32_t* out
);

__global__
void pack_segments_varsize(
    const uint32_t* oversized,
    const dim_t* oversized_offsets,
    const dim_t* offsets,
    const uint32_t num_subs,
    const dim_t base_sub_size,
    uint32_t* out
);