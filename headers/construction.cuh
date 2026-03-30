#pragma once
#include <tuple>
#include <cstdint>

#include <cuda_runtime.h>

#include "data_types.cuh"

struct runconfig;

namespace hgraph {
    class HyperGraph;
}

using namespace hgraph;

// USED BY: neighborhoods kernel

#define SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE 8192u // 16384 is too big for an A100...
#define GM_MIN_BLOCK_DEDUPE_BUFFER_SIZE 256u


// USED BY: coarsening routines (all, touching, hedges, and neighbors)

#define MAX_SM_WARP_DEDUPE_BUFFER_SIZE 3072u // the A100 has 48KB of SM, this is (48KB/4B of uint32s)/4 warps per block
#define MIN_GM_WARP_DEDUPE_BUFFER_SIZE 256u // just for safety, interplays with 'MAX_HASH_PROBE_LENGTH' and 'OVERSIZED_SIZE_MULTIPLIER'


// STEPS

std::tuple<dim_t, uint32_t*, dim_t*, uint32_t*> buildTouchingHost(
    const HyperGraph& hg
);

std::tuple<dim_t, uint32_t*, dim_t*, uint32_t*> buildTouching(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_srcs_count,
    const uint32_t num_nodes,
    const uint32_t num_hedges
);

dim_t sampleMaxNeighborhoodSize(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t num_nodes,
    const uint32_t num_samples
);

std::tuple<dim_t, uint32_t*, dim_t*> buildNeighbors(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t num_nodes,
    const uint32_t max_neighbors,
    uint32_t *d_neighbors,
    dim_t *d_neighbors_offsets
);

std::tuple<dim_t, uint32_t*, dim_t*> coarsenNeighbors(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t *d_groups,
    const uint32_t *d_ungroups,
    const dim_t *d_ungroups_offsets,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes,
    const dim_t neighbors_size,
    uint32_t *d_neighbors,
    dim_t *d_neighbors_offsets
);

std::tuple<dim_t, uint32_t*, dim_t*, uint32_t*> coarsenHedges(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_srcs_count,
    const uint32_t *d_groups,
    const uint32_t num_hedges,
    const dim_t hedges_size
);

std::tuple<dim_t, uint32_t*, dim_t*, uint32_t*> coarsenTouching(
    const runconfig cfg,
    const uint32_t *d_coarse_hedges,
    const dim_t *d_coarse_hedges_offsets,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t *d_inbound_count,
    const uint32_t *d_ungroups,
    const dim_t *d_ungroups_offsets,
    const uint32_t new_num_nodes,
    const uint32_t num_hedges
);


// KERNELS

__global__
void touching_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t num_hedges,
    dim_t* __restrict__ touching_offsets,
    uint32_t* __restrict__ inbound_count
);

__global__
void touching_build_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_hedges,
    uint32_t* __restrict__ touching,
    uint32_t* __restrict__ inserted_inbound,
    uint32_t* __restrict__ inserted_outbound
);

__global__
void neighbors_sample_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    const uint32_t num_samples,
    const uint32_t samples_per_repeat,
    const uint32_t curr_repeat,
    uint32_t* __restrict__ flags_bits,
    dim_t* __restrict__ neighbors_count
);

__global__
void neighborhoods_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    const dim_t max_neighbors,
    const bool discharge,
    uint32_t* __restrict__ neighbors,
    dim_t* __restrict__ neighbors_offsets
);

__global__
void neighborhoods_scatter_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    const dim_t* __restrict__ neighbors_offsets,
    uint32_t* __restrict__ neighbors
);

__global__
void apply_coarsening_hedges_count(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ groups,
    const uint32_t num_hedges,
    uint32_t *coarse_oversized_hedges,
    dim_t* __restrict__ coarse_hedges_offsets,
    uint32_t* __restrict__ coarse_srcs_count
);

__global__
void apply_coarsening_hedges_scatter_dsts(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ groups,
    const uint32_t num_hedges,
    const dim_t* __restrict__ coarse_hedges_offsets,
    uint32_t* __restrict__ coarse_hedges
);

__global__
void apply_coarsening_hedges_scatter_srcs(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ groups,
    const uint32_t num_hedges,
    const dim_t* __restrict__ coarse_hedges_offsets,
    const uint32_t* __restrict__ coarse_srcs_count,
    uint32_t* __restrict__ coarse_hedges
);

__global__
void sum_of_grouped_neighbors_count(
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ ungroups,
    const dim_t* __restrict__ ungroups_offsets,
    const uint32_t num_groups,
    dim_t* __restrict__ grouped_neighbors_offsets
);

__global__
void apply_coarsening_neighbors_count(
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ groups,
    const uint32_t* __restrict__ ungroups,
    const dim_t* __restrict__ ungroups_offsets,
    const dim_t* __restrict__ coarse_oversized_neighbors_offsets,
    const uint32_t num_groups,
    uint32_t* __restrict__ oversized_coarse_neighbors,
    dim_t* __restrict__ coarse_neighbors_offsets
);

__global__
void rebuild_coarsening_neighbors_count(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ groups,
    const uint32_t* __restrict__ ungroups,
    const dim_t* __restrict__ ungroups_offsets,
    const uint32_t num_groups,
    const bool discharge,
    const dim_t* __restrict__ coarse_oversized_neighbors_offsets,
    uint32_t* __restrict__ coarse_oversized_neighbors,
    dim_t* __restrict__ coarse_neighbors_offsets
);

__global__
void rebuild_coarsening_neighbors_scatter(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ groups,
    const uint32_t* __restrict__ ungroups,
    const dim_t* __restrict__ ungroups_offsets,
    const dim_t* __restrict__ coarse_neighbors_offsets,
    const uint32_t num_groups,
    uint32_t* __restrict__ coarse_neighbors
);

__global__
void apply_coarsening_touching_count(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    dim_t* __restrict__ coarse_touching_offsets
) ;

__global__
void apply_coarsening_touching_scatter_inbound(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t* __restrict__ ungroups,
    const dim_t* __restrict__ ungroups_offsets,
    const uint32_t num_groups,
    const dim_t* __restrict__ coarse_touching_offsets,
    uint32_t* __restrict__ coarse_touching,
    uint32_t* __restrict__ coarse_inbound_count
);

__global__
void apply_coarsening_touching_scatter_outbound(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t* __restrict__ ungroups,
    const dim_t* __restrict__ ungroups_offsets,
    const uint32_t num_groups,
    const dim_t* __restrict__ coarse_touching_offsets,
    const uint32_t* __restrict__ coarse_inbound_count,
    uint32_t* __restrict__ coarse_touching
);

__global__
void apply_uncoarsening_partitions(
    const uint32_t* __restrict__ groups,
    const uint32_t* __restrict__ coarse_partitions,
    const uint32_t num_nodes,
    uint32_t* __restrict__ partitions
);

__global__
void pack_segments(
    const uint32_t* __restrict__ oversized,
    const dim_t* __restrict__ offsets,
    const uint32_t num_subs,
    const dim_t sub_size,
    uint32_t* __restrict__ out
);

__global__
void pack_segments_varsize(
    const uint32_t* __restrict__ oversized,
    const dim_t* __restrict__ oversized_offsets,
    const dim_t* __restrict__ offsets,
    const uint32_t num_subs,
    const dim_t base_sub_size,
    uint32_t* __restrict__ out
);
