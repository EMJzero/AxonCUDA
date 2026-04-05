#pragma once
#include <tuple>
#include <cstdint>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "data_types.cuh"

namespace config {
    struct runconfig;
}

using namespace config;

namespace cg = cooperative_groups;


// USED BY: candidates kernel

#define HIST_SIZE 512u // must be a multiple of WARP_SIZE (for the histogram max reduction)

#define DETERMINISTIC_SCORE_NOISE 64u // 256u // => adds a +[0, DETERMINISTIC_SCORE_NOISE - 1]/FIXED_POINT_SCALE symmetric noise while calculating pairing scores; set to 0 to disable; keep it a power of 2 otherwise


// USED BY: grouping kernel

// #define MAX_GROUP_SIZE 1u // => MAX_GROUP_SIZE - 1 slots per node; 2 means pairs [IDEA NOT WORTH IT]
#define PATH_SIZE 224u // initial slots for nodes to see while traversing the pairs tree, TODO: automatically extend if needed (costly...)
#define MAX_MATCHING_REPEATS 64u // maximum number of nodes a single thread can handle, must be less than 32 (due to using one-hot anti-repeat encoding)


// STEPS

void candidatesProposal(
    const runconfig &cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_srcs_count,
    const uint32_t *d_neighbors,
    const dim_t *d_neighbors_offsets,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t *d_inbound_count,
    const float *d_hedge_weights,
    const uint32_t *d_nodes_sizes,
    const uint32_t curr_num_nodes,
    uint32_t *d_pairs,
    uint32_t *d_u_scores
);

void logCandidates(
    const runconfig &cfg,
    const uint32_t *d_pairs,
    const uint32_t *d_u_scores,
    const uint32_t curr_num_nodes
);

std::tuple<uint32_t, uint32_t*, uint32_t*, uint32_t*, dim_t*> groupNodes(
    const runconfig &cfg,
    const cudaDeviceProp props,
    const uint32_t *d_inbound_count,
    const uint32_t *d_pairs,
    const uint32_t *d_u_scores,
    const uint32_t *d_nodes_sizes,
    const uint32_t curr_num_nodes,
    const uint32_t max_nodes_per_part,
    const uint32_t max_inbound_per_part,
    slot *d_slots,
    dp_score *d_dp_scores
);

uint32_t greedyMergeGroups(
    const runconfig &cfg,
    const uint32_t *d_nodes_sizes,
    const uint32_t *d_inbound_count,
    const uint32_t *d_ungroups,
    const dim_t *d_ungroups_offsets,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    uint32_t *d_groups,
    uint32_t *d_groups_sizes
);

void logGroups(
    const runconfig &cfg,
    const uint32_t *d_pairs,
    const uint32_t *d_groups,
    const uint32_t *d_groups_sizes,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes,
    const uint32_t h_max_nodes_per_part
);


// KERNELS

__global__
void candidates_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    uint32_t* __restrict__ pairs,
    uint32_t* __restrict__ scores
);

__global__
void grouping_kernel(
    const uint32_t* __restrict__ pairs,
    const uint32_t* __restrict__ scores,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_nodes,
    const uint32_t num_repeats,
    const uint32_t candidates_count,
    slot* __restrict__ group_slots,
    dp_score* __restrict__ d_dp_scores,
    uint32_t* __restrict__ groups
);
