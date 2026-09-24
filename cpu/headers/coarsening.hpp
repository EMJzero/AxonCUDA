#pragma once
#include <tuple>
#include <cstdint>

#include "data_types.hpp"
#include "prims.hpp"

namespace config {
    struct runconfig;
}

using namespace config;


// USED BY: candidates kernel

// NOTE: must match 'headers/coarsening.cuh'
#define DETERMINISTIC_SCORE_NOISE 64u // 256u // => adds a +[0, DETERMINISTIC_SCORE_NOISE - 1]/FIXED_POINT_SCALE symmetric noise while calculating pairing scores; set to 0 to disable; keep it a power of 2 otherwise


// STEPS

void candidatesProposal(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t *srcs_count,
    const uint32_t *neighbors,
    const dim_t *neighbors_offsets,
    const uint32_t *touching,
    const dim_t *touching_offsets,
    const uint32_t *inbound_count,
    const float *hedge_weights,
    const uint32_t *nodes_sizes,
    const uint32_t *nodes_pins,
    const uint32_t curr_num_nodes,
    uint32_t *pairs,
    uint32_t *u_scores
);

void logCandidates(
    const runconfig &cfg,
    const uint32_t *pairs,
    const uint32_t *u_scores,
    const uint32_t curr_num_nodes
);

std::tuple<uint32_t, buffer<uint32_t>, buffer<uint32_t>, buffer<uint32_t>, buffer<uint32_t>, buffer<dim_t>> groupNodes(
    const runconfig &cfg,
    const uint32_t *inbound_count,
    const uint32_t *pairs,
    const uint32_t *u_scores,
    const uint32_t *nodes_sizes,
    const uint32_t *nodes_pins,
    const uint32_t curr_num_nodes,
    slot *slots
);

void logGroups(
    const runconfig &cfg,
    const uint32_t *pairs,
    const uint32_t *groups,
    const uint32_t *groups_sizes,
    const uint32_t *groups_pins,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes
);


// KERNELS

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
    const uint32_t* __restrict__ nodes_pins,
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    uint32_t* __restrict__ pairs,
    uint32_t* __restrict__ scores
);

void grouping_kernel(
    const uint32_t* __restrict__ pairs,
    const uint32_t* __restrict__ scores,
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    const bool exact,
    slot* __restrict__ group_slots,
    uint32_t* __restrict__ groups
);
