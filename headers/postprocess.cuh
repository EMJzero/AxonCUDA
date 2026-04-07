#pragma once
#include <tuple>
#include <cstdint>

#include <cuda_runtime.h>

#include "data_types.cuh"

namespace config {
    struct runconfig;
}

using namespace config;


// STEPS

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

void mergeSmallPartitions(
    const runconfig &cfg,
    const uint32_t *d_partitions_sizes,
    const uint32_t *d_partitions_inbound_sizes,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    uint32_t *d_partitions
);

/*
// TODO: routine that iterates over hedges and uses atomics to compute the inbound set size of partitions at any time
// => migrate here the one at the end of the refinement paths?
void computeInboundSetSizes(
    const runconfig &cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_srcs_count,
    const uint32_t *d_partitions,
    const uint32_t *d_partitions_sizes,
    // ...
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    uint32_t *d_partitions_inbound_sizes
);
*/