#pragma once
#include <tuple>
#include <cstdint>

#include "data_types.hpp"

namespace config {
    struct runconfig;
}

using namespace config;


// STEPS

uint32_t greedyMergeGroups(
    const runconfig &cfg,
    const uint32_t *nodes_sizes,
    const uint32_t *nodes_pins,
    const uint32_t *inbound_count,
    const uint32_t *ungroups,
    const dim_t *ungroups_offsets,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes,
    uint32_t *groups,
    uint32_t *groups_sizes,
    uint32_t *groups_pins
);

void mergeSmallPartitions(
    const runconfig &cfg,
    const uint32_t *partitions_sizes,
    const uint32_t *partitions_inbound_sizes,
    const uint32_t *partitions_pins,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    uint32_t *partitions
);

uint32_t zeroBaseIds(
    const uint32_t num_items,
    const uint32_t num_ids,
    uint32_t *ids
);
