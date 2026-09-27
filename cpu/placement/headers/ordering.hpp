#pragma once
#include <cstdint>
#include <vector>

#include "prims.hpp"
#include "utils_plc.hpp"
#include "data_types.hpp"
#include "data_types_plc.hpp"
#include "defines_plc.hpp"

namespace config_plc {
    struct runconfig;
}

using namespace config_plc;


// STEPS

buffer<uint32_t> locality_ordering(
    const runconfig &cfg,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t num_hedges,
    const dim_t hedges_size,
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const float* hedge_weights,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint64_t seed
);

void split_partitions_rand(
    const runconfig &cfg,
    uint32_t* partitions,
    uint32_t num_nodes,
    uint32_t num_parts,
    uint32_t batch_size,
    std::vector<xorwow_generator> &gens
);


// KERNELS

// NOTE: every kernel below runs a whole batch of multi-starts at once
// => per-multi-start arrays are one flat allocation of "batch_size" equally-sized segments, multi-start "b" owning [b*size, (b+1)*size)
// => partition ids are composite, "b*num_parts + p", which survives both the "*2" of a bisection and the ">>1" of a fold
// => node idxs are batch-flat, hypergraph pin idxs are not

void split_partitions_kernel(
    const uint32_t* __restrict__ part_offsets,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    uint32_t* __restrict__ partitions
);

void flag_cutnet_events_kernel(
    const uint32_t* __restrict__ part_pins,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    const uint32_t batch_size,
    const dim_t hedges_size,
    uint32_t* __restrict__ flags
);

void cutnet_event_generation_kernel(
    const uint32_t* __restrict__ part_pins,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ flags,
    const uint32_t num_hedges,
    const uint32_t batch_size,
    const dim_t hedges_size,
    float* __restrict__ event_weight,
    uint32_t* __restrict__ event_part
);

void label_propagation_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    bool* __restrict__ moves,
    float* __restrict__ scores
);

void label_move_events_kernel(
    const bool* __restrict__ moves,
    const float* __restrict__ scores,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    uint32_t* __restrict__ even_ev_partition,
    float* __restrict__ even_ev_score,
    uint32_t* __restrict__ even_ev_node,
    uint32_t* __restrict__ odd_ev_partition,
    float* __restrict__ odd_ev_score,
    uint32_t* __restrict__ odd_ev_node
);

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
    const uint32_t num_nodes,
    const uint32_t batch_size,
    float* __restrict__ even_event_score
);

void apply_move_events_kernel(
    const uint32_t* __restrict__ apply_up_to,
    const uint32_t* __restrict__ even_event_part,
    const uint32_t* __restrict__ even_event_node,
    const uint32_t* __restrict__ part_even_event_offsets,
    const uint32_t* __restrict__ part_odd_event_offsets,
    const uint32_t* __restrict__ odd_event_node,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    uint32_t* __restrict__ partitions
);

void update_best_partitions_kernel(
    const uint32_t* __restrict__ partitions,
    const float* __restrict__ cutnet,
    const float* __restrict__ last_best_cutnet,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    uint32_t* __restrict__ last_best_partitions
);

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
    const uint32_t batch_size,
    float* __restrict__ slot_scores
);

void flag_reversals_kernel(
    const float* __restrict__ sibling_score,
    const uint32_t num_parts,
    const uint32_t batch_size,
    bool* __restrict__ reverse
);

void labelprop_activity_kernel(
    const uint32_t* __restrict__ apply_up_to,
    const uint32_t num_part_pairs,
    const uint32_t batch_size,
    uint8_t* __restrict__ active
);

void apply_reversals_kernel(
    const uint32_t* __restrict__ segment,
    const uint32_t* __restrict__ offsets,
    const bool* __restrict__ flag,
    const uint32_t size,
    uint32_t* __restrict__ data
);

void measure_sequence_locality_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ order_idx,
    const uint32_t num_hedges,
    float* __restrict__ hedge_span
);
