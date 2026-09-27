#pragma once
#include <tuple>
#include <string>
#include <cstdint>

#include "topology.hpp"

#include "data_types.hpp"
#include "data_types_plc.hpp"
#include "defines_plc.hpp"

namespace config_plc {
    struct runconfig;
}

using namespace config_plc;
using namespace topology;

// TOPOLOGY:
// NOTE: in CUDA this is a '__constant__' symbol, here a plain global, set once in main and then read everywhere
template<Topology T>
extern T topo;


// STEPS

template<Topology T>
void forceDirectedRefinement(
    const runconfig &cfg,
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    Coord_t<T>* placement,
    uint32_t* inv_placement
);

template<Topology T>
void getLocalityMetrics(
    const runconfig &cfg,
    const Coord_t<T>* placement,
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* srcs_count,
    const float* hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    float* src_dst_distance, // src_dst_distance[start] -> weighted avg. hedge max src-dst manhattan distance of that multi-start
    float* steiner_span // steiner_span[start] -> weighted avg. hedge Steiner tree span of that multi-start
);

template<Topology T>
void logForces(
    const float *forces,
    const uint32_t num_nodes
);

template<Topology T>
void logTensions(
    const runconfig &cfg,
    const uint32_t *pairs,
    const uint32_t *scores,
    const uint32_t num_nodes
);

template<Topology T>
void logSwapPairs(
    const slot *swap_slots,
    const uint32_t num_nodes
);

void logEvents(
    const swap *ev_swaps,
    const float *ev_scores,
    const uint32_t num_nodes,
    const std::string flare
);


// KERNELS

// NOTE: every kernel below runs a whole batch of multi-starts at once
// => per-multi-start arrays are one flat allocation of "batch_size" equally-sized segments, multi-start "b" owning [b*size, (b+1)*size)
// => node idxs stored in "pairs", "swap_slots", "ev_swaps" and "inv_placement" are batch-flat, hypergraph pin idxs are not
// => "active[b]" retires a converged multi-start

template<Topology T>
void inverse_placement_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    uint32_t* __restrict__ inv_placement
);

template<Topology T>
void forces_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const Coord_t<T>* __restrict__ placement,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    float* __restrict__ forces
);

template<Topology T>
void tensions_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ inv_placement,
    const float* __restrict__ forces,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    const uint8_t* __restrict__ active,
    const uint32_t candidates_count,
    uint32_t* __restrict__ pairs,
    uint32_t* __restrict__ scores
);

template<Topology T>
void exclusive_swaps_kernel(
    const uint32_t* __restrict__ pairs,
    const uint32_t* __restrict__ scores,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    const uint32_t candidates_count,
    slot* __restrict__ swap_slots
);

template<Topology T>
void swap_events_kernel(
    const slot* __restrict__ swap_slots,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    swap* __restrict__ ev_swaps,
    float* __restrict__ ev_scores
);

template<Topology T>
void scatter_ranks_kernel(
    const swap* __restrict__ ev_swaps,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    uint32_t* __restrict__ nodes_rank
);

template<Topology T>
void resolve_empty_conflicts_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ inv_placement,
    const uint32_t* __restrict__ nodes_rank,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    const uint8_t* __restrict__ active,
    swap* __restrict__ ev_swaps
);

template<Topology T>
void cascade_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const Coord_t<T>* __restrict__ placement,
    const swap* __restrict__ ev_swaps,
    const uint32_t* __restrict__ nodes_rank,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    float* __restrict__ scores
);

void prefix_gain_kernel(
    const float* __restrict__ ev_scores,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    uint32_t* __restrict__ num_good_swaps,
    uint8_t* __restrict__ active
);

template<Topology T>
void apply_swaps_kernel(
    const swap* __restrict__ ev_swaps,
    const uint32_t* __restrict__ num_good_swaps,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    const uint8_t* __restrict__ active,
    Coord_t<T>* __restrict__ placement,
    uint32_t* __restrict__ inv_placement
);

template<Topology T>
void max_src_dst_distance_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    float* __restrict__ result
);

template<Topology T>
void tot_src_dst_distance_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    float* __restrict__ result
);

template<Topology T>
void min_spanning_tree_weight_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    float* __restrict__ result
);
