#pragma once
#include <tuple>
#include <string>
#include <cstdint>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "topology.hpp"

#include "data_types.cuh"
#include "data_types_plc.cuh"
#include "defines_plc.cuh"

namespace config_plc {
    struct runconfig;
}

using namespace config_plc;
using namespace topology;

namespace cg = cooperative_groups;

// DEVICE CONSTANTS:
template<Topology T>
extern __constant__ T c_topo;


// USED BY: exclusive swaps kernel

#define SWAPS_PATH_SIZE 4096u // initial slots for places to see while traversing the swaps tree
#define MAX_SWAPS_MATCHING_REPEATS 64u // number of places that can be handled by the same thread in case of limited space for the cooperative kernel launch (must be <= 64)


// USED BY: locality metrics estimation

#define REG_PINS_CAPACITY 64u // maximum number of pins per hedge that can fit in registers (32+1 bit each) -> use in-register MST algorithm


// STEPS

template<Topology T>
void forceDirectedRefinement(
    const runconfig &cfg,
    const cudaDeviceProp props,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const uint32_t* d_touching,
    const dim_t* d_touching_offsets,
    const float* d_hedge_weights,
    const uint32_t num_nodes,
    Coord_t<T>* d_placement,
    uint32_t* d_inv_placement,
    const cudaStream_t stream,
    const int tid
);

template<Topology T>
std::tuple<float, float> getLocalityMetrics(
    const runconfig &cfg,
    const Coord_t<T>* d_placement,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const uint32_t* d_srcs_count,
    const float* d_hedge_weights,
    const uint32_t num_hedges,
    const cudaStream_t stream,
    const int tid
);

template<Topology T>
void logForces(
    const float *d_forces,
    const uint32_t num_nodes,
    const cudaStream_t stream,
    const int tid
);

template<Topology T>
void logTensions(
    const runconfig &cfg,
    const uint32_t *d_pairs,
    const uint32_t *d_scores,
    const uint32_t num_nodes,
    const cudaStream_t stream,
    const int tid
);

template<Topology T>
void logSwapPairs(
    const slot *d_swap_slots,
    const uint32_t *d_swap_flags,
    const uint32_t num_nodes,
    const cudaStream_t stream,
    const int tid
);

void logEvents(
    const swap *d_ev_swaps,
    const float *d_ev_scores,
    const uint32_t num_nodes,
    const std::string flare,
    const cudaStream_t stream,
    const int tid
);


// KERNELS

template<Topology T>
__global__
void inverse_placement_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t num_nodes,
    uint32_t* __restrict__ inv_placement
);

template<Topology T>
__global__
void forces_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const Coord_t<T>* __restrict__ placement,
    const uint32_t num_nodes,
    float* __restrict__ forces
);

template<Topology T>
__global__
void tensions_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ inv_placement,
    const float* __restrict__ forces,
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    uint32_t* __restrict__ pairs,
    uint32_t* __restrict__ scores
);

template<Topology T>
__global__
void exclusive_swaps_kernel(
    const uint32_t* __restrict__ pairs,
    const uint32_t* __restrict__ scores,
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    slot* __restrict__ swap_slots,
    uint32_t* __restrict__ swap_flags
);

template<Topology T>
__global__
void swap_events_kernel(
    const slot* __restrict__ swap_slots,
    const uint32_t* __restrict__ swap_flags,
    const uint32_t num_nodes,
    swap* __restrict__ ev_swaps,
    float* __restrict__ ev_scores
);

template<Topology T>
__global__
void scatter_ranks_kernel(
    const swap* __restrict__ ev_swaps,
    const uint32_t num_events,
    uint32_t* __restrict__ nodes_rank
);

template<Topology T>
__global__
void resolve_empty_conflicts_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ inv_placement,
    const uint32_t* __restrict__ nodes_rank,
    const uint32_t num_events,
    swap* __restrict__ ev_swaps
);

template<Topology T>
__global__
void cascade_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const Coord_t<T>* __restrict__ placement,
    const swap* __restrict__ ev_swaps,
    const uint32_t* __restrict__ nodes_rank,
    const uint32_t num_events,
    float* __restrict__ scores
);

template<Topology T>
__global__
void apply_swaps_kernel(
    const swap* __restrict__ ev_swaps,
    const uint32_t num_good_swaps,
    Coord_t<T>* __restrict__ placement,
    uint32_t* __restrict__ inv_placement
);

template<Topology T>
__global__
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
__global__
void tot_src_dst_distance_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    float* __restrict__ result
);

template<Topology T>
__global__
void min_spanning_tree_weight_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    float* __restrict__ result
);
