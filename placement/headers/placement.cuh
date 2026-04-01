#pragma once
#include <tuple>
#include <cstdint>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "data_types.cuh"
#include "data_types_plc.cuh"
#include "defines_plc.cuh"

namespace config_plc {
    struct runconfig;
}

using namespace config_plc;

namespace cg = cooperative_groups;

// DEVICE CONSTANTS:
extern __constant__ uint32_t max_width;
extern __constant__ uint32_t max_height;


// USED BY: exclusive swaps kernel

#define SWAPS_PATH_SIZE 224u // initial slots for places to see while traversing the swaps tree
#define MAX_SWAPS_MATCHING_REPEATS 64u // number of places that can be handled by the same thread in case of limited space for the cooperative kernel launch


// STEPS

void force_directed_refinement(
    const runconfig cfg,
    const cudaDeviceProp props,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const uint32_t* d_touching,
    const dim_t* d_touching_offsets,
    const float* d_hedge_weights,
    const uint32_t num_nodes,
    coords* d_placement,
    uint32_t* d_inv_placement
);

void logForces(
    const uint32_t *d_forces,
    const uint32_t num_nodes
);

void logTensions(
    const uint32_t *d_pairs,
    const uint32_t *d_scores,
    const uint32_t num_nodes
);

void logSwapPairs(
    const slot *d_swap_slots,
    const uint32_t *d_swap_flags,
    const uint32_t num_nodes
);

void logEvents(
    const swap *d_ev_swaps,
    const float *d_ev_scores,
    const uint32_t num_nodes,
    const  std::string flare
);


// KERNELS

__global__
void inverse_placement_kernel(
    const coords* placement,
    const uint32_t num_nodes,
    uint32_t* inv_placement
);

__global__
void forces_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const float* hedge_weights,
    const coords* placement,
    const uint32_t num_nodes,
    float* forces
);

__global__
void tensions_kernel(
    const coords* placement,
    const uint32_t* inv_placement,
    const float* forces,
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    uint32_t* pairs,
    uint32_t* scores
);

__global__
void exclusive_swaps_kernel(
    const uint32_t* pairs,
    const uint32_t* scores,
    const uint32_t num_nodes,
    const uint32_t num_repeats,
    const uint32_t candidates_count,
    slot* swap_slots,
    uint32_t* swap_flags
);

__global__
void swap_events_kernel(
    const slot* swap_slots,
    const uint32_t* swap_flags,
    const uint32_t num_nodes,
    swap* ev_swaps,
    float* ev_scores
);

__global__
void scatter_ranks_kernel(
    const swap* ev_swaps,
    const uint32_t num_events,
    uint32_t* nodes_rank
);

__global__
void cascade_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const float* hedge_weights,
    const coords* placement,
    const swap* ev_swaps,
    const uint32_t* nodes_rank,
    const uint32_t num_events,
    float* scores
);

__global__
void apply_swaps_kernel(
    const swap* ev_swaps,
    const uint32_t num_good_swaps,
    coords* placement,
    uint32_t* inv_placement
);
