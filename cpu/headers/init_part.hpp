#pragma once
#include <tuple>
#include <cstdint>

#include "defines.hpp"
#include "data_types.hpp"
#include "prims.hpp"

namespace config {
    struct runconfig;
}

using namespace config;


// USED BY: initial part

// NOTE: must match 'headers/init_part.cuh'
#define MAX_OMP_THREADS 16


// STEPS

std::tuple<buffer<uint32_t>, buffer<uint32_t>> initial_partitioning_kahypar(
    const runconfig &cfg,
    const uint32_t num_nodes,
    const uint32_t num_hedges,
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const float* hedge_weights,
    const dim_t* touching_offsets,
    const dim_t hedges_size,
    const uint32_t* nodes_sizes,
    const uint32_t k,
    const float epsilon
);


// KERNELS

void armonic_degree_score_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    float* __restrict__ hedge_ratio
);

void prune_hedges_kernel(
    const float* __restrict__ hedge_weights,
    const float* __restrict__ hedge_ratio,
    const uint32_t num_hedges,
    const float threshold,
    const uint32_t seed,
    float* __restrict__ hedge_scaled_weights,
    uint8_t* __restrict__ keep
);
