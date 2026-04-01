#pragma once
#include <tuple>
#include <cstdint>

#include <cuda_runtime.h>

#include "data_types.cuh"
#include "data_types_plc.cuh"
#include "defines_plc.cuh"

namespace config_plc {
    struct runconfig;
}

namespace hgraph {
    class HyperGraph;
}

using namespace config_plc;
using namespace hgraph;


// STEPS

std::tuple<uint32_t*, dim_t*> buildTouchingHost(
    const HyperGraph& hg
);

std::tuple<uint32_t*, dim_t*> buildTouching(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t num_nodes,
    const uint32_t num_hedges
);


// KERNELS

__global__
void touching_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    dim_t* __restrict__ touching_offsets
);

__global__
void touching_build_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_hedges,
    uint32_t* __restrict__ touching,
    uint32_t* __restrict__ inserted_count
);