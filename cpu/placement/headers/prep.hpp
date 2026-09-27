#pragma once
#include <tuple>
#include <cstdint>

#include "prims.hpp"
#include "data_types.hpp"
#include "data_types_plc.hpp"
#include "defines_plc.hpp"

namespace config_plc {
    struct runconfig;
}

namespace hgraph {
    class HyperGraph;
}

using namespace config_plc;
using namespace hgraph;


// STEPS

std::tuple<buffer<uint32_t>, buffer<dim_t>> buildTouchingHost(
    const runconfig &cfg,
    const HyperGraph& hg
);

std::tuple<buffer<uint32_t>, buffer<dim_t>> buildTouching(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t num_nodes,
    const uint32_t num_hedges
);


// KERNELS

void touching_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    dim_t* __restrict__ touching_offsets
);

void touching_build_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_hedges,
    uint32_t* __restrict__ touching,
    uint32_t* __restrict__ inserted_count
);
