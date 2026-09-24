#pragma once
#include <tuple>
#include <cstdint>

#include "data_types.hpp"
#include "prims.hpp"

namespace config {
    struct runconfig;
}

namespace hgraph {
    class HyperGraph;
}

using namespace config;
using namespace hgraph;


// STEPS

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>, buffer<uint32_t>> buildTouchingHost(
    const runconfig &cfg,
    const HyperGraph& hg
);

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>, buffer<uint32_t>> buildTouching(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t *srcs_count,
    const uint32_t num_nodes,
    const uint32_t num_hedges
);

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>> buildNeighbors(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t *touching,
    const dim_t *touching_offsets,
    const uint32_t num_nodes
);

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>> coarsenNeighbors(
    const runconfig &cfg,
    const uint32_t *neighbors,
    const dim_t *neighbors_offsets,
    const uint32_t *groups,
    const uint32_t *ungroups,
    const dim_t *ungroups_offsets,
    const uint32_t new_num_nodes
);

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>, buffer<uint32_t>> coarsenHedges(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t *srcs_count,
    const uint32_t *groups,
    const uint32_t num_hedges,
    const uint32_t new_num_nodes
);

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>, buffer<uint32_t>> coarsenTouching(
    const runconfig &cfg,
    const uint32_t *touching,
    const dim_t *touching_offsets,
    const uint32_t *inbound_count,
    const uint32_t *ungroups,
    const dim_t *ungroups_offsets,
    const uint32_t new_num_nodes,
    const uint32_t num_hedges
);


// KERNELS

void touching_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t num_hedges,
    dim_t* __restrict__ touching_offsets,
    uint32_t* __restrict__ inbound_count
);

void touching_build_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_hedges,
    uint32_t* __restrict__ touching,
    uint32_t* __restrict__ inserted_inbound,
    uint32_t* __restrict__ inserted_outbound
);

void touching_sort_kernel(
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t num_nodes,
    uint32_t* __restrict__ touching
);

void neighborhoods_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    csr_builder &neighbors
);

void apply_coarsening_neighbors_kernel(
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ groups,
    const uint32_t* __restrict__ ungroups,
    const dim_t* __restrict__ ungroups_offsets,
    const uint32_t num_groups,
    csr_builder &coarse_neighbors
);

void apply_coarsening_hedges_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ groups,
    const uint32_t num_hedges,
    const uint32_t num_groups,
    csr_builder &coarse_hedges,
    uint32_t* __restrict__ coarse_srcs_count
);

void apply_coarsening_touching_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t* __restrict__ ungroups,
    const dim_t* __restrict__ ungroups_offsets,
    const uint32_t num_groups,
    const uint32_t num_hedges,
    csr_builder &coarse_touching,
    uint32_t* __restrict__ coarse_inbound_count
);

void apply_uncoarsening_partitions(
    const uint32_t* __restrict__ groups,
    const uint32_t* __restrict__ coarse_partitions,
    const uint32_t num_nodes,
    uint32_t* __restrict__ partitions
);
