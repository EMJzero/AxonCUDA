#include <tuple>
#include <iostream>

#include "hgraph.hpp"
#include "runconfig_plc.hpp"

#include "prep.hpp"
#include "utils.hpp"
#include "prims.hpp"
#include "defines_plc.hpp"

using namespace config_plc;

std::tuple<buffer<uint32_t>, buffer<dim_t>> buildTouchingHost(
    const runconfig &cfg,
    const HyperGraph& hg
) {
    ERR(cfg) std::cerr << "WARNING: building incidence sets sequentially will take a while...\n";

    // HP: hedges already internally deduplicated (acyclic), keeping the dst whenever a duplicate is between srcs and dsts
    const uint32_t num_nodes = hg.nodes();

    // prepare touching sets
    buffer<dim_t> touching_offsets((dim_t)num_nodes + 1); // touching_offsets[node idx] -> touching set start idx in touching
    touching_offsets[0] = 0;
    for (uint32_t n = 0; n < num_nodes; ++n)
        touching_offsets[n + 1] = touching_offsets[n] + (dim_t)std::ranges::distance(hg.inboundSortedIds(n)) + (dim_t)std::ranges::distance(hg.outboundSortedIds(n));

    buffer<uint32_t> touching(touching_offsets[num_nodes]); // contigous inbound+outbout sets array (first inbound, then outbound)
    for (uint32_t n = 0; n < num_nodes; ++n) {
        dim_t idx = touching_offsets[n];
        // NOTE: must put in inbounds first!
        for (uint32_t h : hg.inboundSortedIds(n))
            touching[idx++] = h;
        for (uint32_t h : hg.outboundSortedIds(n))
            touching[idx++] = h;
    }

    return std::make_tuple(std::move(touching), std::move(touching_offsets));
}

std::tuple<buffer<uint32_t>, buffer<dim_t>> buildTouching(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t num_nodes,
    const uint32_t num_hedges
) {
    // HP: hedges already internally deduplicated (acyclic), keeping the dst whenever a duplicate is between srcs and dsts
    buffer<dim_t> touching_offsets((dim_t)num_nodes + 1); // touching_offsets[node idx] -> touching set start idx in touching
    par_fill<dim_t>(touching_offsets.data(), (dim_t)num_nodes + 1, 0ull); // remember to leave the first offset at 0

    LAUNCH(cfg) RUN << "touching count kernel (threads=" << cfg.threads << ") ...\n";
    touching_count_kernel(
        hedges,
        hedges_offsets,
        num_hedges,
        touching_offsets.data()
    );

    par_inclusive_scan<dim_t>(touching_offsets.data(), (dim_t)num_nodes + 1); // the first entry is zero, making this an exclusive scan of the counts
    const dim_t touching_size = touching_offsets[num_nodes]; // total number of touching hedges among all sets

    buffer<uint32_t> touching(touching_size); // contigous touching sets array
    buffer<uint32_t> inserted_count(num_nodes); // inserted_count[node idx] -> hedges already inserted in the node's touching set
    par_fill<uint32_t>(inserted_count.data(), num_nodes, 0u);

    LAUNCH(cfg) RUN << "touching build kernel (threads=" << cfg.threads << ") ...\n";
    touching_build_kernel(
        hedges,
        hedges_offsets,
        touching_offsets.data(),
        num_hedges,
        touching.data(),
        inserted_count.data()
    );
    inserted_count.release();

    // each node's hedges were claimed by racing atomics, so sort every set to pin down its order
    // => that order decides which lane visits which pin later on, and with it the summation order of every touching-based float reduction
    const dim_t* offsets = touching_offsets.data();
    par_segmented_sort<uint32_t>(touching.data(), num_nodes, [=](uint32_t node) { return offsets[node]; });

    return std::make_tuple(std::move(touching), std::move(touching_offsets));
}
