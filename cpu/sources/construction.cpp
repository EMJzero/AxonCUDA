#include <tuple>
#include <ranges>
#include <vector>
#include <iomanip>
#include <iostream>

#include "hgraph.hpp"
#include "runconfig.hpp"

#include "construction.hpp"

#include "utils.hpp"
#include "defines.hpp"

using namespace config;

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>, buffer<uint32_t>> buildTouchingHost(
    const runconfig &cfg,
    const HyperGraph &hg
) {
    ERR(cfg) std::cerr << "WARNING: building inbound and outbound sets sequentially will take a while...\n";

    // HP: hedges already internally deduplicated (acyclic), keeping the dst whenever a duplicate is between srcs and dsts
    const uint32_t num_nodes = hg.nodes();

    // prepare touching sets
    // HP: no duplicates in either set, eventually duplicates in outbound w.r.t. inbounds will also be lost,
    //     inbounds must come first and their part must be sorted by id (ascending)
    buffer<dim_t> touching_offsets((dim_t)num_nodes + 1);
    buffer<uint32_t> inbound_count(num_nodes);
    touching_offsets[0] = 0;
    for (uint32_t n = 0; n < num_nodes; ++n) {
        inbound_count[n] = (uint32_t)std::ranges::distance(hg.inboundSortedIds(n));
        touching_offsets[n + 1] = touching_offsets[n] + inbound_count[n] + (dim_t)std::ranges::distance(hg.outboundSortedIds(n));
    }
    const dim_t touching_size = touching_offsets[num_nodes];

    buffer<uint32_t> touching(touching_size);
    for (uint32_t n = 0; n < num_nodes; ++n) {
        dim_t idx = touching_offsets[n];
        // NOTE: must put in inbounds first!
        for (uint32_t h : hg.inboundSortedIds(n))
            touching[idx++] = h;
        for (uint32_t h : hg.outboundSortedIds(n))
            touching[idx++] = h;
    }

    return std::make_tuple(touching_size, std::move(touching), std::move(touching_offsets), std::move(inbound_count));
}

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>, buffer<uint32_t>> buildTouching(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t *srcs_count,
    const uint32_t num_nodes,
    const uint32_t num_hedges
) {
    // HP: hedges already internally deduplicated (acyclic), keeping the dst whenever a duplicate is between srcs and dsts
    buffer<dim_t> touching_offsets((dim_t)num_nodes + 1);
    buffer<uint32_t> inbound_count(num_nodes);
    par_fill<dim_t>(touching_offsets.data(), (dim_t)num_nodes + 1, 0ull); // remember to leave the first offset at 0
    par_fill<uint32_t>(inbound_count.data(), num_nodes, 0u);

    LAUNCH(cfg) RUN << "touching count kernel (threads=" << cfg.threads << ") ...\n";
    touching_count_kernel(
        hedges,
        hedges_offsets,
        srcs_count,
        num_hedges,
        touching_offsets.data(),
        inbound_count.data()
    );

    par_inclusive_scan<dim_t>(touching_offsets.data(), (dim_t)num_nodes + 1);
    const dim_t touching_size = touching_offsets[num_nodes]; // last value in the inclusive scan = full reduce = total number of touching hedges among all sets
    buffer<uint32_t> touching(touching_size);

    buffer<uint32_t> inserted_inbound(num_nodes);
    buffer<uint32_t> inserted_outbound(num_nodes);
    par_fill<uint32_t>(inserted_inbound.data(), num_nodes, 0u);
    par_copy<uint32_t>(inserted_outbound.data(), inbound_count.data(), num_nodes); // initialize to inbound_count (to spare an add in the kernel)

    LAUNCH(cfg) RUN << "touching build kernel (threads=" << cfg.threads << ") ...\n";
    touching_build_kernel(
        hedges,
        hedges_offsets,
        srcs_count,
        touching_offsets.data(),
        num_hedges,
        touching.data(),
        inserted_inbound.data(),
        inserted_outbound.data()
    );

    // sort each inbound (and outbound) touching set
    LAUNCH(cfg) RUN << "touching sort kernel (threads=" << cfg.threads << ") ...\n";
    touching_sort_kernel(
        touching_offsets.data(),
        inbound_count.data(),
        num_nodes,
        touching.data()
    );

    return std::make_tuple(touching_size, std::move(touching), std::move(touching_offsets), std::move(inbound_count));
}

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>> buildNeighbors(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t *touching,
    const dim_t *touching_offsets,
    const uint32_t num_nodes
) {
    // HP: no duplicates in neighbors, no one's own self among one's neighbors
    // NOTE: in CUDA VRAM forces sampling the largest neighborhood, then chunking the construction or rebuilding neighbors
    //       when short, here one path suffices: deduplicate, then pack
    csr_builder builder(num_nodes);

    LAUNCH(cfg) RUN << "neighborhoods kernel (threads=" << cfg.threads << ") ...\n";
    neighborhoods_kernel(
        hedges,
        hedges_offsets,
        touching,
        touching_offsets,
        num_nodes,
        builder
    );

    buffer<uint32_t> neighbors = builder.pack();
    const dim_t total_neighbors = builder.offsets[num_nodes];

    return std::make_tuple(total_neighbors, std::move(neighbors), std::move(builder.offsets));
}

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>> coarsenNeighbors(
    const runconfig &cfg,
    const uint32_t *neighbors,
    const dim_t *neighbors_offsets,
    const uint32_t *groups,
    const uint32_t *ungroups,
    const dim_t *ungroups_offsets,
    const uint32_t new_num_nodes
) {
    csr_builder builder(new_num_nodes);

    LAUNCH(cfg) RUN << "coarsening kernel (neighbors) (threads=" << cfg.threads << ") ...\n";
    apply_coarsening_neighbors_kernel(
        neighbors,
        neighbors_offsets,
        groups,
        ungroups,
        ungroups_offsets,
        new_num_nodes,
        builder
    );

    buffer<uint32_t> coarse_neighbors = builder.pack();
    const dim_t new_neighbors_size = builder.offsets[new_num_nodes];

    return std::make_tuple(new_neighbors_size, std::move(coarse_neighbors), std::move(builder.offsets));
}

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>, buffer<uint32_t>> coarsenHedges(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t *srcs_count,
    const uint32_t *groups,
    const uint32_t num_hedges,
    const uint32_t new_num_nodes
) {
    csr_builder builder(num_hedges); // NOTE: the number of hedges never decreases (for now), unlike that of nodes!
    buffer<uint32_t> coarse_srcs_count(num_hedges);

    LAUNCH(cfg) RUN << "coarsening kernel (hedges) (threads=" << cfg.threads << ") ...\n";
    apply_coarsening_hedges_kernel(
        hedges,
        hedges_offsets,
        srcs_count,
        groups,
        num_hedges,
        new_num_nodes,
        builder,
        coarse_srcs_count.data()
    );

    buffer<uint32_t> coarse_hedges = builder.pack();
    const dim_t new_hedges_size = builder.offsets[num_hedges];

    return std::make_tuple(new_hedges_size, std::move(coarse_hedges), std::move(builder.offsets), std::move(coarse_srcs_count));
}

std::tuple<dim_t, buffer<uint32_t>, buffer<dim_t>, buffer<uint32_t>> coarsenTouching(
    const runconfig &cfg,
    const uint32_t *touching,
    const dim_t *touching_offsets,
    const uint32_t *inbound_count,
    const uint32_t *ungroups,
    const dim_t *ungroups_offsets,
    const uint32_t new_num_nodes,
    const uint32_t num_hedges
) {
    csr_builder builder(new_num_nodes); // NOTE: the number nodes decreases!
    buffer<uint32_t> coarse_inbound_count(new_num_nodes);

    LAUNCH(cfg) RUN << "coarsening kernel (touching) (threads=" << cfg.threads << ") ...\n";
    apply_coarsening_touching_kernel(
        touching,
        touching_offsets,
        inbound_count,
        ungroups,
        ungroups_offsets,
        new_num_nodes,
        num_hedges,
        builder,
        coarse_inbound_count.data()
    );

    buffer<uint32_t> coarse_touching = builder.pack();
    const dim_t new_touching_size = builder.offsets[new_num_nodes];

    return std::make_tuple(new_touching_size, std::move(coarse_touching), std::move(builder.offsets), std::move(coarse_inbound_count));
}
