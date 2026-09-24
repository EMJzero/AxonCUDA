#include <vector>
#include <algorithm>
#include <iostream>

#include "runconfig.hpp"

#include "chaining.hpp"

#include "utils.hpp"
#include "prims.hpp"


// given a set of src->dst pairs, each with a size and a weight, try to construct
// subsequences of pairs with similar size and highest total weight such that each's dst is
// the src of the next pair, stopping upon forming a cycle. The concatenation of subsequences
// by descending weight is then the final sequence returned.
// => deterministic tie-breaking is always based on the pair's idx (aka node/move idx)
void chaining(
    const runconfig &cfg,
    const uint32_t* srcs_og,
    const uint32_t* dsts_og,
    const uint32_t* size_og, // node sizes
    const float* weights_og,
    const uint32_t num_edges,
    uint32_t* sequence_idx
) {
    if (num_edges == 0) return;

    LAUNCH(cfg) RUN << "chaining kernels (threads=" << cfg.threads << ") ...\n";

    // sort edges by (src, -weight), stable w.r.t. the original idx
    // NOTE: in CUDA this is a comparison sort, where -0 and +0 are equal, hence "+ 0.0f" folds -0 into +0 before making radix keys
    const uint32_t max_src = par_reduce<uint32_t>(num_edges, 0u, [=](dim_t i) { return srcs_og[i]; }, [](uint32_t a, uint32_t b) { return a > b ? a : b; });
    buffer<uint32_t> orig = par_sort_permutation(num_edges, 32u + bits_for(max_src), [=](dim_t i) {
        return ((uint64_t)srcs_og[i] << 32) | (uint64_t)float_to_ordered_uint(-weights_og[i] + 0.0f);
    }); // orig[idx] -> original idx of the idx-th pair after reordering
    buffer<uint32_t> srcs(num_edges); // srcs[idx] -> source partition / source node of the idx-th pair
    buffer<uint32_t> dsts(num_edges); // dsts[idx] -> destination partition / destination node of the idx-th pair
    buffer<uint32_t> size(num_edges); // size[idx] -> size of the idx-th pair's moving node
    buffer<float> w(num_edges); // w[idx] -> weight / gain of the idx-th pair
    par_gather(orig.data(), num_edges, srcs_og, srcs.data());
    par_gather(orig.data(), num_edges, dsts_og, dsts.data());
    par_gather(orig.data(), num_edges, size_og, size.data());
    par_gather(orig.data(), num_edges, weights_og, w.data());

    // for each edge i, its outgoing candidate list is OUT[dst[i]] = edges whose src == dst[i]
    buffer<int> out_begin(num_edges); // out_begin[idx] -> begin of the outgoing candidates range of the idx-th pair
    buffer<int> out_end(num_edges); // out_end[idx] -> end of the outgoing candidates range of the idx-th pair
    #pragma omp parallel for schedule(static) if(num_edges > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < num_edges; i++) {
        out_begin[i] = (int)(std::lower_bound(srcs.data(), srcs.data() + num_edges, dsts[i]) - srcs.data());
        out_end[i] = (int)(std::upper_bound(srcs.data(), srcs.data() + num_edges, dsts[i]) - srcs.data());
    }

    // chaining state
    buffer<uint32_t> next(num_edges); // next[idx] -> chosen successor pair of the idx-th pair
    buffer<uint32_t> prev(num_edges); // prev[idx] -> chosen predecessor pair of the idx-th pair
    buffer<uint32_t> succ_choice(num_edges); // succ_choice[idx] -> proposed successor of the idx-th pair
    buffer<float> succ_score(num_edges); // succ_score[idx] -> score of the proposed successor of the idx-th pair
    buffer<uint64_t> best_claim(num_edges); // best_claim[idx] -> packed best predecessor claim received by successor idx
    par_fill<uint32_t>(next.data(), num_edges, UINT32_MAX); // UINT32_MAX -> no successor chosen yet
    par_fill<uint32_t>(prev.data(), num_edges, UINT32_MAX); // UINT32_MAX -> no predecessor chosen yet

    // multi-iteration greedy build
    for (int it = 0; it < CHAIN_ITERS; ++it) {
        par_fill<uint64_t>(best_claim.data(), num_edges, 0ull); // 0 -> no predecessor claimed this successor yet

        propose_successor(
            num_edges,
            dsts.data(),
            size.data(),
            w.data(),
            out_begin.data(),
            out_end.data(),
            prev.data(),
            next.data(),
            CHAIN_WINDOW,
            CHAIN_ALPHA,
            succ_choice.data(),
            succ_score.data()
        );

        resolve_successor_conflicts(
            num_edges,
            succ_choice.data(),
            succ_score.data(),
            prev.data(),
            best_claim.data()
        );

        commit_links(
            num_edges,
            next.data(),
            prev.data(),
            best_claim.data()
        );
    }

    // component id and position
    buffer<uint32_t> comp(num_edges); // comp[idx] -> representative component id of the idx-th pair
    buffer<uint32_t> pos(num_edges); // pos[idx] -> position of the idx-th pair inside its chain
    compute_comp_and_pos(
        num_edges,
        prev.data(),
        comp.data(),
        pos.data()
    );

    // sort pairs by component (stable), then reduce their weights per component
    buffer<uint32_t> by_comp = par_sort_permutation(num_edges, bits_for(num_edges), [&](dim_t i) { return (uint64_t)comp[i]; }); // by_comp[idx] -> pair idx sorted by component
    buffer<uint32_t> comp_heads = par_copy_if(num_edges, [&](uint32_t i) { return i == 0 || comp[by_comp[i]] != comp[by_comp[i - 1]]; }); // comp_heads[c] -> first sorted position of the c-th unique component
    const uint32_t num_components = (uint32_t)comp_heads.size();
    buffer<uint32_t> unique_comp(num_components); // unique_comp[c] -> component id of the c-th unique component
    buffer<float> comp_wsum(num_components); // comp_wsum[c] -> total weight of the c-th unique component
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK) if(num_components > PARALLEL_GRAIN)
    for (uint32_t c = 0; c < num_components; c++) {
        const uint32_t begin = comp_heads[c], end = c + 1 < num_components ? comp_heads[c + 1] : num_edges;
        unique_comp[c] = comp[by_comp[begin]];
        float wsum = w[by_comp[begin]];
        for (uint32_t i = begin + 1; i < end; i++)
            wsum += w[by_comp[i]];
        comp_wsum[c] = wsum;
    }

    // rank components by descending weight, ties by component position
    const uint32_t comp_bits = bits_for(num_components);
    buffer<uint32_t> comp_idx = par_sort_permutation(num_components, 32u + comp_bits, [&](dim_t c) {
        return ((uint64_t)(~float_to_ordered_uint(comp_wsum[c] + 0.0f)) << comp_bits) | (uint64_t)c;
    }); // comp_idx[rank] -> unique component position with that rank
    buffer<uint32_t> comp_to_rank(num_edges); // comp_to_rank[comp id] -> rank of that component in the final chain ordering
    #pragma omp parallel for schedule(static) if(num_components > PARALLEL_GRAIN)
    for (uint32_t r = 0; r < num_components; r++)
        comp_to_rank[unique_comp[comp_idx[r]]] = r;

    // sort pairs by (comp_rank, pos, edge_id)
    const uint32_t pos_bits = bits_for(CHAIN_MAX_STEPS);
    buffer<uint32_t> edge_id = par_sort_permutation(num_edges, bits_for(num_components) + pos_bits, [&](dim_t i) {
        return ((uint64_t)comp_to_rank[comp[i]] << pos_bits) | (uint64_t)pos[i];
    }); // edge_id[idx] -> pair idx in sorted-space order

    // assign final sequence_idx = global position w.r.t. that sorted order
    #pragma omp parallel for schedule(static) if(num_edges > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < num_edges; i++)
        sequence_idx[orig[edge_id[i]]] = i;
}

// given a set of pairs proposed between nodes (pairs), isolate nodes without a pair,
// try to force them into a pair with another node in the same condition such that their
// combined size, distinct inbound count, and inbound pins count are within constraints. The
// objective is an almost-maximal number of formed pairs.
void build_orphan_pairs(
    const runconfig &cfg,
    const uint32_t *nodes_sizes,
    const uint32_t *nodes_pins,
    const uint32_t *inbound_count,
    const uint32_t *pairs,
    const uint32_t curr_num_nodes,
    const uint32_t candidates_count,
    uint32_t *groups // pre-initialized -> this routine writes group ids for paired nodes only.
) {
    buffer<uint32_t> free_nodes = par_copy_if(curr_num_nodes, [=](uint32_t i) { return pairs[i * candidates_count] == UINT32_MAX; }); // free_nodes[idx] -> idx-th orphan node

    const uint32_t num_free = (uint32_t)free_nodes.size();
    LOG(cfg) std::cout << "Orphans nodes found: " << num_free << "\n";
    if (num_free < 2) return;

    // sort orphans by (size, idx)
    const uint32_t max_size = par_reduce<uint32_t>(num_free, 0u, [&](dim_t i) { return nodes_sizes[free_nodes[i]]; }, [](uint32_t a, uint32_t b) { return a > b ? a : b; });
    buffer<uint32_t> by_size = par_sort_permutation(num_free, 32u + bits_for(max_size), [&](dim_t i) {
        return ((uint64_t)nodes_sizes[free_nodes[i]] << 32) | (uint64_t)free_nodes[i];
    });
    buffer<uint32_t> sorted_free(num_free); // sorted_free[idx] -> orphan node sorted by (size, idx)
    par_gather(by_size.data(), num_free, free_nodes.data(), sorted_free.data());

    LAUNCH(cfg) RUN << "pair orphans kernel (threads=" << cfg.threads << ") ...\n";
    pair_kth_smallest_with_kth_largest(
        sorted_free.data(),
        num_free,
        nodes_sizes,
        nodes_pins,
        inbound_count,
        groups
    );
}
