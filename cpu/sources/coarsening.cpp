#include <set>
#include <tuple>
#include <vector>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include "runconfig.hpp"

#include "coarsening.hpp"

#include "utils.hpp"
#include "defines.hpp"
#include "chaining.hpp"
#include "constants.hpp"

using namespace config;

void candidatesProposal(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t *srcs_count,
    const uint32_t *neighbors,
    const dim_t *neighbors_offsets,
    const uint32_t *touching,
    const dim_t *touching_offsets,
    const uint32_t *inbound_count,
    const float *hedge_weights,
    const uint32_t *nodes_sizes,
    const uint32_t *nodes_pins,
    const uint32_t curr_num_nodes,
    uint32_t *pairs,
    uint32_t *u_scores
) {
    // NOTE: the kernel writes every candidate slot, no need to zero-out its outputs
    LAUNCH(cfg) RUN << "candidates kernel (threads=" << cfg.threads << ") ...\n";
    candidates_kernel(
        hedges,
        hedges_offsets,
        srcs_count,
        neighbors,
        neighbors_offsets,
        touching,
        touching_offsets,
        inbound_count,
        hedge_weights,
        nodes_sizes,
        nodes_pins,
        curr_num_nodes,
        cfg.candidates_count,
        pairs,
        u_scores
    );
}

std::tuple<uint32_t, buffer<uint32_t>, buffer<uint32_t>, buffer<uint32_t>, buffer<uint32_t>, buffer<dim_t>> groupNodes(
    const runconfig &cfg,
    const uint32_t *inbound_count,
    const uint32_t *pairs,
    const uint32_t *u_scores,
    const uint32_t *nodes_sizes,
    const uint32_t *nodes_pins,
    const uint32_t curr_num_nodes,
    slot *slots
) {
    // prepare this level's coarsening groups
    buffer<uint32_t> groups(curr_num_nodes); // groups[node idx] -> node's group id (zero-based)

    LAUNCH(cfg) RUN << "grouping kernel (threads=" << cfg.threads << ", " << (cfg.exact_matching ? "exact" : "edge") << " gains) ...\n";
    grouping_kernel(
        pairs,
        u_scores,
        curr_num_nodes,
        cfg.candidates_count,
        cfg.exact_matching,
        slots,
        groups.data()
    );

    // for nodes that have no candidate and are left alone (no -valid- neighbors), try to pair them up with each other as to create as many groups as possible
    // => impose the sum of sizes and the sum of inbound set sizes < constraints
    // => the idea is to try pairs among non-neighbors, therefore the inbound set size intersection can already be taken as empty (hence, sum set sizes)
    build_orphan_pairs(
        cfg,
        nodes_sizes,
        nodes_pins,
        inbound_count,
        pairs,
        curr_num_nodes,
        cfg.candidates_count,
        groups.data()
    );

    // order groups (parallel label compression)
    // as of now "groups" contains the new non-zero-based group id for every node, and group ids are node ids
    // => flag the group ids in use, their prefix sum gives the new zero-based group ids, in the same order as the old ones
    buffer<uint32_t> new_group_id(curr_num_nodes); // new_group_id[old group id] -> 1 if the id is in use, then (after the scan) its new group id + 1
    par_fill<uint32_t>(new_group_id.data(), curr_num_nodes, 0u);
    #pragma omp parallel for schedule(static) if(curr_num_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < curr_num_nodes; node++)
        atomic_store<uint32_t>(&new_group_id[groups[node]], 1u);
    par_inclusive_scan<uint32_t>(new_group_id.data(), curr_num_nodes);
    // the last value, after the scan, gives you the total number of distinct groups
    const uint32_t new_num_nodes = new_group_id[curr_num_nodes - 1];
    #pragma omp parallel for schedule(static) if(curr_num_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < curr_num_nodes; node++)
        groups[node] = new_group_id[groups[node]] - 1;
    new_group_id.release();
    // => now "groups[idx]" contains the new zero-based group ID for every node

    // prepare uncoarsening map (node ids sorted by group id, ties by node id)
    buffer<uint32_t> ungroups = par_sort_permutation(curr_num_nodes, bits_for(new_num_nodes), [&](dim_t node) { return (uint64_t)groups[node]; }); // ungroups[ungroups_offsets[group id] + i] -> the group's i-th node (its original idx)

    // prepare uncoarsening map offsets (offset where each ungroup starts)
    buffer<dim_t> ungroups_offsets((dim_t)new_num_nodes + 1); // ungroups_offsets[group id] -> offset of the group's first node in ungroups
    #pragma omp parallel for schedule(static) if(curr_num_nodes > PARALLEL_GRAIN)
    for (uint32_t pos = 0; pos < curr_num_nodes; pos++)
        if (pos == 0 || groups[ungroups[pos]] != groups[ungroups[pos - 1]])
            ungroups_offsets[groups[ungroups[pos]]] = pos;
    ungroups_offsets[new_num_nodes] = curr_num_nodes;

    // prepare this level's cumulative groups sizes and pins
    // NOTE: "node sizes" = size of the nodes that entered this level, "group sizes" = cumulative size of groups constructed on this level
    // NOTE: inbound pins are additive just like sizes, a node always brings along its whole inbound set
    buffer<uint32_t> groups_sizes(new_num_nodes); // group_sizes[group id] = sum of sizes of all nodes in that group
    buffer<uint32_t> groups_pins(new_num_nodes); // group_pins[group id] = sum of pins of all nodes in that group
    #pragma omp parallel for schedule(static) if(new_num_nodes > PARALLEL_GRAIN)
    for (uint32_t group = 0; group < new_num_nodes; group++) {
        uint32_t size = 0u, pins = 0u;
        for (dim_t u = ungroups_offsets[group]; u < ungroups_offsets[group + 1]; u++) {
            size += nodes_sizes[ungroups[u]];
            pins += nodes_pins[ungroups[u]];
        }
        groups_sizes[group] = size;
        groups_pins[group] = pins;
    }

    return std::make_tuple(new_num_nodes, std::move(groups), std::move(groups_sizes), std::move(groups_pins), std::move(ungroups), std::move(ungroups_offsets));
}


// LOGGING

void logCandidates(
    const runconfig &cfg,
    const uint32_t *pairs_tmp,
    const uint32_t *scores_tmp,
    const uint32_t curr_num_nodes
) {
    std::vector<std::set<uint32_t>> candidates_count(cfg.candidates_count);
    std::cout << "Pairing results:";
    for (uint32_t i = 0; i < curr_num_nodes; ++i) {
        if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH))
            std::cout << "\n  node " << i << " ->";
        for (uint32_t j = 0; j < cfg.candidates_count; ++j) {
            float score = ((float)scores_tmp[i * cfg.candidates_count + j])/FIXED_POINT_SCALE;
            uint32_t target = pairs_tmp[i * cfg.candidates_count + j];
            candidates_count[j].insert(target);
            if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH)) {
                if (target == UINT32_MAX) std::cout << " (" << j << " target=none score=none)";
                else if (target == i) std::cout << " !!SELF TARGETED!! ";
                else std::cout << " (" << j << " target=" << target << " score=" << std::fixed << std::setprecision(3) << score << ")";
            }
            if (target == UINT32_MAX) continue;
            // check the symmetry invariant: mutual pairs or the other has found a higher score pair (or one with lower id - tiebreaker) [easy for j = 0, for j > 0 check first that the target wasn't already used at a lower j]
            const uint32_t *target_pairs = pairs_tmp + target * cfg.candidates_count;
            if (
                target_pairs[j] != i && target_pairs[j] != UINT32_MAX
                && std::find(target_pairs, target_pairs + j, i) == target_pairs + j
                && !(scores_tmp[target * cfg.candidates_count + j] > score || scores_tmp[target * cfg.candidates_count + j] == score && target_pairs[j] < i)
            ) {
                std::cerr
                    << "\n  WARNING, symmetry violated: node " << i
                    << " (" << j << " target=" << target << " score=" << std::fixed << std::setprecision(3) << score
                    << ") AND node " << target << " (" << j << " target=" << target_pairs[j]
                    << " score=" << std::fixed << std::setprecision(3) << scores_tmp[target * cfg.candidates_count + j] << ") !!";
            }
        }
    }
    std::cout << "\n";
    for (uint32_t j = 0; j < cfg.candidates_count; ++j)
        std::cout << "Candidates count (" << j << "): " << candidates_count[j].size() << "\n";
}

void logGroups(
    const runconfig &cfg,
    const uint32_t *pairs_tmp,
    const uint32_t *groups_tmp,
    const uint32_t *groups_sizes_tmp,
    const uint32_t *groups_pins_tmp,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes
) {
    std::unordered_map<uint32_t, int> groups_count;
    std::cout << "Grouping results:\n";
    for (uint32_t i = 0; i < curr_num_nodes; ++i) {
        uint32_t group = groups_tmp[i];
        uint32_t group_size = groups_sizes_tmp[group];
        uint32_t group_pins = groups_pins_tmp[group];
        groups_count[group]++;
        if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH)) {
            std::cout << "  node " << i << " ->";
            for (uint32_t j = 0; j < cfg.candidates_count; ++j) {
                uint32_t target = pairs_tmp[i * cfg.candidates_count + j];
                if (target == UINT32_MAX) std::cout << " (" << j << " target=none)";
                else std::cout << " (" << j << " target=" << target << ")";
            }
            std::cout << " group=" << group << " group_size=" << group_size << " group_pins=" << group_pins << "\n";
        }
    }
    long long max_gs = 0, sum_gs = 0;
    long long max_gp = 0, sum_gp = 0;
    for (uint32_t i = 0; i < new_num_nodes; ++i) {
        uint32_t group_size = groups_sizes_tmp[i];
        uint32_t group_pins = groups_pins_tmp[i];
        sum_gs += group_size;
        sum_gp += group_pins;
        if (group_size > max_gs) max_gs = group_size;
        if (group_pins > max_gp) max_gp = group_pins;
        if (group_size > max_nodes_per_part)
            std::cerr << "  WARNING, max group size constraint (" << max_nodes_per_part << ") violated by group=" << i << " with group_size=" << group_size << " !!\n";
        if (group_pins > max_pins_per_part)
            std::cerr << "  WARNING, max group pins constraint (" << max_pins_per_part << ") violated by group=" << i << " with group_pins=" << group_pins << " !!\n";
    }
    long long max_cgs = 0, sum_cgs = 0;
    for (const auto& [group, count] : groups_count) {
        sum_cgs += count;
        if (count > max_cgs) max_cgs = count;
    }
    std::cout << "Groups count: " << groups_count.size() << "\n  Max coarse group size: " << max_cgs << ", Avg coarse group size: " << std::fixed << std::setprecision(2) << (float)sum_cgs/groups_count.size() << "\n";
    std::cout << "  Max nodes group size: " << max_gs << ", Avg nodes group size: " << std::fixed << std::setprecision(2) << (float)sum_gs/groups_count.size() << "\n";
    std::cout << "  Max nodes group pins: " << max_gp << ", Avg nodes group pins: " << std::fixed << std::setprecision(2) << (float)sum_gp/groups_count.size() << "\n";
}
