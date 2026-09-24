#include <tuple>
#include <vector>
#include <iostream>
#include <algorithm>
#include <parallel/algorithm>

#include "runconfig.hpp"

#include "postprocess.hpp"

#include "utils.hpp"
#include "constants.hpp"
#include "prims.hpp"
#include "defines.hpp"

using namespace config;

// greedily pack items, in the given order, into consecutive segments, within caps on their summed sizes, inbound, and pins
// => binary lifting, parallel over items: jump tables over the monotone "next segment start" relation tell each item how
//    many segments precede it, that is, the id of its greedy pack
// returns pack_of[pos] -> greedy pack id of the item at sorted position pos
static buffer<uint32_t> greedyPacks(
    const dim_t *sorted_sizes,
    const dim_t *sorted_inbound,
    const dim_t *sorted_pins,
    const uint32_t num_items
) {
    // build exclusive-prefix buffers to query any greedy segment sum
    buffer<dim_t> prefix_sizes((dim_t)num_items + 1); // prefix_sizes[pos] -> sum of sorted_sizes in [0, pos)
    buffer<dim_t> prefix_inbound((dim_t)num_items + 1); // prefix_inbound[pos] -> sum of sorted_inbound in [0, pos)
    buffer<dim_t> prefix_pins((dim_t)num_items + 1); // prefix_pins[pos] -> sum of sorted_pins in [0, pos)
    prefix_sizes[0] = 0;
    prefix_inbound[0] = 0;
    prefix_pins[0] = 0;
    par_copy<dim_t>(prefix_sizes.data() + 1, sorted_sizes, num_items);
    par_copy<dim_t>(prefix_inbound.data() + 1, sorted_inbound, num_items);
    par_copy<dim_t>(prefix_pins.data() + 1, sorted_pins, num_items);
    par_inclusive_scan<dim_t>(prefix_sizes.data(), (dim_t)num_items + 1);
    par_inclusive_scan<dim_t>(prefix_inbound.data(), (dim_t)num_items + 1);
    par_inclusive_scan<dim_t>(prefix_pins.data(), (dim_t)num_items + 1);

    // find the first violating prefix slot for each possible greedy segment start, then turn it into the next greedy segment start
    std::vector<buffer<uint32_t>> jump; // jump[k][pos] -> applying next_start 2^k times from pos
    jump.emplace_back((dim_t)num_items + 1);
    uint32_t *next_start = jump[0].data(); // next_start[pos] -> next segment start after greedily packing from pos
    #pragma omp parallel for schedule(static) if(num_items > PARALLEL_GRAIN)
    for (uint32_t pos = 0; pos < num_items; pos++) {
        const uint32_t size_ub = (uint32_t)(std::upper_bound(prefix_sizes.data(), prefix_sizes.data() + num_items + 1, prefix_sizes[pos] + (dim_t)max_nodes_per_part) - prefix_sizes.data());
        const uint32_t inbound_ub = (uint32_t)(std::upper_bound(prefix_inbound.data(), prefix_inbound.data() + num_items + 1, prefix_inbound[pos] + (dim_t)max_inbound_per_part) - prefix_inbound.data());
        const uint32_t pins_ub = (uint32_t)(std::upper_bound(prefix_pins.data(), prefix_pins.data() + num_items + 1, prefix_pins[pos] + (dim_t)max_pins_per_part) - prefix_pins.data());
        next_start[pos] = std::max(pos + 1u, std::min(size_ub, std::min(inbound_ub, pins_ub)) - 1u);
    }
    next_start[num_items] = num_items;

    // build binary-lifting jump tables over the monotone next-start relation
    while ((1u << (jump.size() - 1)) < num_items) {
        const uint32_t *prev_jump = jump.back().data();
        buffer<uint32_t> next_jump((dim_t)num_items + 1); // next_jump[pos] -> applying the current jump twice from pos
        par_gather(prev_jump, (dim_t)num_items + 1, prev_jump, next_jump.data());
        jump.emplace_back(std::move(next_jump));
    }

    // recover, for each sorted position, how many greedy jumps are needed to reach it from zero
    buffer<uint32_t> pack_of(num_items);
    #pragma omp parallel for schedule(static) if(num_items > PARALLEL_GRAIN)
    for (uint32_t pos = 0; pos < num_items; pos++) {
        uint32_t curr_pos = 0, packs = 0;
        // try the current jump length and keep it whenever it does not pass the queried position
        for (int level = (int)jump.size() - 1; level >= 0; --level) {
            const uint32_t candidate = jump[level][curr_pos];
            if (candidate <= pos) {
                curr_pos = candidate;
                packs += 1u << level;
            }
        }
        pack_of[pos] = packs;
    }
    return pack_of;
}

// map each greedy pack to the lowest original id it contains, and write it for every sorted position
// => the packs of consecutive positions are consecutive (and non-decreasing) ids
static buffer<uint32_t> packRepresentatives(
    const uint32_t *pack_of,
    const uint32_t *sorted_ids,
    const uint32_t num_items
) {
    const uint32_t num_packs = num_items > 0 ? pack_of[num_items - 1] + 1 : 0;
    buffer<uint32_t> rep_ids(num_packs); // rep_ids[pack] -> representative original id of that greedy pack
    par_fill<uint32_t>(rep_ids.data(), num_packs, UINT32_MAX);
    #pragma omp parallel for schedule(static) if(num_items > PARALLEL_GRAIN)
    for (uint32_t pos = 0; pos < num_items; pos++)
        if (pos == 0 || pack_of[pos] != pack_of[pos - 1]) {
            uint32_t rep = UINT32_MAX;
            for (uint32_t p = pos; p < num_items && pack_of[p] == pack_of[pos]; p++)
                rep = std::min(rep, sorted_ids[p]);
            rep_ids[pack_of[pos]] = rep;
        }
    buffer<uint32_t> rep_of(num_items); // rep_of[pos] -> representative id of the item at sorted position pos
    par_gather(pack_of, num_items, rep_ids.data(), rep_of.data());
    return rep_of;
}

uint32_t zeroBaseIds(
    const uint32_t num_items,
    const uint32_t num_ids,
    uint32_t *ids
) {
    // make ids zero-based again, preserving their order: flag the ids in use, their prefix sum gives the new ids
    buffer<uint32_t> new_id(num_ids); // new_id[old id] -> 1 if the id is in use, then (after the scan) its new id + 1
    par_fill<uint32_t>(new_id.data(), num_ids, 0u);
    #pragma omp parallel for schedule(static) if(num_items > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < num_items; i++)
        atomic_store<uint32_t>(&new_id[ids[i]], 1u);
    par_inclusive_scan<uint32_t>(new_id.data(), num_ids);
    #pragma omp parallel for schedule(static) if(num_items > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < num_items; i++)
        ids[i] = new_id[ids[i]] - 1;
    return num_ids > 0 ? new_id[num_ids - 1] : 0;
}

uint32_t greedyMergeGroups(
    const runconfig &cfg,
    const uint32_t *nodes_sizes,
    const uint32_t *nodes_pins,
    const uint32_t *inbound_count,
    const uint32_t *ungroups,
    const dim_t *ungroups_offsets,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes,
    uint32_t *groups,
    uint32_t *groups_sizes,
    uint32_t *groups_pins
) {
    /*
    * IDEA:
    * - merge together as many groups as possible (an heuristic is good enough) within constraints
    * - no need for costly exact constraint checks (e.g. with the exact inbound set intersection),
    *   just add together sizes, inbound counts, and pins between groups to get an upper-bound on constraints
    * - update "groups", "groups_sizes" and "groups_pins" accordingly
    * - return the new number number of groups
    *
    * NOTE: group inbound set sizes are not available, infer them from the inbound count of each node in the group
    * NOTE: no need to update ungroup and ungroup_offsets
    */

    (void)nodes_sizes; (void)nodes_pins; // NOTE: merged groups' sizes and pins are re-summed from the groups', rather than from their nodes'

    if (curr_num_nodes == 0 || new_num_nodes < 2) return new_num_nodes;
    INFO(cfg) std::cout << "Greedy groups merge over " << new_num_nodes << " groups ...\n";

    // infer a cheap inbound upper-bound for each current group
    buffer<uint32_t> groups_inbound(new_num_nodes); // groups_inbound[g] -> summed inbound count upper-bound of group g
    #pragma omp parallel for schedule(static) if(new_num_nodes > PARALLEL_GRAIN)
    for (uint32_t group = 0; group < new_num_nodes; group++) {
        uint32_t inbound = 0u;
        for (dim_t u = ungroups_offsets[group]; u < ungroups_offsets[group + 1]; u++)
            inbound += inbound_count[ungroups[u]];
        groups_inbound[group] = inbound;
    }

    // sort groups by increasing (size, inbound, pins, id) to greedily consume the smallest ones first
    std::vector<uint32_t> group_index(new_num_nodes); // group_index[pos] -> original group id at sorted position pos
    par_sequence<uint32_t>(group_index.data(), new_num_nodes);
    __gnu_parallel::sort(group_index.begin(), group_index.end(), [&](uint32_t a, uint32_t b) {
        if (groups_sizes[a] != groups_sizes[b]) return groups_sizes[a] < groups_sizes[b];
        if (groups_inbound[a] != groups_inbound[b]) return groups_inbound[a] < groups_inbound[b];
        if (groups_pins[a] != groups_pins[b]) return groups_pins[a] < groups_pins[b];
        return a < b;
    });

    // gather the sorted sizes, inbound bounds, and pins that drive the greedy packing
    buffer<dim_t> sorted_sizes(new_num_nodes), sorted_inbound(new_num_nodes), sorted_pins(new_num_nodes);
    #pragma omp parallel for schedule(static) if(new_num_nodes > PARALLEL_GRAIN)
    for (uint32_t pos = 0; pos < new_num_nodes; pos++) {
        sorted_sizes[pos] = groups_sizes[group_index[pos]];
        sorted_inbound[pos] = groups_inbound[group_index[pos]];
        sorted_pins[pos] = groups_pins[group_index[pos]];
    }

    // greedily pack groups, and map each pack to the lowest original group id it contains
    buffer<uint32_t> merge_groups = greedyPacks(sorted_sizes.data(), sorted_inbound.data(), sorted_pins.data(), new_num_nodes); // merge_groups[pos] -> greedy pack id of the group at sorted position pos
    buffer<uint32_t> group_map_sorted = packRepresentatives(merge_groups.data(), group_index.data(), new_num_nodes); // group_map_sorted[pos] -> representative id of the group at sorted position pos

    // build the original-group-id remap induced by the greedy packing
    buffer<uint32_t> group_map(new_num_nodes); // group_map[g] -> representative id that group g is merged into
    #pragma omp parallel for schedule(static) if(new_num_nodes > PARALLEL_GRAIN)
    for (uint32_t pos = 0; pos < new_num_nodes; pos++)
        group_map[group_index[pos]] = group_map_sorted[pos];

    // rewrite each node's group id through the greedy group remap, and rebuild them as a compact zero-based range
    #pragma omp parallel for schedule(static) if(curr_num_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < curr_num_nodes; node++)
        groups[node] = group_map[groups[node]];
    const uint32_t new_num_groups = zeroBaseIds(curr_num_nodes, new_num_nodes, groups);

    // rebuild cumulative sizes and pins for the compacted merged groups (sums of those of the merged groups)
    buffer<uint32_t> old_groups_sizes(new_num_nodes), old_groups_pins(new_num_nodes);
    par_copy<uint32_t>(old_groups_sizes.data(), groups_sizes, new_num_nodes);
    par_copy<uint32_t>(old_groups_pins.data(), groups_pins, new_num_nodes);
    par_fill<uint32_t>(groups_sizes, new_num_groups, 0u);
    par_fill<uint32_t>(groups_pins, new_num_groups, 0u);
    // NOTE: ungroups still lists the nodes of the old groups, all nodes of an old group now share the compacted id of its pack
    #pragma omp parallel for schedule(static) if(new_num_nodes > PARALLEL_GRAIN)
    for (uint32_t group = 0; group < new_num_nodes; group++) {
        const uint32_t compact_id = groups[ungroups[ungroups_offsets[group]]];
        atomic_add<uint32_t>(&groups_sizes[compact_id], old_groups_sizes[group]);
        atomic_add<uint32_t>(&groups_pins[compact_id], old_groups_pins[group]);
    }
    INFO(cfg) std::cout << "Greedy groups merge reduced groups from " << new_num_nodes << " to " << new_num_groups << "\n";

    return new_num_groups;
}

void mergeSmallPartitions(
    const runconfig &cfg,
    const uint32_t *partitions_sizes,
    const uint32_t *partitions_inbound_sizes,
    const uint32_t *partitions_pins,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    uint32_t *partitions
) {
    /*
    * Given a set of partitions, try to merge together smaller the ones, within constraints
    * => reduce for free the total number of partitions, where feasible
    * NOTE: this does NOT update "partitions_sizes", "partitions_inbound_sizes" and "partitions_pins" !!
    */

    // enumerate all partition ids and extract the ones considered small
    buffer<uint32_t> small_parts_buf = par_copy_if(num_partitions, [=](uint32_t p) { return partitions_sizes[p] < SMALL_PART_MERGE_SIZE_THRESHOLD; }); // small_parts[pos] -> partition id of the pos-th small partition
    const uint32_t smallest_part_size = par_reduce<uint32_t>(num_partitions, UINT32_MAX, [=](dim_t p) { return partitions_sizes[p]; }, [](uint32_t a, uint32_t b) { return a < b ? a : b; });
    INFO(cfg) std::cout << "Smallest partition size: " << smallest_part_size << "\n";
    const uint32_t num_small_parts = (uint32_t)small_parts_buf.size();
    if (num_small_parts == 0) {
        INFO(cfg) std::cout << "Partitions compression not performed ...\n";
        return;
    }
    INFO(cfg) std::cout << "Partitions compression over " << num_small_parts << " partitions ...\n";

    // sort the small partitions by increasing (size, inbound, pins, id)
    std::vector<uint32_t> small_parts(small_parts_buf.data(), small_parts_buf.data() + num_small_parts);
    __gnu_parallel::sort(small_parts.begin(), small_parts.end(), [=](uint32_t a, uint32_t b) {
        if (partitions_sizes[a] != partitions_sizes[b]) return partitions_sizes[a] < partitions_sizes[b];
        if (partitions_inbound_sizes[a] != partitions_inbound_sizes[b]) return partitions_inbound_sizes[a] < partitions_inbound_sizes[b];
        if (partitions_pins[a] != partitions_pins[b]) return partitions_pins[a] < partitions_pins[b];
        return a < b;
    });

    // gather the sorted sizes, inbound bounds, and pins that drive the greedy packing
    buffer<dim_t> sorted_sizes(num_small_parts), sorted_inbound(num_small_parts), sorted_pins(num_small_parts);
    #pragma omp parallel for schedule(static) if(num_small_parts > PARALLEL_GRAIN)
    for (uint32_t pos = 0; pos < num_small_parts; pos++) {
        sorted_sizes[pos] = partitions_sizes[small_parts[pos]];
        sorted_inbound[pos] = partitions_inbound_sizes[small_parts[pos]];
        sorted_pins[pos] = partitions_pins[small_parts[pos]];
    }

    // greedily pack small partitions, and map each pack to the lowest original partition id it contains
    buffer<uint32_t> packs = greedyPacks(sorted_sizes.data(), sorted_inbound.data(), sorted_pins.data(), num_small_parts); // packs[pos] -> greedy pack id of the small partition at sorted position pos
    buffer<uint32_t> new_pids = packRepresentatives(packs.data(), small_parts.data(), num_small_parts); // new_pids[pos] -> representative id of the small partition at sorted position pos

    // build the partition-id remap induced by the greedy packing
    buffer<uint32_t> pid_map(num_partitions); // pid_map[p] -> representative partition id that p is merged into
    par_sequence<uint32_t>(pid_map.data(), num_partitions);
    #pragma omp parallel for schedule(static) if(num_small_parts > PARALLEL_GRAIN)
    for (uint32_t pos = 0; pos < num_small_parts; pos++)
        pid_map[small_parts[pos]] = new_pids[pos];

    // rewrite each node's partition id through the greedy partition remap
    #pragma omp parallel for schedule(static) if(num_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < num_nodes; node++)
        partitions[node] = pid_map[partitions[node]];
}
