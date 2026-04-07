#include <set>
#include <tuple>
#include <vector>

#include "thruster.cuh"

#include "runconfig.hpp"

#include "postprocess.cuh"

#include "utils.cuh"
#include "defines.cuh"

using namespace config;

uint32_t greedyMergeGroups(
    const runconfig &cfg,
    const uint32_t *d_nodes_sizes,
    const uint32_t *d_inbound_count,
    const uint32_t *d_ungroups,
    const dim_t *d_ungroups_offsets,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    uint32_t *d_groups,
    uint32_t *d_groups_sizes
) {
    /*
    * IDEA:
    * - merge together as many groups as possible (an heuristic is good enough) within constraints
    * - no need for costly exact constraint checks (e.g. with the exact inbound set intersection),
    *   just add together sizes and inbound counts between groups to get an upper-bound on constraints
    * - update "d_groups" and "d_groups_sizes" accordingly
    * - return the new number number of groups
    *
    * NOTE: group inbound set sizes are not available, infer them from the inbound count of each node in the group
    * NOTE: no need to update ungroup and ungroup_offsets
    */

    if (curr_num_nodes == 0 || new_num_nodes < 2) return new_num_nodes;
    INFO(cfg) std::cout << "Greedy groups merge over " << new_num_nodes << " groups ...\n";

    // prepare device views over the current groups data
    thrust::device_ptr<uint32_t> t_groups(d_groups);
    thrust::device_ptr<uint32_t> t_groups_sizes(d_groups_sizes);
    thrust::device_ptr<const uint32_t> t_nodes_sizes(d_nodes_sizes);
    thrust::device_ptr<const uint32_t> t_inbound_count(d_inbound_count);
    thrust::device_ptr<const uint32_t> t_ungroups(d_ungroups);
    thrust::device_ptr<const dim_t> t_ungroups_offsets(d_ungroups_offsets);

    // materialize the sorted-space group id of each node listed in d_ungroups
    thrust::device_vector<uint32_t> t_group_keys(curr_num_nodes, 0u); // group_keys[pos] -> group id of d_ungroups[pos]
    thrust::device_vector<uint32_t> t_group_index(new_num_nodes); // group_index[pos] -> original group id at sorted position pos
    thrust::sequence(t_group_index.begin(), t_group_index.end());
    thrust::scatter(
        t_group_index.begin(), t_group_index.end(),
        t_ungroups_offsets, t_group_keys.begin()
    );
    thrust::inclusive_scan(
        t_group_keys.begin(), t_group_keys.end(),
        t_group_keys.begin(), thrust::maximum<uint32_t>()
    );

    // infer a cheap inbound upper-bound for each current group.
    thrust::device_vector<uint32_t> t_groups_inbound(new_num_nodes); // groups_inbound[g] -> summed inbound count upper-bound of group g
    auto inbound_values_begin = thrust::make_permutation_iterator(t_inbound_count, t_ungroups);
    thrust::reduce_by_key(
        t_group_keys.begin(), t_group_keys.end(), inbound_values_begin,
        thrust::make_discard_iterator(), t_groups_inbound.begin()
    );
    uint32_t *d_groups_inbound_ptr = thrust::raw_pointer_cast(t_groups_inbound.data());

    // sort groups by increasing (size, inbound, id) to greedily consume the smallest ones first
    thrust::stable_sort(
        t_group_index.begin(), t_group_index.end(),
        [=] __host__ __device__ (uint32_t a, uint32_t b) {
            uint32_t sa = d_groups_sizes[a];
            uint32_t sb = d_groups_sizes[b];
            if (sa != sb) return sa < sb;
            uint32_t ia = d_groups_inbound_ptr[a];
            uint32_t ib = d_groups_inbound_ptr[b];
            if (ia != ib) return ia < ib;
            return a < b;
        }
    );

    // gather the sorted sizes and inbound bounds that drive the greedy packing
    thrust::device_vector<dim_t> t_sorted_sizes(new_num_nodes); // sorted_sizes[pos] -> size of the group at sorted position pos
    thrust::device_vector<dim_t> t_sorted_inbound(new_num_nodes); // sorted_inbound[pos] -> inbound upper-bound of the group at sorted position pos
    thrust::gather(t_group_index.begin(), t_group_index.end(), t_groups_sizes, t_sorted_sizes.begin());
    thrust::gather(t_group_index.begin(), t_group_index.end(), t_groups_inbound.begin(), t_sorted_inbound.begin());

    // build exclusive-prefix buffers to query any greedy segment sum
    thrust::device_vector<dim_t> t_prefix_sizes(1 + new_num_nodes); // prefix_sizes[pos] -> sum of sorted_sizes in [0, pos)
    thrust::device_vector<dim_t> t_prefix_inbound(1 + new_num_nodes); // prefix_inbound[pos] -> sum of sorted_inbound in [0, pos)
    t_prefix_sizes[0] = 0;
    t_prefix_inbound[0] = 0;
    thrust::inclusive_scan(t_sorted_sizes.begin(), t_sorted_sizes.end(), t_prefix_sizes.begin() + 1);
    thrust::inclusive_scan(t_sorted_inbound.begin(), t_sorted_inbound.end(), t_prefix_inbound.begin() + 1);
    dim_t *d_prefix_sizes_ptr = thrust::raw_pointer_cast(t_prefix_sizes.data());
    dim_t *d_prefix_inbound_ptr = thrust::raw_pointer_cast(t_prefix_inbound.data());

    // compute the largest prefix values each greedy segment may reach from every start position
    thrust::device_vector<dim_t> t_size_targets(new_num_nodes); // size_targets[pos] -> max allowed prefix size for a segment starting at pos
    thrust::device_vector<dim_t> t_inbound_targets(new_num_nodes); // inbound_targets[pos] -> max allowed prefix inbound for a segment starting at pos
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0), thrust::make_counting_iterator<uint32_t>(new_num_nodes),
        t_size_targets.begin(),
        [=] __host__ __device__ (uint32_t i) { return d_prefix_sizes_ptr[i] + (dim_t)h_max_nodes_per_part; }
    );
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0), thrust::make_counting_iterator<uint32_t>(new_num_nodes),
        t_inbound_targets.begin(),
        [=] __host__ __device__ (uint32_t i) { return d_prefix_inbound_ptr[i] + (dim_t)h_max_inbound_per_part; }
    );

    // find the first violating prefix slot for each possible greedy segment start
    thrust::device_vector<uint32_t> t_size_next(new_num_nodes); // size_next[pos] -> first prefix slot that violates the size cap from pos
    thrust::device_vector<uint32_t> t_inbound_next(new_num_nodes); // inbound_next[pos] -> first prefix slot that violates the inbound cap from pos
    thrust::upper_bound(
        t_prefix_sizes.begin(), t_prefix_sizes.end(),
        t_size_targets.begin(), t_size_targets.end(),
        t_size_next.begin()
    );
    thrust::upper_bound(
        t_prefix_inbound.begin(), t_prefix_inbound.end(),
        t_inbound_targets.begin(), t_inbound_targets.end(),
        t_inbound_next.begin()
    );

    // convert the first violating prefix slot into the next greedy segment start
    thrust::device_vector<uint32_t> t_next_start(1 + new_num_nodes, new_num_nodes); // next_start[pos] -> next segment start after greedily packing from pos
    auto next_start_input_begin = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::make_counting_iterator<uint32_t>(0),
        t_size_next.begin(), t_inbound_next.begin()
    ));
    thrust::transform(
        next_start_input_begin,
        next_start_input_begin + new_num_nodes,
        t_next_start.begin(),
        [] __host__ __device__ (const thrust::tuple<uint32_t, uint32_t, uint32_t>& x) {
            uint32_t pos = thrust::get<0>(x);
            uint32_t size_ub = thrust::get<1>(x);
            uint32_t inbound_ub = thrust::get<2>(x);
            return max(pos + 1u, min(size_ub, inbound_ub) - 1u);
        }
    );

    // build binary-lifting jump tables over the monotone next-start relation
    std::vector<thrust::device_vector<uint32_t>> t_jump; // jump[k][pos] -> applying next_start 2^k times from pos
    t_jump.emplace_back(t_next_start);
    while ((1u << (t_jump.size() - 1)) < new_num_nodes) {
        thrust::device_vector<uint32_t> t_next_jump(1 + new_num_nodes); // next_jump[pos] -> applying the current jump twice from pos
        thrust::gather(
            t_jump.back().begin(), t_jump.back().end(),
            t_jump.back().begin(), t_next_jump.begin()
        );
        t_jump.emplace_back(std::move(t_next_jump));
    }

    // recover, for each sorted position, how many greedy jumps are needed to reach it from zero
    thrust::device_vector<uint32_t> t_merge_groups(new_num_nodes, 0u); // merge_groups[pos] -> greedy pack id of the group at sorted position pos
    thrust::device_vector<uint32_t> t_curr_pos(new_num_nodes, 0u); // curr_pos[pos] -> current jump-chain position while binary lifting pos
    thrust::device_vector<uint32_t> t_sorted_pos(new_num_nodes); // sorted_pos[pos] -> identity sorted position pos
    thrust::sequence(t_sorted_pos.begin(), t_sorted_pos.end());
    thrust::device_vector<uint32_t> t_candidate_pos(new_num_nodes); // candidate_pos[pos] -> tentative jump destination at the current binary-lifting level
    uint32_t *d_merge_groups_ptr = thrust::raw_pointer_cast(t_merge_groups.data());
    uint32_t *d_curr_pos_ptr = thrust::raw_pointer_cast(t_curr_pos.data());
    uint32_t *d_sorted_pos_ptr = thrust::raw_pointer_cast(t_sorted_pos.data());
    uint32_t *d_candidate_pos_ptr = thrust::raw_pointer_cast(t_candidate_pos.data());
    // NOTE: this host side control loop is O(log(n)), not too cheap, not too pricey...
    for (int level = (int)t_jump.size() - 1; level >= 0; --level) {
        // try the current jump length and keep it whenever it does not pass the queried position
        thrust::gather(
            t_curr_pos.begin(), t_curr_pos.end(),
            t_jump[level].begin(), t_candidate_pos.begin()
        );
        uint32_t step = 1u << level;
        thrust::for_each_n(
            thrust::make_counting_iterator<uint32_t>(0), new_num_nodes,
            [=] __device__ (uint32_t i) {
                uint32_t candidate = d_candidate_pos_ptr[i];
                if (candidate <= d_sorted_pos_ptr[i]) {
                    d_curr_pos_ptr[i] = candidate;
                    d_merge_groups_ptr[i] += step;
                }
            }
        );
    }

    // map each greedy pack to the lowest original group id it contains
    thrust::device_vector<uint32_t> t_rep_ids(new_num_nodes); // rep_ids[pack] -> representative original group id of that greedy pack
    auto rep_end = thrust::reduce_by_key(
        t_merge_groups.begin(), t_merge_groups.end(),
        t_group_index.begin(), thrust::make_discard_iterator(),
        t_rep_ids.begin(), thrust::equal_to<uint32_t>(), thrust::minimum<uint32_t>()
    );
    t_rep_ids.resize(rep_end.second - t_rep_ids.begin());

    // build the original-group-id remap induced by the greedy packing
    thrust::device_vector<uint32_t> t_group_map_sorted(new_num_nodes); // group_map_sorted[pos] -> representative id of the group at sorted position pos
    thrust::gather(
        t_merge_groups.begin(), t_merge_groups.end(),
        t_rep_ids.begin(), t_group_map_sorted.begin()
    );
    thrust::device_vector<uint32_t> t_group_map(new_num_nodes); // group_map[g] -> representative id that group g is merged into
    thrust::scatter(
        t_group_map_sorted.begin(), t_group_map_sorted.end(),
        t_group_index.begin(), t_group_map.begin()
    );

    // rewrite each node's group id through the greedy group remap
    uint32_t *d_group_map_ptr = thrust::raw_pointer_cast(t_group_map.data());
    thrust::transform(
        t_groups, t_groups + curr_num_nodes, t_groups,
        [=] __host__ __device__ (uint32_t g) { return d_group_map_ptr[g]; }
    );

    // reorder merged group ids and rebuild them as a compact zero-based range
    thrust::device_vector<uint32_t> t_indices(curr_num_nodes); // indices[pos] -> original node id of the node at sorted position pos
    thrust::sequence(t_indices.begin(), t_indices.end());
    thrust::sort_by_key(t_groups, t_groups + curr_num_nodes, t_indices.begin());
    thrust::device_vector<uint32_t> t_headflags(curr_num_nodes); // headflags[pos] -> compact merged group id of the node at sorted position pos
    t_headflags[0] = 0;
    thrust::transform(
        t_groups + 1, t_groups + curr_num_nodes, t_groups, t_headflags.begin() + 1,
        [] __device__ (uint32_t curr, uint32_t prev) { return curr != prev ? 1u : 0u; }
    );
    thrust::inclusive_scan(t_headflags.begin(), t_headflags.end(), t_headflags.begin());
    const uint32_t new_num_groups = t_headflags.back() + 1;

    // rebuild cumulative sizes for the compacted merged groups
    auto node_sizes_values_begin = thrust::make_permutation_iterator(t_nodes_sizes, t_indices.begin());
    thrust::reduce_by_key(
        t_headflags.begin(), t_headflags.end(), node_sizes_values_begin,
        thrust::make_discard_iterator(), t_groups_sizes
    );

    // scatter the compact merged group ids back to the original node order
    thrust::scatter(t_headflags.begin(), t_headflags.end(), t_indices.begin(), t_groups);
    INFO(cfg) std::cout << "Greedy groups merge reduced groups from " << new_num_nodes << " to " << new_num_groups << "\n";

    return new_num_groups;
}

void mergeSmallPartitions(
    const runconfig &cfg,
    const uint32_t *d_partitions_sizes,
    const uint32_t *d_partitions_inbound_sizes,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    uint32_t *d_partitions
) {
    /*
    * Given a set of partitions, try to merge together smaller the ones, within constraints
    * => reduce for free the total number of partitions, where feasible
    * NOTE: this does NOT update "partitions_sizes" and "partitions_inbound_sizes" !!
    */

    // prepare device views over the current partitioning
    thrust::device_ptr<uint32_t> t_partitions(d_partitions);
    thrust::device_ptr<const uint32_t> t_partitions_sizes(d_partitions_sizes);
    thrust::device_ptr<const uint32_t> t_partitions_inbound_sizes(d_partitions_inbound_sizes);

    // enumerate all partition ids and extract the ones considered small
    thrust::device_vector<uint32_t> t_part_index(num_partitions); // part_index[pos] -> partition id at position pos
    thrust::sequence(t_part_index.begin(), t_part_index.end());
    thrust::device_vector<uint32_t> t_small_parts(num_partitions); // small_parts[pos] -> partition id of the pos-th small partition
    auto small_end = thrust::copy_if(t_part_index.begin(), t_part_index.end(), t_small_parts.begin(), [=] __host__ __device__ (uint32_t p) { return t_partitions_sizes[p] < SMALL_PART_MERGE_SIZE_THRESHOLD; });
    t_small_parts.resize(small_end - t_small_parts.begin());
    uint32_t smallest_part_size = thrust::reduce(t_partitions_sizes, t_partitions_sizes + num_partitions, UINT32_MAX, thrust::minimum<uint32_t>());
    INFO(cfg) std::cout << "Smallest partition size: " << smallest_part_size << "\n";
    if (!t_small_parts.empty()) {
        INFO(cfg) std::cout << "Partitions compression over " << t_small_parts.size() << " partitions ...\n";
        // sort the small partitions by increasing (size, inbound, id)
        thrust::stable_sort(
            t_small_parts.begin(), t_small_parts.end(),
            [=] __host__ __device__ (uint32_t a, uint32_t b) {
                uint32_t sa = t_partitions_sizes[a];
                uint32_t sb = t_partitions_sizes[b];
                if (sa != sb) return sa < sb;
                uint32_t ia = t_partitions_inbound_sizes[a];
                uint32_t ib = t_partitions_inbound_sizes[b];
                if (ia != ib) return ia < ib; return a < b;
            }
        );

        // gather the sorted sizes and inbound bounds that drive the greedy packing
        uint32_t num_small_parts = (uint32_t)t_small_parts.size();
        thrust::device_vector<dim_t> t_sorted_sizes(num_small_parts); // sorted_sizes[pos] -> size of the small partition at sorted position pos
        thrust::device_vector<dim_t> t_sorted_inbound(num_small_parts); // sorted_inbound[pos] -> inbound size of the small partition at sorted position pos
        thrust::gather(t_small_parts.begin(), t_small_parts.end(), t_partitions_sizes, t_sorted_sizes.begin());
        thrust::gather(t_small_parts.begin(), t_small_parts.end(), t_partitions_inbound_sizes, t_sorted_inbound.begin());

        // build exclusive-prefix buffers to query any greedy segment sum
        thrust::device_vector<dim_t> t_prefix_sizes(1 + num_small_parts); // prefix_sizes[pos] -> sum of sorted_sizes in [0, pos)
        thrust::device_vector<dim_t> t_prefix_inbound(1 + num_small_parts); // prefix_inbound[pos] -> sum of sorted_inbound in [0, pos)
        t_prefix_sizes[0] = 0;
        t_prefix_inbound[0] = 0;
        thrust::inclusive_scan(t_sorted_sizes.begin(), t_sorted_sizes.end(), t_prefix_sizes.begin() + 1);
        thrust::inclusive_scan(t_sorted_inbound.begin(), t_sorted_inbound.end(), t_prefix_inbound.begin() + 1);
        dim_t *d_prefix_sizes_ptr = thrust::raw_pointer_cast(t_prefix_sizes.data());
        dim_t *d_prefix_inbound_ptr = thrust::raw_pointer_cast(t_prefix_inbound.data());

        // compute the largest prefix values each greedy segment may reach from every start position
        thrust::device_vector<dim_t> t_size_targets(num_small_parts); // size_targets[pos] -> max allowed prefix size for a segment starting at pos
        thrust::device_vector<dim_t> t_inbound_targets(num_small_parts); // inbound_targets[pos] -> max allowed prefix inbound for a segment starting at pos
        thrust::transform(
            thrust::make_counting_iterator<uint32_t>(0), thrust::make_counting_iterator<uint32_t>(num_small_parts),
            t_size_targets.begin(),
            [=] __host__ __device__ (uint32_t i) { return d_prefix_sizes_ptr[i] + (dim_t)h_max_nodes_per_part; }
        );
        thrust::transform(
            thrust::make_counting_iterator<uint32_t>(0), thrust::make_counting_iterator<uint32_t>(num_small_parts),
            t_inbound_targets.begin(),
            [=] __host__ __device__ (uint32_t i) { return d_prefix_inbound_ptr[i] + (dim_t)h_max_inbound_per_part; }
        );

        // find the first violating prefix slot for each possible greedy segment start
        thrust::device_vector<uint32_t> t_size_next(num_small_parts); // size_next[pos] -> first prefix slot that violates the size cap from pos
        thrust::device_vector<uint32_t> t_inbound_next(num_small_parts); // inbound_next[pos] -> first prefix slot that violates the inbound cap from pos
        thrust::upper_bound(
            t_prefix_sizes.begin(), t_prefix_sizes.end(),
            t_size_targets.begin(), t_size_targets.end(),
            t_size_next.begin()
        );
        thrust::upper_bound(
            t_prefix_inbound.begin(), t_prefix_inbound.end(),
            t_inbound_targets.begin(), t_inbound_targets.end(),
            t_inbound_next.begin()
        );

        // convert the first violating prefix slot into the next greedy segment start
        thrust::device_vector<uint32_t> t_next_start(1 + num_small_parts, num_small_parts); // next_start[pos] -> next segment start after greedily packing from pos
        auto next_start_input_begin = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator<uint32_t>(0),
            t_size_next.begin(), t_inbound_next.begin()
        ));
        thrust::transform(
            next_start_input_begin,
            next_start_input_begin + num_small_parts,
            t_next_start.begin(),
            [] __host__ __device__ (const thrust::tuple<uint32_t, uint32_t, uint32_t>& x) {
                uint32_t pos = thrust::get<0>(x);
                uint32_t size_ub = thrust::get<1>(x);
                uint32_t inbound_ub = thrust::get<2>(x);
                return max(pos + 1u, min(size_ub, inbound_ub) - 1u);
            }
        );

        // build binary-lifting jump tables over the monotone next-start relation
        std::vector<thrust::device_vector<uint32_t>> t_jump; // jump[k][pos] -> applying next_start 2^k times from pos
        t_jump.emplace_back(t_next_start);
        while ((1u << (t_jump.size() - 1)) < num_small_parts) {
            thrust::device_vector<uint32_t> t_next_jump(1 + num_small_parts); // next_jump[pos] -> applying the current jump twice from pos
            thrust::gather(
                t_jump.back().begin(), t_jump.back().end(),
                t_jump.back().begin(), t_next_jump.begin()
            );
            t_jump.emplace_back(std::move(t_next_jump));
        }

        // recover, for each sorted position, how many greedy jumps are needed to reach it from zero
        thrust::device_vector<uint32_t> t_groups(num_small_parts, 0u); // groups[pos] -> greedy pack id of the small partition at sorted position pos
        thrust::device_vector<uint32_t> t_curr_pos(num_small_parts, 0u); // curr_pos[pos] -> current jump-chain position while binary lifting pos
        thrust::device_vector<uint32_t> t_sorted_pos(num_small_parts); // sorted_pos[pos] -> identity sorted position pos
        thrust::sequence(t_sorted_pos.begin(), t_sorted_pos.end());
        thrust::device_vector<uint32_t> t_candidate_pos(num_small_parts); // candidate_pos[pos] -> tentative jump destination at the current binary-lifting level
        uint32_t *d_groups_ptr = thrust::raw_pointer_cast(t_groups.data());
        uint32_t *d_curr_pos_ptr = thrust::raw_pointer_cast(t_curr_pos.data());
        uint32_t *d_sorted_pos_ptr = thrust::raw_pointer_cast(t_sorted_pos.data());
        uint32_t *d_candidate_pos_ptr = thrust::raw_pointer_cast(t_candidate_pos.data());
        for (int level = (int)t_jump.size() - 1; level >= 0; --level) {
            // try the current jump length and keep it whenever it does not pass the queried position
            thrust::gather(
                t_curr_pos.begin(), t_curr_pos.end(),
                t_jump[level].begin(), t_candidate_pos.begin()
            );
            uint32_t step = 1u << level;
            thrust::for_each_n(
                thrust::make_counting_iterator<uint32_t>(0), num_small_parts,
                [=] __device__ (uint32_t i) {
                    uint32_t candidate = d_candidate_pos_ptr[i];
                    if (candidate <= d_sorted_pos_ptr[i]) {
                        d_curr_pos_ptr[i] = candidate;
                        d_groups_ptr[i] += step;
                    }
                }
            );
        }

        // map each greedy pack to the lowest original partition id it contains
        thrust::device_vector<uint32_t> t_rep_ids(t_groups.size()); // rep_ids[pack] -> representative original partition id of that greedy pack
        auto rep_end = thrust::reduce_by_key(
            t_groups.begin(), t_groups.end(), t_small_parts.begin(),
            thrust::make_discard_iterator(), t_rep_ids.begin(),
            thrust::equal_to<uint32_t>(), thrust::minimum<uint32_t>()
        );
        t_rep_ids.resize(rep_end.second - t_rep_ids.begin());

        // build the partition-id remap induced by the greedy packing
        thrust::device_vector<uint32_t> pid_map(num_partitions); // pid_map[p] -> representative partition id that p is merged into
        thrust::sequence(pid_map.begin(), pid_map.end());
        thrust::device_vector<uint32_t> new_pids(t_small_parts.size()); // new_pids[pos] -> representative id of the small partition at sorted position pos
        thrust::gather(t_groups.begin(), t_groups.end(), t_rep_ids.begin(), new_pids.begin());
        thrust::scatter(new_pids.begin(), new_pids.end(), t_small_parts.begin(), pid_map.begin());

        // rewrite each node's partition id through the greedy partition remap.
        uint32_t *d_pid_map_ptr = thrust::raw_pointer_cast(pid_map.data());
        thrust::transform(
            t_partitions, t_partitions + num_nodes, t_partitions,
            [=] __host__ __device__ (uint32_t p) { return d_pid_map_ptr[p]; }
        );
    } else {
        INFO(cfg) std::cout << "Partitions compression not performed ...\n";
    }
}