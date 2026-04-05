#include <set>
#include <tuple>
#include <vector>

#include "thruster.cuh"

#include "runconfig.hpp"

#include "coarsening.cuh"

#include "utils.cuh"
#include "defines.cuh"
#include "chaining.cuh"

using namespace config;

void candidatesProposal(
    const runconfig &cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_srcs_count,
    const uint32_t *d_neighbors,
    const dim_t *d_neighbors_offsets,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t *d_inbound_count,
    const float *d_hedge_weights,
    const uint32_t *d_nodes_sizes,
    const uint32_t curr_num_nodes,
    uint32_t *d_pairs,
    uint32_t *d_u_scores
) {
    // zero-out candidates kernel's outputs
    CUDA_CHECK(cudaMemset(d_pairs, 0xFF, curr_num_nodes * sizeof(uint32_t) * cfg.candidates_count)); // 0xFF -> UINT32_MAX
    // NOTE: no need to init. "d_u_scores" if we use "d_pairs" to see which locations are valid
    
    {
        // launch configuration - candidates kernel
        // NOTE: choose threads_per_block multiple of WARP_SIZE
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = curr_num_nodes ; // 1 warp per node
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        size_t bytes_per_warp = 3 * HIST_SIZE * sizeof(uint32_t);
        size_t shared_bytes = warps_per_block * bytes_per_warp;
        // launch - candidates kernel
        LAUNCH(cfg) << "candidates kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        candidates_kernel<<<blocks, threads_per_block, shared_bytes>>>(
            d_hedges,
            d_hedges_offsets,
            d_srcs_count,
            d_neighbors,
            d_neighbors_offsets,
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            d_hedge_weights,
            d_nodes_sizes,
            curr_num_nodes,
            cfg.candidates_count,
            d_pairs,
            d_u_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

std::tuple<uint32_t, uint32_t*, uint32_t*, uint32_t*, dim_t*> groupNodes(
    const runconfig &cfg,
    const cudaDeviceProp props,
    const uint32_t *d_inbound_count,
    const uint32_t *d_pairs,
    const uint32_t *d_u_scores,
    const uint32_t *d_nodes_sizes,
    const uint32_t curr_num_nodes,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    slot *d_slots,
    dp_score *d_dp_scores
) {
    // zero-out grouping kernel's outputs
    slot init_slot; init_slot.id = UINT32_MAX; init_slot.score = 0u;
    thrust::device_ptr<slot> d_slots_ptr(d_slots);
    thrust::fill(d_slots_ptr, d_slots_ptr + curr_num_nodes, init_slot); // upper 32 bits to 0x00, lower 32 to 0xFF
    CUDA_CHECK(cudaMemset(d_dp_scores, 0x00, curr_num_nodes * sizeof(dp_score)));
    
    // prepare this level's coarsening groups
    uint32_t *d_groups = nullptr; // groups[node idx] -> node's group id (zero-based)
    CUDA_CHECK(cudaMalloc(&d_groups, curr_num_nodes * sizeof(uint32_t)));

    // launch configuration - grouping kernel
    {
        int threads_per_block = 256;
        int num_threads_needed = curr_num_nodes; // 1 thread per node
        int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        size_t bytes_per_thread = 0; //TODO
        size_t shared_bytes = threads_per_block * bytes_per_thread;
        // additional checks for the cooperative kernel mode
        int blocks_per_SM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_SM, grouping_kernel, threads_per_block, shared_bytes);
        int max_blocks = blocks_per_SM * props.multiProcessorCount;
        uint32_t num_repeats = 1;
        if (blocks > max_blocks) {
            num_repeats = (blocks + max_blocks - 1) / max_blocks;
            LOG(cfg) std::cout << "NOTE: grouping kernel required blocks=" << blocks << ", but max-blocks=" << max_blocks << ", setting repeats=" << num_repeats << " ...\n";
            blocks = (blocks + num_repeats - 1) / num_repeats;
            if (num_repeats > MAX_MATCHING_REPEATS) {
                ERR(cfg) std::cerr << "ABORTING: grouping kernel required repeats=" << num_repeats << ", but max-repeats=" << MAX_MATCHING_REPEATS << " !!\n";
                abort();
            }
        }
        // launch - grouping kernel
        LAUNCH(cfg) << "grouping kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        void *kernel_args[] = {
            (void*)&d_pairs,
            (void*)&d_u_scores,
            (void*)&d_nodes_sizes,
            (void*)&curr_num_nodes,
            (void*)&num_repeats,
            (void*)&cfg.candidates_count,
            (void*)&d_slots,
            (void*)&d_dp_scores,
            (void*)&d_groups
        };
        cudaLaunchCooperativeKernel((void*)grouping_kernel, blocks, threads_per_block, kernel_args, shared_bytes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // for nodes that have no candidate and are left alone (no -valid- neighbors), try to pair them up with each other as to create as many groups as possible
    // => impose the sum of sizes and the sum of inbound set sizes < constraints
    // => the idea is to try pairs among non-neighbors, therefore the inbound set size intersection can already be taken as empty (hence, sum set sizes)
    build_orphan_pairs(
        cfg,
        d_nodes_sizes,
        d_inbound_count,
        d_pairs,
        curr_num_nodes,
        h_max_nodes_per_part,
        h_max_inbound_per_part,
        cfg.candidates_count,
        d_groups
    );

    // prepare uncoarsening map (node ids sorted by group id)
    uint32_t *d_ungroups = nullptr; // ungroups[ungroups_offsets[group id] + i] -> the group's i-th node (its original idx)
    CUDA_CHECK(cudaMalloc(&d_ungroups, curr_num_nodes * sizeof(uint32_t)));

    // order groups kernel (parallel label compression)
    // as of now "d_groups" contains the new non-zero-based group id for every node
    thrust::device_ptr<uint32_t> t_nodes(d_ungroups);
    thrust::sequence(t_nodes, t_nodes + curr_num_nodes);
    // sort by groups, carrying node indices (represented by the sequence) along; after d_groups is sorted, t_nodes tells where each sorted element came from
    thrust::device_ptr<uint32_t> t_groups(d_groups);
    thrust::sort_by_key(t_groups, t_groups + curr_num_nodes, t_nodes); // sort groups and carry indices along for a ride
    // build "head of group flags": 1 at first occurrence of each group in the sorted array, 0 otherwise ( flags[i] = 1 if i == 0 or d_groups[i] != d_groups[i-1] )
    thrust::device_vector<uint32_t> t_headflags(curr_num_nodes);
    // the first element is part of groups zero (the initial default)
    t_headflags[0] = 0;
    thrust::transform(t_groups + 1, t_groups + curr_num_nodes, t_groups, t_headflags.begin() + 1, [] __device__ (uint32_t curr, uint32_t prev) { return curr != prev ? 1u : 0u; });
    // the prefix sum of head flags gives the new group id per element (w.r.t. the sorted order) ( new_id[i] = number of heads before position i )
    thrust::inclusive_scan(t_headflags.begin(), t_headflags.end(), t_headflags.begin()); // in-place
    // the last flag, after the scan, gives you the total number of distinct groups
    const uint32_t new_num_nodes = t_headflags.back() + 1;
    // ======================================
    // prepare this level's cumulative groups sizes
    // NOTE: "node sizes" = size of the nodes that entered this level, "group sizes" = cumulative size of groups constructed on this level
    uint32_t *d_groups_sizes = nullptr; // group_sizes[group id] = sum of sizes of all nodes in that group
    CUDA_CHECK(cudaMalloc(&d_groups_sizes, new_num_nodes * sizeof(uint32_t)));
    // extra step: compute cumulative group sizes
    thrust::device_ptr<uint32_t> t_groups_sizes(d_groups_sizes);
    thrust::device_ptr<const uint32_t> t_nodes_sizes(d_nodes_sizes);
    // premute node sizes in "sorted-by-group" order, using indices that already reflect such ordering ( t_nodes[i] tells which original idx got sorted in position i )
    auto nodes_sizes_values_begin = thrust::make_permutation_iterator(t_nodes_sizes, t_nodes);
    // reduce (sum) nodes_sizes inside each group (group = key, marked by having the same headflag, that by now corresponds to the zero-based group id) ( headflags[i] is the new group ID for sorted position i )
    thrust::reduce_by_key(t_headflags.begin(), t_headflags.end(), nodes_sizes_values_begin, thrust::make_discard_iterator(), t_groups_sizes);
    // => now "d_groups_sizes[idx]" holds the sum of nodes_size over all nodes in group idx, for idx in [0, new_num_nodes)
    // ======================================
    // scatter the new ids back to original positions using the sequence; for sorted position i, original index is t_nodes[i]; we want: d_groups[t_nodes[i]] = t_headflags[i]
    thrust::scatter(t_headflags.begin(), t_headflags.end(), t_nodes, t_groups);
    // if the number of groups has reached the required threshold, they become the partitions
    // => now "d_groups[idx]" contains the new zero-based group ID for every node

    // prepare uncoarsening map offsets (offset where each ungroup starts)
    dim_t *d_ungroups_offsets = nullptr; // ungroups_offsets[node idx] -> node's group id (zero-based)
    CUDA_CHECK(cudaMalloc(&d_ungroups_offsets, (1 + new_num_nodes) * sizeof(dim_t)));
    
    // build reverse multifunction from groups to their original nodes
    // from above, t_nodes is the list of node idxs sorted by their group id, hence, the reverse list is simply t_nodes, we just need to compute the offsets to reach, from each group id, its original nodes
    thrust::device_ptr<dim_t> t_ungroups_offsets(d_ungroups_offsets);
    // predicate to detect group starts: is_group_start(i) = (i == 0) || (headflags[i] != headflags[i-1])
    auto is_group_start = [heads = t_headflags.begin()] __device__ (uint32_t i) { return (i == 0) || (heads[i] != heads[i - 1]); };
    // counting iterator over sorted positions
    auto t_iter_begin = thrust::make_counting_iterator<uint32_t>(0);
    auto t_iter_end = thrust::make_counting_iterator<uint32_t>(curr_num_nodes);
    // copy positions of (only) group starts directly into ungroups_offsets
    thrust::copy_if(t_iter_begin, t_iter_end, t_iter_begin, t_ungroups_offsets, is_group_start);
    // append the (curr_num_nodes + 1)-th value
    dim_t dim_t_curr_num_nodes = (dim_t)curr_num_nodes;
    CUDA_CHECK(cudaMemcpy(d_ungroups_offsets + new_num_nodes, &dim_t_curr_num_nodes, sizeof(dim_t), cudaMemcpyHostToDevice));

    return std::make_tuple(new_num_nodes, d_groups, d_groups_sizes, d_ungroups, d_ungroups_offsets);
}

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
    thrust::transform(
        t_size_next.begin(), t_size_next.end(),
        t_inbound_next.begin(),
        t_next_start.begin(),
        [] __host__ __device__ (uint32_t size_ub, uint32_t inbound_ub) {
            // the upper_bound-s returned the first prefix positions that violate constraint;
            // the next greedy pack starts at the last valid exclusive end, i.e. ub - 1
            return min(size_ub, inbound_ub) - 1u;
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


// LOGGING

void logCandidates(
    const runconfig &cfg,
    const uint32_t *d_pairs,
    const uint32_t *d_u_scores,
    const uint32_t curr_num_nodes
) {
    std::vector<uint32_t> pairs_tmp(curr_num_nodes * cfg.candidates_count);
    std::vector<uint32_t> scores_tmp(curr_num_nodes * cfg.candidates_count);
    std::vector<std::set<uint32_t>> candidates_count(cfg.candidates_count);
    CUDA_CHECK(cudaMemcpy(pairs_tmp.data(), d_pairs, curr_num_nodes * sizeof(uint32_t) * cfg.candidates_count, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scores_tmp.data(), d_u_scores, curr_num_nodes * sizeof(uint32_t) * cfg.candidates_count, cudaMemcpyDeviceToHost));
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
            if (
                pairs_tmp[target * cfg.candidates_count + j] != i && pairs_tmp[target * cfg.candidates_count + j] != UINT32_MAX
                && std::find(pairs_tmp.begin() + target * cfg.candidates_count, pairs_tmp.begin() + target * cfg.candidates_count + j, i) == pairs_tmp.begin() + target * cfg.candidates_count + j
                && !(scores_tmp[target * cfg.candidates_count + j] > score || scores_tmp[target * cfg.candidates_count + j] == score && pairs_tmp[target * cfg.candidates_count + j] < i)
            ) {
                std::cerr
                    << "\n  WARNING, symmetry violated: node " << i
                    << " (" << j << " target=" << target << " score=" << std::fixed << std::setprecision(3) << score
                    << ") AND node " << target << " (" << j << " target=" << pairs_tmp[target * cfg.candidates_count + j]
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
    const uint32_t *d_pairs,
    const uint32_t *d_groups,
    const uint32_t *d_groups_sizes,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes,
    const uint32_t h_max_nodes_per_part
) {
    std::vector<uint32_t> pairs_tmp(curr_num_nodes * cfg.candidates_count);
    std::vector<uint32_t> groups_tmp(curr_num_nodes);
    std::vector<uint32_t> groups_sizes_tmp(new_num_nodes);
    CUDA_CHECK(cudaMemcpy(pairs_tmp.data(), d_pairs, curr_num_nodes * sizeof(uint32_t) * cfg.candidates_count, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(groups_tmp.data(), d_groups, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(groups_sizes_tmp.data(), d_groups_sizes, new_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::unordered_map<uint32_t, int> groups_count;
    std::cout << "Grouping results:\n";
    for (uint32_t i = 0; i < curr_num_nodes; ++i) {
        uint32_t group = groups_tmp[i];
        uint32_t group_size = groups_sizes_tmp[group];
        groups_count[group]++;
        if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH)) {
            std::cout << "  node " << i << " ->";
            for (uint32_t j = 0; j < cfg.candidates_count; ++j) {
                uint32_t target = pairs_tmp[i * cfg.candidates_count + j];
                if (target == UINT32_MAX) std::cout << " (" << j << " target=none)";
                else std::cout << " (" << j << " target=" << target << ")";
            }
            std::cout << " group=" << group << " group_size=" << group_size << "\n";
        }
    }
    long long max_gs = 0, sum_gs = 0;
    for (uint32_t i = 0; i < new_num_nodes; ++i) {
        uint32_t group_size = groups_sizes_tmp[i];
        sum_gs += group_size;
        if (group_size > max_gs) max_gs = group_size;
        if (group_size > h_max_nodes_per_part)
            std::cerr << "  WARNING, max group size constraint (" << h_max_nodes_per_part << ") violated by group=" << i << " with group_size=" << group_size << " !!\n";
    }
    long long max_cgs = 0, sum_cgs = 0;
    for (const auto& [group, count] : groups_count) {
        sum_cgs += count;
        if (count > max_cgs) max_cgs = count;
    }
    std::cout << "Groups count: " << groups_count.size() << "\n  Max coarse group size: " << max_cgs << ", Avg coarse group size: " << std::fixed << std::setprecision(2) << (float)sum_cgs/groups_count.size() << "\n";
    std::cout << "  Max nodes group size: " << max_gs << ", Avg nodes group size: " << std::fixed << std::setprecision(2) << (float)sum_gs/groups_count.size() << "\n";
}
