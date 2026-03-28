#include <tuple>
#include <set>

#include "thruster.cuh"

#include "runconfig.hpp"

#include "utils.cuh"
#include "defines.cuh"
#include "chaining.cuh"
#include "coarsening.cuh"

void candidatesProposal(
    const runconfig cfg,
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
        std::cout << "Running candidates kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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

void logCandidates(
    const runconfig cfg,
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
            if (pairs_tmp[target * cfg.candidates_count + j] != i && pairs_tmp[target * cfg.candidates_count + j] != UINT32_MAX && std::find(pairs_tmp.begin() + target * cfg.candidates_count, pairs_tmp.begin() + target * cfg.candidates_count + j, i) == pairs_tmp.begin() + target * cfg.candidates_count + j && !(scores_tmp[target * cfg.candidates_count + j] > score || scores_tmp[target * cfg.candidates_count + j] == score && pairs_tmp[target * cfg.candidates_count + j] < i))
                std::cerr << "\n  WARNING, symmetry violated: node " << i << " (" << j << " target=" << target << " score=" << std::fixed << std::setprecision(3) << score << ") AND node " << target << " (" << j << " target=" << pairs_tmp[target * cfg.candidates_count + j] << " score=" << std::fixed << std::setprecision(3) << scores_tmp[target * cfg.candidates_count + j] << ") !!";
        }
    }
    std::cout << "\n";
    for (uint32_t j = 0; j < cfg.candidates_count; ++j)
        std::cout << "Candidates count (" << j << "): " << candidates_count[j].size() << "\n";
}

std::tuple<uint32_t, uint32_t*, uint32_t*, uint32_t*, dim_t*> groupNodes(
    const runconfig cfg,
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
            std::cout << "NOTE: grouping kernel required blocks=" << blocks << ", but max-blocks=" << max_blocks << ", setting repeats=" << num_repeats << " ...\n";
            blocks = (blocks + num_repeats - 1) / num_repeats;
            if (num_repeats > MAX_REPEATS) {
                std::cout << "ABORTING: grouping kernel required repeats=" << num_repeats << ", but max-repeats=" << MAX_REPEATS << " !!\n";
                abort();
            }
        }
        // launch - grouping kernel
        std::cout << "Running grouping kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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

void logGroups(
    const runconfig cfg,
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
