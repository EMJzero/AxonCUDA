#include <vector>
#include <algorithm>

#include "thruster.cuh"

#include "runconfig.hpp"

#include "chaining.cuh"

#include "utils.cuh"
#include "constants.cuh"


// given a set of src->dst pairs, each with a size and a weight, try to construct
// subsequences of pairs with similar size and highest total weight such that each's dst is
// the src of the next pair, stopping upon forming a cycle. The concatenation of subsequences
// by descending weight is then the final sequence returned.
// => deterministic tie-breaking is always based on the pair's idx (aka node/move idx)
void chaining(
    const runconfig &cfg,
    const uint32_t* d_srcs_og,
    const uint32_t* d_dsts_og,
    const uint32_t* d_size_og, // node sizes
    //const uint32_t* icnt, // inbound set sizes
    const float* d_weights_og,
    const uint32_t num_edges,
    uint32_t* sequence_idx
) {
    if (num_edges == 0) return;

    const int num_threads_needed = (int)num_edges;
    const int threads_per_block = 256;
    const int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;

    if (cfg.verbose_kernel_launches) std::cout << "Running chaining kernels (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";

    uint32_t *d_srcs = nullptr; // srcs[idx] -> source partition / source node of the idx-th pair
    uint32_t *d_dsts = nullptr; // dsts[idx] -> destination partition / destination node of the idx-th pair
    uint32_t *d_size = nullptr; // size[idx] -> size of the idx-th pair's moving node
    float *d_w = nullptr; // w[idx] -> weight / gain of the idx-th pair
    uint32_t *d_orig = nullptr; // orig[idx] -> original idx of the idx-th pair after reordering
    src_negweight_key *d_src_keys = nullptr; // src_keys[idx] -> (src, -weight) sort key of the idx-th pair
    int *d_out_begin = nullptr; // out_begin[idx] -> begin of the outgoing candidates range of the idx-th pair
    int *d_out_end = nullptr; // out_end[idx] -> end of the outgoing candidates range of the idx-th pair
    uint32_t *d_next = nullptr; // next[idx] -> chosen successor pair of the idx-th pair, later reused as comp[idx]
    uint32_t *d_prev = nullptr; // prev[idx] -> chosen predecessor pair of the idx-th pair, later reused as comp_key[idx] and comp_to_rank[idx]
    uint32_t *d_succ_choice = nullptr; // succ_choice[idx] -> proposed successor of the idx-th pair, later reused as pos[idx], comp_count[idx], edge_id[idx]
    float *d_succ_score = nullptr; // succ_score[idx] -> score of the proposed successor of the idx-th pair, later reused as w_val[idx]
    uint64_t *d_best_claim = nullptr; // best_claim[idx] -> packed best predecessor claim received by successor idx
    uint32_t *d_counts = nullptr; // counts[idx] -> scratch uint32 buffer, reused as one[idx], unique_comp[idx], comp_id_rank[idx]
    float *d_weights = nullptr; // weights[idx] -> scratch float buffer, reused as comp_wsum[idx]
    edge_order_key *d_edge_keys = nullptr; // edge_keys[idx] -> final ordering key of the idx-th pair

    CUDA_CHECK(cudaMalloc(&d_srcs, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dsts, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_size, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_w, num_edges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_orig, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_src_keys, num_edges * sizeof(src_negweight_key)));
    CUDA_CHECK(cudaMalloc(&d_out_begin, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_end, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_prev, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_succ_choice, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_succ_score, num_edges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_claim, num_edges * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_weights, num_edges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edge_keys, num_edges * sizeof(edge_order_key)));

    thrust::device_ptr<uint32_t> t_srcs(d_srcs);
    thrust::device_ptr<uint32_t> t_dsts(d_dsts);
    thrust::device_ptr<uint32_t> t_size(d_size);
    thrust::device_ptr<float> t_w(d_w);
    thrust::device_ptr<uint32_t> t_orig(d_orig);
    thrust::device_ptr<src_negweight_key> t_src_keys(d_src_keys);
    thrust::device_ptr<int> t_out_begin(d_out_begin);
    thrust::device_ptr<int> t_out_end(d_out_end);
    thrust::device_ptr<edge_order_key> t_edge_keys(d_edge_keys);

    // aliases
    uint32_t *d_comp = d_next; // comp[idx] -> representative component id of the idx-th pair
    uint32_t *d_comp_key = d_prev; // comp_key[idx] -> component id values reordered for reductions
    uint32_t *d_comp_to_rank = d_prev; // comp_to_rank[comp id] -> rank of that component in the final chain ordering
    uint32_t *d_pos = d_succ_choice; // pos[idx] -> position of the idx-th pair inside its chain
    uint32_t *d_comp_count = reinterpret_cast<uint32_t*>(d_src_keys); // comp_count[idx] -> number of pairs in the idx-th unique component
    uint32_t *d_edge_id = d_succ_choice; // edge_id[idx] -> pair idx in sorted-space order
    float *d_w_val = d_succ_score; // w_val[idx] -> pair weights reordered alongside comp_key for reduce_by_key
    uint32_t *d_one = reinterpret_cast<uint32_t*>(d_out_begin); // one[idx] -> constant 1, used to count pairs per component
    uint32_t *d_unique_comp = reinterpret_cast<uint32_t*>(d_out_end); // unique_comp[idx] -> unique component id produced by reduce_by_key
    uint32_t *d_comp_id_rank = d_counts; // comp_id_rank[idx] -> component id that occupies the idx-th rank
    thrust::device_ptr<uint32_t> t_comp_key(d_prev); // comp_key[idx] -> component id values reordered for reductions
    thrust::device_ptr<uint32_t> t_comp_count(d_comp_count); // comp_count[idx] -> number of pairs in the idx-th unique component
    thrust::device_ptr<uint32_t> t_edge_id(d_succ_choice); // edge_id[idx] -> pair idx in sorted-space order
    thrust::device_ptr<float> t_w_val(d_succ_score); // w_val[idx] -> pair weights reordered alongside comp_key for reduce_by_key
    thrust::device_ptr<uint32_t> t_one(d_one); // one[idx] -> constant 1, used to count pairs per component
    thrust::device_ptr<uint32_t> t_unique_comp(d_unique_comp); // unique_comp[idx] -> unique component id produced by reduce_by_key
    thrust::device_ptr<uint32_t> t_comp_id_rank(d_counts); // comp_id_rank[idx] -> component id that occupies the idx-th rank
    thrust::device_ptr<float> t_comp_wsum(d_weights); // comp_wsum[idx] -> total weight of the idx-th unique component

    CUDA_CHECK(cudaMemcpy(d_srcs, d_srcs_og, num_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_dsts, d_dsts_og, num_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_size, d_size_og, num_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    //CUDA_CHECK(cudaMemcpy(d_icnt, icnt, num_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, d_weights_og, num_edges * sizeof(float), cudaMemcpyDeviceToDevice));

    thrust::sequence(t_orig, t_orig + num_edges);

    // sort edges by (src, -weight)
    build_src_keys<<<blocks, threads_per_block>>>(
        num_edges,
        d_srcs,
        d_w,
        d_src_keys
    );
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(t_srcs, t_dsts, t_size, t_w, t_orig));
    //auto zipped = thrust::make_zip_iterator(thrust::make_tuple(t_srcs, t_dsts, t_size, t_icnt, t_w, t_orig));
    thrust::stable_sort_by_key(t_src_keys, t_src_keys + num_edges, zipped);

    // for each edge i, its outgoing candidate list is OUT[dst[i]] = edges whose src == dst[i]
    thrust::lower_bound(t_srcs, t_srcs + num_edges, t_dsts, t_dsts + num_edges, t_out_begin);
    thrust::upper_bound(t_srcs, t_srcs + num_edges, t_dsts, t_dsts + num_edges, t_out_end);

    // chaining state
    CUDA_CHECK(cudaMemset(d_next, 0xFF, num_edges * sizeof(uint32_t))); // UINT32_MAX -> no successor chosen yet
    CUDA_CHECK(cudaMemset(d_prev, 0xFF, num_edges * sizeof(uint32_t))); // UINT32_MAX -> no predecessor chosen yet

    // multi-iteration greedy build
    for (int it = 0; it < CHAIN_ITERS; ++it) {
        CUDA_CHECK(cudaMemset(d_best_claim, 0x00, num_edges * sizeof(uint64_t))); // 0 -> no predecessor claimed this successor yet

        propose_successor<<<blocks, threads_per_block>>>(
            num_edges,
            d_dsts,
            d_size,
            //d_icnt,
            d_w,
            d_out_begin,
            d_out_end,
            d_prev,
            d_next,
            CHAIN_WINDOW,
            CHAIN_ALPHA,
            d_succ_choice,
            d_succ_score
        );

        resolve_successor_conflicts<<<blocks, threads_per_block>>>(
            num_edges,
            d_succ_choice,
            d_succ_score,
            d_prev,
            d_best_claim
        );

        commit_links<<<blocks, threads_per_block>>>(
            num_edges,
            d_next,
            d_prev,
            d_best_claim
        );
    }

    // component id and position
    compute_comp_and_pos<<<blocks, threads_per_block>>>(
        num_edges,
        d_prev,
        d_comp,
        d_pos
    );

    CUDA_CHECK(cudaMemcpy(d_comp_key, d_comp, num_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_val, d_w, num_edges * sizeof(float), cudaMemcpyDeviceToDevice));

    thrust::fill(t_one, t_one + num_edges, 1u);

    thrust::sort_by_key(t_comp_key, t_comp_key + num_edges, thrust::make_zip_iterator(thrust::make_tuple(t_w_val, t_one)));

    auto end1 = thrust::reduce_by_key(
        t_comp_key, t_comp_key + num_edges,
        t_w_val, t_unique_comp,
        t_comp_wsum
    );
    int num_components = (int)(end1.first - t_unique_comp);

    auto end2 = thrust::reduce_by_key(
        t_comp_key, t_comp_key + num_edges,
        t_one, t_unique_comp, // overwriting is ok (same keys)
        t_comp_count
    );
    int num_components_2 = (int)(end2.first - t_unique_comp);
    if (num_components_2 < num_components) num_components = num_components_2;
    thrust::device_ptr<uint32_t> t_comp_idx = t_comp_key; // comp_idx[idx] -> component rank candidates sorted by descending component weight
    thrust::sequence(t_comp_idx, t_comp_idx + num_components);

    thrust::sort(t_comp_idx, t_comp_idx + num_components,
        [wsum = d_weights] __device__ (uint32_t a, uint32_t b) {
            float wa = wsum[a];
            float wb = wsum[b];
            if (wa > wb) return true;
            if (wa < wb) return false;
            return a < b;
        }
    );

    thrust::gather(t_comp_idx, t_comp_idx + num_components, t_unique_comp, t_comp_id_rank);

    CUDA_CHECK(cudaMemset(d_comp_to_rank, 0x00, num_edges * sizeof(uint32_t)));

    thrust::for_each_n(
        thrust::make_counting_iterator(0), num_components,
        [comp_id_rank = d_comp_id_rank, comp_to_rank = d_comp_to_rank] __device__ (int r) {
            uint32_t c = comp_id_rank[r];
            comp_to_rank[c] = (uint32_t)r;
        }
    );

    // build per-edge sort keys (comp_rank, pos, edge_id)
    build_edge_order_keys<<<blocks, threads_per_block>>>(
        num_edges,
        d_comp,
        d_pos,
        d_comp_to_rank,
        d_edge_keys
    );

    thrust::sequence(t_edge_id, t_edge_id + num_edges);
    thrust::sort_by_key(t_edge_keys, t_edge_keys + num_edges, t_edge_id);

    // assign final sequence_idx = global position w.r.t. that sorted order
    scatter_sequence_idx<<<blocks, threads_per_block>>>(
        num_edges,
        d_edge_id,
        d_orig,
        sequence_idx
    );

    CUDA_CHECK(cudaFree(d_edge_keys));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_best_claim));
    CUDA_CHECK(cudaFree(d_succ_score));
    CUDA_CHECK(cudaFree(d_succ_choice));
    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_out_end));
    CUDA_CHECK(cudaFree(d_out_begin));
    CUDA_CHECK(cudaFree(d_src_keys));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_size));
    CUDA_CHECK(cudaFree(d_dsts));
    CUDA_CHECK(cudaFree(d_srcs));
    CUDA_CHECK(cudaFree(d_orig));
}

// given a set of pairs proposed between nodes (d_pairs), isolate nodes without a pair,
// try to force them into a pair with another node in the same condition such that their
// combined size and inbound set cardinality are within constraints. The objective is an
// almost-maximal number of formed pairs.
void build_orphan_pairs(
    const runconfig &cfg,
    const uint32_t *d_nodes_sizes,
    const uint32_t *d_inbound_count,
    const uint32_t *d_pairs,
    const uint32_t curr_num_nodes,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    const uint32_t candidates_count,
    uint32_t *d_groups // pre-initialized -> this routine writes group ids for paired nodes only.
) {
    thrust::device_ptr<const uint32_t> t_pairs(d_pairs);

    uint32_t *d_free_indices = nullptr; // free_indices[idx] -> idx-th orphan node, later sorted by (size, idx)
    uint64_t *d_free_keys = nullptr; // free_keys[idx] -> deterministic sort key (size << 32) | idx of the idx-th orphan node

    CUDA_CHECK(cudaMalloc(&d_free_indices, curr_num_nodes * sizeof(uint32_t)));
    thrust::device_ptr<uint32_t> t_free_indices(d_free_indices);

    auto idx_begin = thrust::counting_iterator<uint32_t>(0);
    auto idx_end = thrust::counting_iterator<uint32_t>(curr_num_nodes);

    // copy_if from 0..curr_num_nodes into d_free_indices
    auto out_it = thrust::copy_if(
        idx_begin, idx_end, t_free_indices,
        [t_pairs, candidates_count] __device__ (uint32_t i) {
            return t_pairs[i * candidates_count] == UINT32_MAX;
        }
    );

    uint32_t num_free = (uint32_t)(out_it - t_free_indices);
    LOG(cfg) std::cout << "Orphans nodes found: " << num_free << "\n";
    if (num_free < 2) {
        CUDA_CHECK(cudaFree(d_free_indices));
        return;
    }

    CUDA_CHECK(cudaMalloc(&d_free_keys, num_free * sizeof(uint64_t)));
    thrust::device_ptr<uint64_t> t_free_keys(d_free_keys);

    thrust::transform(
        t_free_indices, t_free_indices + num_free, t_free_keys,
        [d_nodes_sizes] __device__ (uint32_t idx) -> uint64_t {
            uint64_t s = (uint64_t)d_nodes_sizes[idx];
            return (s << 32) | (uint64_t)idx;
        }
    );

    thrust::sort_by_key(t_free_keys, t_free_keys + num_free, t_free_indices);

    int threads_per_block = 256;
    int num_threads_needed = num_free / 2; // 1 thread per one-in-two free nodes
    int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
    LAUNCH(cfg) << "pair orphans kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    pair_kth_smallest_with_kth_largest<<<blocks, threads_per_block>>>(
        d_free_indices,
        num_free,
        d_nodes_sizes,
        d_inbound_count,
        h_max_nodes_per_part,
        h_max_inbound_per_part,
        d_groups
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_free_keys));
    CUDA_CHECK(cudaFree(d_free_indices));
}

// given a set of partitions, try to merge together smaller the ones, within constraints
// => reduce for free the total number of partitions, where feasible
// NOTE: this does NOT update "partitions_sizes" and "partitions_inbound_sizes" !!
void merge_small_partitions(
    const runconfig &cfg,
    const uint32_t *d_partitions_sizes,
    const uint32_t *d_partitions_inbound_sizes,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    uint32_t *d_partitions
) {
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
        thrust::transform(
            t_size_next.begin(), t_size_next.end(),
            t_inbound_next.begin(), t_next_start.begin(),
            [] __host__ __device__ (uint32_t size_ub, uint32_t inbound_ub) {
                // upper_bound returns the first violating prefix slot, the next greedy pack -> must start at the previous exclusive end
                return min(size_ub, inbound_ub) - 1u;
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
