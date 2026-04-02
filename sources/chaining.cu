#include <algorithm>

#include "thruster.cuh"

#include "chaining.cuh"

#include "utils.cuh"
#include "constants.cuh"


// given a set of src->dst pairs, each with a size and a weight, try to construct
// subsequences of pairs with similar size and highest total weight such that each's dst is
// the src of the next pair, stopping upon forming a cycle. The concatenation of subsequences
// by descending weight is then the final sequence returned.
// => deterministic tie-breaking is always based on the pair's idx (aka node/move idx)
void chaining(
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

    std::cout << "Running chaining kernels (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";

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
    const uint32_t* d_nodes_sizes,
    const uint32_t* d_inbound_count,
    const uint32_t* d_pairs,
    const uint32_t curr_num_nodes,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    const uint32_t candidates_count,
    uint32_t* d_groups // pre-initialized -> this routine writes group ids for paired nodes only.
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
    #if VERBOSE
    std::cout << "Orphans nodes found: " << num_free << "\n";
    #endif
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
    std::cout << "Running pair orphans kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
