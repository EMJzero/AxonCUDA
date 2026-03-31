#include <stdint.h>
#include <algorithm>

#include <cuda_runtime.h>
#include <curand.h>

#include "../headers/thruster.cuh"

#include "../headers/utils.cuh"
#include "utils.cuh"

#define LABELPROP_REPEATS 16


// for every node, map its partition to to p*2 or p*2+1 depending on its position in the sorted (by rnd value) array
__global__
void split_partitions_kernel(
    const uint32_t* __restrict__ part_offsets,
    const uint32_t num_nodes,
    uint32_t* __restrict__ partitions // output in sorted order
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t part = partitions[tid];
    uint32_t part_start = part_offsets[part];
    uint32_t part_end = part_offsets[part + 1];
    uint32_t part_size = part_end - part_start;
    uint32_t rank = tid - part_start;

    // if you are ranked < ceil(count/2), go to 2*p, otherwise to 2*p+1
    // any odd element lands to p*2
    uint32_t left_size = (part_size + 1u) / 2u;
    partitions[tid] = (rank < left_size) ? (2u * part) : (2u * part + 1u);
}















// find in which partition (between a pair that was just bisected) each node wants to stay in
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
// SHUFFLES OVER: d
__global__
void label_propagation_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions, // partitions[idx] -> the partition node idx is part of
    const uint32_t num_nodes,
    bool* __restrict__ moves, // moves[idx] -> true if the node would like to move to the other side of the bisection
    uint32_t* __restrict__ even_event_idx, // event_idx[idx] -> 1u if the node would like to move and is in an even partition
    uint32_t* __restrict__ odd_event_idx, // event_idx[idx] -> ... odd partition
    float* __restrict__ scores // scores[idx] -> gain for node idx's move
) {
    // STYLE: one node per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;
    
    /*
    * Idea:
    * - one node per warp
    * - each node visits the pins of each of its touching hedge
    * - for every pin, if it is in the same partition "p" as the node, the hedge's weight goes in favor of staying in "p"
    * - if it is in partition "p+1", the weight goes in favor of moving to "p+1"
    * - the higher-total-weight partition is proposed
    */
    
    const uint32_t my_partition = partitions[warp_id];
    const uint32_t other_partition = (my_partition & 1u) == 0 ? my_partition + 1 : my_partition - 1;
    
    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
    float curr_p_score = 0.0f;
    float other_p_score = 0.0f;

    // scan touching hyperedges
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            uint32_t pin = *my_hedge;
            if (pin == warp_id) continue;
            if (partitions[pin] == my_partition) // the pin is on my same side
                curr_p_score += my_hedge_weight;
            else if (partitions[pin] == other_partition) // the pin is on the other side of the bisection
                other_p_score += my_hedge_weight;
        }
    }

    curr_p_score = warpReduceSumLN0<float>(curr_p_score);
    other_p_score = warpReduceSumLN0<float>(other_p_score);

    if (lane_id == 0) {
        if (curr_p_score >= other_p_score) {
            moves[warp_id] = false;
            even_event_idx[warp_id] = 0u;
            odd_event_idx[warp_id] = 0u;
            scores[warp_id] = 0.0f;
        } else {
            moves[warp_id] = true;
            even_event_idx[warp_id] = (my_partition & 1u) == 0;
            odd_event_idx[warp_id] = (my_partition & 1u);
            scores[warp_id] = other_p_score - curr_p_score;
        }
    }
}

// transform moves into a sequence of events
__global__
void label_move_events_kernel(
    const bool* __restrict__ moves,
    const float* __restrict__ scores,
    const uint32_t* __restrict__ even_ev_idx,
    const uint32_t* __restrict__ odd_ev_idx,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_nodes,
    uint32_t* __restrict__ even_ev_partition,
    float* __restrict__ even_ev_score,
    uint32_t* __restrict__ even_ev_node,
    uint32_t* __restrict__ odd_ev_partition,
    float* __restrict__ odd_ev_score,
    uint32_t* __restrict__ odd_ev_node
) {
    // STYLE: one node (move) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    const bool move = moves[tid];
    const float score = scores[tid];
    const uint32_t part = partitions[tid];
    
    if (move) {
        if ((part & 1u) == 0) {
            const uint32_t offset = even_ev_idx[tid];
            even_ev_partition[offset] = part >> 1; // stored as p/2
            even_ev_score[offset] = -score; // temporarily negative, to use an ascending sort as if it were descending
            even_ev_node[offset] = tid;
        } else {
            const uint32_t offset = odd_ev_idx[tid];
            odd_ev_partition[offset] = part >> 1; // stored as (p-1)/2
            odd_ev_score[offset] = -score; // temporarily negative, to use an ascending sort as if it were descending
            odd_ev_node[offset] = tid;
        }
    }
}

// update move gains in sequence and merge them into pairs
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
// SHUFFLES OVER: d
__global__
void label_cascade_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions, // partitions[node idx] -> the partition node idx is part of
    const uint32_t* __restrict__ part_even_event_offsets, // part_event_offsets[part/2 idx] -> starting idx for events regarding even partition part
    const uint32_t* __restrict__ part_odd_event_offsets, // part_event_offsets[(part-1)/2 idx] -> ...
    const uint32_t* __restrict__ even_ranks, // ranks[node idx] -> even event index for node idx (UINT32_MAX if no event)
    const uint32_t* __restrict__ odd_ranks, // ranks[node idx] -> ...
    const uint32_t* __restrict__ even_event_node,
    const uint32_t* __restrict__ odd_event_node,
    const uint32_t even_events_count,
    const uint32_t odd_events_count,
    float* __restrict__ even_event_score
) { 
    // STYLE: one event per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= even_events_count + odd_events_count) return;

    /*
    * Idea:
    * - same as label_propagation_kernel, but check first if your neighbor will change partition before you (in-sequence)
    * - moreover, now we need to shuffle between the two lists of even/odd events and their relative ranks
    */
    
    const bool even = warp_id < even_events_count;
    uint32_t event_id;
    // |
    const uint32_t* my_part_event_offsets;
    const uint32_t* my_ranks;
    const uint32_t* my_event_node;
    // |
    const uint32_t* other_part_event_offsets;
    const uint32_t* other_ranks;
    //const uint32_t* other_event_node;
    if (even) {
        event_id = warp_id;
        // |
        my_part_event_offsets = part_even_event_offsets;
        my_ranks = even_ranks;
        my_event_node = even_event_node;
        // |
        other_part_event_offsets = part_odd_event_offsets;
        other_ranks = odd_ranks;
        //other_event_node = odd_event_node;
    } else {
        event_id = warp_id - even_events_count;
        // |
        my_part_event_offsets = part_odd_event_offsets;
        my_ranks = odd_ranks;
        my_event_node = odd_event_node;
        // |
        other_part_event_offsets = part_even_event_offsets;
        other_ranks = even_ranks;
        //other_event_node = even_event_node;
    }

    uint32_t my_node = my_event_node[event_id];
    
    const uint32_t my_partition = partitions[my_node];
    const uint32_t other_partition = (my_partition & 1u) == 0 ? my_partition + 1 : my_partition - 1;

    const uint32_t my_part_events_offset = my_part_event_offsets[my_partition >> 1];
    const uint32_t my_part_event_rank = event_id - my_part_events_offset;
    const uint32_t other_part_events_offset = other_part_event_offsets[other_partition >> 1];
    const uint32_t other_part_events_count = other_part_event_offsets[(other_partition >> 1) + 1] - other_part_event_offsets[other_partition >> 1];
    if (my_part_event_rank >= other_part_events_count) return; // omit events that don't have a pair (those exceeding the minimum of the events count between the two partitions)

    const uint32_t* my_touching = touching + touching_offsets[my_node];
    const uint32_t* not_my_touching = touching + touching_offsets[my_node + 1];
    float curr_p_score = 0.0f;
    float other_p_score = 0.0f;

    // scan touching hyperedges
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            uint32_t pin = *my_hedge;
            if (pin == my_node) continue;
            if (partitions[pin] == my_partition) {
                if (my_ranks[pin] == UINT32_MAX || my_ranks[pin] > event_id) // the pin is on my same side and didn't move before me
                    curr_p_score += my_hedge_weight;
                else // the pin was on my same side, but move before me
                    other_p_score += my_hedge_weight;
            } else if (partitions[pin] == other_partition) {
                if (other_ranks[pin] == UINT32_MAX || other_ranks[pin] - other_part_events_offset > my_part_event_rank) // the pin is on the other side of the bisection and didn't move before me
                    other_p_score += my_hedge_weight;
                else // the pin was on the other side of the bisection, but move before me
                    curr_p_score += my_hedge_weight;
            }
        }
    }

    curr_p_score = warpReduceSumLN0<float>(curr_p_score);
    other_p_score = warpReduceSumLN0<float>(other_p_score);

    if (lane_id == 0) {
        // accumulate everything in the even partition's score
        const uint32_t idx = even ? event_id : part_even_event_offsets[my_partition >> 1] + my_part_event_rank;
        atomicAdd(&even_event_score[idx], other_p_score - curr_p_score);
    }
}

// apply pair swaps (handle each node of a pair at the same time, from the "even" side)
__global__
void apply_move_events_kernel(
    const uint32_t* __restrict__ apply_up_to, // apply_up_to[p/2] -> last absolute event idx to apply for partition p or p+1
    const uint32_t* __restrict__ even_event_part, // even_event_part[event idx] -> p/2 of the event
    const uint32_t* __restrict__ even_event_node, // even_event_node[event idx] -> node of the event
    const uint32_t* __restrict__ part_even_event_offsets, // part_even_event_offsets[p/2] -> first event idx among even events for part p
    const uint32_t* __restrict__ part_odd_event_offsets, // part_odd_event_offsets[p/2] -> first event idx among odd events for part p
    const uint32_t* __restrict__ odd_event_node, // odd_event_node[event idx] -> node of the event
    const uint32_t even_events_count,
    uint32_t* __restrict__ partitions
) {
    // STYLE: one event per thread!
    const uint32_t my_event = blockIdx.x * blockDim.x + threadIdx.x;
    if (my_event >= even_events_count) return;

    // check if the move is to apply
    const uint32_t my_part_half = even_event_part[my_event];
    const uint32_t apply_idx = apply_up_to[my_part_half];
    if (my_event > apply_idx || apply_idx == UINT32_MAX) return;

    // check if the move exists for both even and odd events (the maximum could have been overconfident)
    const uint32_t my_part_offset = part_even_event_offsets[my_part_half];
    const uint32_t other_part_offset = part_odd_event_offsets[my_part_half];
    const uint32_t not_other_part_offset = part_odd_event_offsets[my_part_half + 1];
    const uint32_t other_event = other_part_offset + (my_event - my_part_offset);
    if (other_event >= not_other_part_offset) return;

    const uint32_t my_node = even_event_node[my_event];
    const uint32_t other_node = odd_event_node[other_event];
    
    partitions[my_node] = (my_part_half << 1) + 1u; // partitions[my_node] was even by construction
    partitions[other_node] = (my_part_half << 1);
}









// for each pair of partitions p*2 and p*2+1 in the bisection tree, evaluate its total connection weight with the sibling subtree of p
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
// SHUFFLES OVER: d
__global__
void sibling_tree_connection_strength_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ order, // order[idx] -> node currently in position idx
    const uint32_t* __restrict__ ord_part, // ord_part[idx] -> partition of node in order[idx]
    const uint32_t* __restrict__ partitions, // partitions[node idx] -> the partition node idx is part of
    const uint32_t num_nodes,
    float* __restrict__ scores // scores[p] -> score in favor of p being near the sibling of partition floor(p/2)
) {
    // STYLE: one node per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;
    
    /*
    * Idea:
    * - one node per warp
    * - each node visits its neighbors and if they are in the sibling tree of its parent partition, then it accumulates their connections strength in favor of its partition
    * - the higher-scoring partition gets to be near the sibling
    */
    
    const uint32_t my_node = order[warp_id];
    const uint32_t my_part = ord_part[warp_id];
    const uint32_t my_part_half = my_part >> 1;
    const uint32_t sibling_part_half = (my_part_half & 1u) == 0 ? my_part_half + 1 : my_part_half - 1;
    
    const uint32_t* my_touching = touching + touching_offsets[my_node];
    const uint32_t* not_my_touching = touching + touching_offsets[my_node + 1];
    float score = 0.0f;

    // scan touching hyperedges
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            uint32_t pin = *my_hedge;
            uint32_t pin_part_half = partitions[pin] >> 1;
            if (pin_part_half == sibling_part_half) // the pin is in the sibling subtree
                score += my_hedge_weight;
        }
    }

    score = warpReduceSumLN0<float>(score);

    if (lane_id == 0) {
        atomicAdd(&scores[my_part], score);
    }
}

// flag subtrees that need to be internally reversed
__global__
void flag_reversals_kernel(
    const float* __restrict__ sibling_score,
    const uint32_t num_parts,
    bool* __restrict__ reverse
) {
    // STYLE: one (half) partition per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    const float left_score = sibling_score[tid*2];
    const float right_score = sibling_score[tid*2 + 1];

    const bool is_sibling_to_the_left = (tid & 1u) == 1;

    reverse[tid] = (is_sibling_to_the_left && left_score < right_score) || (!is_sibling_to_the_left && left_score > right_score);
}

// reverse elements in each flagged segment
__global__ void apply_reversals_kernel(
    const uint32_t* segment, // segment[n] -> segment idx of which data[i] is part
    const uint32_t* offsets, // offsets[i] -> start idx of the i-th segment in "data"
    const bool* flag, // flag[i] -> true if the i-th segment in "data" is to be reversed
    const uint32_t size, // size of data
    uint32_t* data // data[n] -> segments of values being reversed
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    const uint32_t seg = segment[tid];
    if (!flag[seg]) return;

    const uint32_t seg_begin = offsets[seg];
    const uint32_t seg_end = offsets[seg + 1];
    const uint32_t seg_half_size = (seg_end - seg_begin) >> 1;
    const uint32_t seg_tid = tid - seg_begin;
    if (seg_tid >= seg_half_size) return;

    const uint32_t other = seg_end - seg_tid - 1;
    uint32_t tmp = data[tid];
    data[tid] = data[other];
    data[other] = tmp;
}







// for each partition, randomly bisect it, mapping every partition id "p" to either "p*2" or "p*2+1"
void split_partitions_rand(
    uint32_t* d_partitions,
    uint32_t num_nodes,
    uint32_t num_parts,
    curandGenerator_t gen
) {
    thrust::device_ptr<uint32_t> t_partitions(d_partitions);

    thrust::device_vector<uint32_t> t_original_idx(num_nodes); // original_idx[i] -> idx of node currently in partition partitions[i]
    thrust::sequence(t_original_idx.begin(), t_original_idx.end());

    uint32_t* d_partitions_cpy = nullptr; // auxiliary copy of current partitions for sorting and scattering
    CUDA_CHECK(cudaMalloc(&d_partitions_cpy, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_partitions_cpy, d_partitions, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    thrust::device_ptr<uint32_t> t_partitions_cpy(d_partitions_cpy);

    // generate one random uint32 per element, seeded
    thrust::device_vector<uint32_t> t_rand_keys(num_nodes);
    CURAND_CHECK(curandGenerate(gen, thrust::raw_pointer_cast(t_rand_keys.data()), num_nodes));

    // sort by (partition, random), carrying along the original indices
    // => now partitions_cpy is grouped by "p", with random order inside each group
    auto part_rand_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_partitions_cpy, t_rand_keys.begin()));
    auto part_rand_key_end = part_rand_key_begin + num_nodes;
    thrust::sort_by_key(part_rand_key_begin, part_rand_key_end, t_original_idx.begin());

    // build offset indices over reordered partitions
    thrust::device_vector<uint32_t> t_part_offsets(num_parts + 1); // part_offsets[p] -> first index of partition p in partitions_cpy
    thrust::counting_iterator<uint32_t> search_begin(0);
    thrust::lower_bound(
        t_partitions_cpy, t_partitions_cpy + num_nodes,
        search_begin, search_begin + num_parts,
        t_part_offsets.begin()
    );
    t_part_offsets[num_parts] = num_nodes;

    // split each partition in half:
    // - inside each partition, original node indices are not randomly ordered
    // - take the lower half of those indices and map it to p*2, take the upper half and map it to p*2+1
    {
        // launch configuration - split partitions kernel
        int threads_per_block = 256;
        int num_threads_needed = num_nodes; // 1 thread per node
        int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - split partitions kernel
        std::cout << "Running split partitions kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        split_partitions_kernel<<<blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(t_part_offsets.data()),
            num_nodes,
            d_partitions_cpy
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // undo the sort - scatter back to updated partitions to their original idxs
    thrust::scatter(
        t_partitions_cpy,
        t_partitions_cpy + num_nodes,
        t_original_idx.begin(),
        t_partitions
    );

    CUDA_CHECK(cudaFree(d_partitions_cpy));
}








// ================================================================
// Entry Point
// ================================================================

// return a high-locality, seeded 1D ordering of nodes
uint32_t* locality_ordering(
    const uint32_t num_nodes,
    const uint32_t num_hedges,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const float* d_hedge_weights,
    const uint32_t* d_touching,
    const dim_t* d_touching_offsets,
    const uint64_t seed
) {
    /*
    * IDEA:
    * - recursive bisection
    * - divide nodes in two halves, and each half in half again, and so creating a binary tree where each leaf is a single node
    *  - done with a random balanced bipartitioning and repeated label propagation
    * - then go back up the binary tree
    * - at each branch, decide how to order the two halves ("which goes first")
    *   - check whether you are on the left or right side of your grandparent
    *   - compute how strongly each half is connected with the other side of your grandparent
    *   - if the grandparent has you on the left, and your left side is more strongly connected to grandpa's right,
    *       reverse the order of every leaf under yourself, trapping good connections inside, while favoring links with grandpa's second child
    *   - else, everything stays as it is  - mirrored idea if you are on grandpa's right side
    * - recurse up until the root, that gives you a strong 1D ordering that trapped connection locality as much as possible
    * 
    * Label propagation:
    * - 60% of the logic is the one from your failed initial partition implementation
    * - you need to compute pins-per-partition on the fly, but they are just for a bi-partitioning, so it’s cheap (just two counters per hedge as seen from each super-partition)
    * - use the same gain-even-on-no-disconnect logic to encourage node moves at the beginning, but stabilize to no gain unless you disconnect on later iterations
    * - seed the random initial partitioning, before
    * - to select moves to apply, do a segmented sort by gain, where segments are the moves proposed in each super-partition
    * - to keep the strictest balance:
    *     - initialize as balanced
    *     - sort moves by gain AND such that every move is followed by one in the inverse direct, then count such that you keep a number that keeps balance (be wary of partitions differing by one in size
    * - yes, you need two partition arrays, one for super-partitions (the layer about you in the tree), and one for newly built ones
    */

    /* 
    * How to generate partition ids:
    * - given a partition with id "p", currently being bisected, its two child partitions will have ids:
    *   - p*2   - p*2 + 1
    * - this works because the number of partitions doubles at every bisection of every partition
    * - uneven partitions will lead to some non-existing id, be wary
    * - when re-merging partitions, fuse all even "p"s with their odd successor and give both the "p/2" id
    */

    assert(num_nodes > 0); // how, why, what?!

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    uint32_t* d_partitions = nullptr; // partitions[node idx] -> current partition (of bypartitions) the node is in

    CUDA_CHECK(cudaMalloc(&d_partitions, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_partitions, 0x00, num_nodes * sizeof(uint32_t))); // everyone starts in the same partition
    thrust::device_ptr<uint32_t> t_partitions(d_partitions);

    uint32_t num_parts = 1u;

    bool* d_moves = nullptr; // move[node idx] -> false if the node doesn't want to move, true if the node would like to switch partition p*2->p*2+1 or p*2+1->p*2
    float* d_scores = nullptr; // score[node idx] -> connectivity gain for the above move (even not moving is done with a "gain")
    uint32_t* d_even_event_idx = nullptr; // event_idx[node idx] -> event idx for the node, in case the node decided to move (the node was in an even partition)
    uint32_t* d_odd_event_idx = nullptr; // event_idx[node idx] -> ... (the node was in an odd partition)

    CUDA_CHECK(cudaMalloc(&d_moves, num_nodes * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_scores, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_even_event_idx, (num_nodes + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_odd_event_idx, (num_nodes + 1) * sizeof(uint32_t)));
    thrust::device_ptr<uint32_t> t_even_event_idx(d_even_event_idx);
    thrust::device_ptr<uint32_t> t_odd_event_idx(d_odd_event_idx);

    uint32_t level_idx = 0u;
    while (num_parts < (num_nodes + 1) / 2) { // as long as partitions do not strictly contain 1 or 2 nodes...
        std::cout << "Bisection level " << level_idx << " number of partitions=" << num_parts << "\n";
        level_idx++;

        // random bisection of every partition
        split_partitions_rand(
            d_partitions,
            num_nodes,
            num_parts,
            gen
        );

        num_parts *= 2;

        for (uint32_t lp_repeat = 0u; lp_repeat < LABELPROP_REPEATS; lp_repeat++) {
            // compute gains (and moves) in-isolation
            // NOTE: no need to init. "d_moves" and "d_scores", they are overwritten anyway
            {
                // launch configuration - label propagation kernel
                int threads_per_block = 128; // 128/32 -> 4 warps per block
                int warps_per_block = threads_per_block / WARP_SIZE;
                int num_warps_needed = num_nodes; // 1 warp per node
                int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
                // launch - label propagation kernel
                std::cout << "Running label propagation kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                label_propagation_kernel<<<blocks, threads_per_block>>>(
                    d_hedges,
                    d_hedges_offsets,
                    d_touching,
                    d_touching_offsets,
                    d_hedge_weights,
                    d_partitions,
                    num_nodes,
                    d_moves,
                    d_even_event_idx,
                    d_odd_event_idx,
                    d_scores
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            
            // build move events (partition, score, node)
            CUDA_CHECK(cudaMemset(d_even_event_idx + num_nodes, 0u, sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(d_odd_event_idx + num_nodes, 0u, sizeof(uint32_t)));
            thrust::exclusive_scan(t_even_event_idx, t_even_event_idx + num_nodes + 1, t_even_event_idx); // in-place
            thrust::exclusive_scan(t_odd_event_idx, t_odd_event_idx + num_nodes + 1, t_odd_event_idx); // in-place
            uint32_t even_events_count;
            uint32_t odd_events_count;
            CUDA_CHECK(cudaMemcpy(&even_events_count, d_even_event_idx + num_nodes, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&odd_events_count, d_odd_event_idx + num_nodes, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (even_events_count == 0 || odd_events_count == 0) {
                std::cout << "No valid label propagation move found on level " << level_idx << " repeat " << lp_repeat << " (even events count=" << even_events_count << " odd events count=" << odd_events_count << ") !!\n";
                break;
            }
            std::cout << "Label propagation on level " << level_idx << " repeat " << lp_repeat << " (even events count=" << even_events_count << " odd events count=" << odd_events_count << ")\n";

            // NOTE: partitions inside events are stored as p/2 and (p-1)/2 !!
            uint32_t* d_even_event_part = nullptr; // part[idx] -> src partition / 2 for the idx-th move (partition being even)
            float* d_even_event_score = nullptr; // score[idx] -> gain for the idx-th move
            uint32_t* d_even_event_node = nullptr; // node[idx] -> node moved in the idx-th move
            uint32_t* d_odd_event_part = nullptr; // part[idx] -> (src partition - 1) / 2 ... (partition being odd)
            float* d_odd_event_score = nullptr; // score[idx] -> ...
            uint32_t* d_odd_event_node = nullptr; // node[idx] -> ...
            CUDA_CHECK(cudaMalloc(&d_even_event_part, even_events_count * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_even_event_score, even_events_count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_even_event_node, even_events_count * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_odd_event_part, odd_events_count * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_odd_event_score, odd_events_count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_odd_event_node, odd_events_count * sizeof(uint32_t)));
            thrust::device_ptr<uint32_t> t_even_event_part(d_even_event_part);
            thrust::device_ptr<float> t_even_event_score(d_even_event_score);
            thrust::device_ptr<uint32_t> t_even_event_node(d_even_event_node);
            thrust::device_ptr<uint32_t> t_odd_event_part(d_odd_event_part);
            thrust::device_ptr<float> t_odd_event_score(d_odd_event_score);
            thrust::device_ptr<uint32_t> t_odd_event_node(d_odd_event_node);
            {
                // launch configuration - label move events kernel
                int threads_per_block = 256;
                int num_threads_needed = num_nodes; // 1 thread per node
                int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
                // launch - label move events kernel
                std::cout << "Running label move events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                label_move_events_kernel<<<blocks, threads_per_block>>>(
                    d_moves,
                    d_scores,
                    d_even_event_idx,
                    d_odd_event_idx,
                    d_partitions,
                    num_nodes,
                    d_even_event_part,
                    d_even_event_score,
                    d_even_event_node,
                    d_odd_event_part,
                    d_odd_event_score,
                    d_odd_event_node
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // sort events by (partition, score, node)
            auto move_even_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_even_event_part, t_even_event_score, t_even_event_node));
            auto move_even_events_key_end = move_even_events_key_begin + even_events_count;
            thrust::sort(move_even_events_key_begin, move_even_events_key_end);
            // |
            auto move_odd_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_odd_event_part, t_odd_event_score, t_odd_event_node));
            auto move_odd_events_key_end = move_odd_events_key_begin + odd_events_count;
            thrust::sort(move_odd_events_key_begin, move_odd_events_key_end);

            // build offset indices over reordered events per partition
            uint32_t* d_part_even_event_offsets = nullptr; // part_even_event_offsets[p] -> first index of partition p*2 in even_event_part
            CUDA_CHECK(cudaMalloc(&d_part_even_event_offsets, (num_parts/2 + 1) * sizeof(uint32_t)));
            thrust::device_ptr<uint32_t> t_part_even_event_offsets(d_part_even_event_offsets);
            thrust::counting_iterator<uint32_t> even_search_begin(0);
            // NOTE: the search was "made to work" by storing p/2 inside event_part-s, hence it is enough to search from 0 to num_parts/2
            thrust::lower_bound(
                t_even_event_part, t_even_event_part + even_events_count,
                even_search_begin, even_search_begin + num_parts/2,
                t_part_even_event_offsets
            );
            CUDA_CHECK(cudaMemcpy(d_part_even_event_offsets + num_parts/2, &even_events_count, sizeof(uint32_t), cudaMemcpyHostToDevice));
            // |
            uint32_t* d_part_odd_event_offsets = nullptr; // part_odd_event_offsets[p] -> first index of partition p*2+1 in odd_event_part
            CUDA_CHECK(cudaMalloc(&d_part_odd_event_offsets, (num_parts/2 + 1) * sizeof(uint32_t)));
            thrust::device_ptr<uint32_t> t_part_odd_event_offsets(d_part_odd_event_offsets);
            thrust::counting_iterator<uint32_t> odd_search_begin(0);
            thrust::lower_bound(
                t_odd_event_part, t_odd_event_part + odd_events_count,
                odd_search_begin, odd_search_begin + num_parts/2,
                t_part_odd_event_offsets
            );
            CUDA_CHECK(cudaMemcpy(d_part_odd_event_offsets + num_parts/2, &odd_events_count, sizeof(uint32_t), cudaMemcpyHostToDevice));

            // repurpose "event_idx" to store the reverse map: event_idx[node] -> event-idx (if any) of node - in other words this scatter does "event_idx[event_node[i]] = i"
            CUDA_CHECK(cudaMemset(d_even_event_idx, 0xFF, num_nodes * sizeof(uint32_t)));
            thrust::scatter(thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(even_events_count), t_even_event_node, t_even_event_idx);
            CUDA_CHECK(cudaMemset(d_odd_event_idx, 0xFF, num_nodes * sizeof(uint32_t)));
            thrust::scatter(thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(odd_events_count), t_odd_event_node, t_odd_event_idx);

            // update gains in-sequence
            // assume moves are done in pairs => re-compute the pair's gain in-sequence, assuming all prior pairs already swapped
            // => already accumulate the two scores on the "even" segment's event (only up to the length of the smallest events segment between even and odd)
            CUDA_CHECK(cudaMemset(d_even_event_score, 0x00, even_events_count * sizeof(float)));
            {
                // launch configuration - label cascade kernel
                int threads_per_block = 128; // 128/32 -> 4 warps per block
                int warps_per_block = threads_per_block / WARP_SIZE;
                int num_warps_needed = even_events_count + odd_events_count; // 1 warp per event
                int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
                // launch - label cascade kernel
                std::cout << "Running label cascade kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                label_cascade_kernel<<<blocks, threads_per_block>>>(
                    d_hedges,
                    d_hedges_offsets,
                    d_touching,
                    d_touching_offsets,
                    d_hedge_weights,
                    d_partitions,
                    d_part_even_event_offsets,
                    d_part_odd_event_offsets,
                    d_even_event_idx,
                    d_odd_event_idx,
                    d_even_event_node,
                    d_odd_event_node,
                    even_events_count,
                    odd_events_count,
                    d_even_event_score
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

             // inclusive scan inside each key (= partition) on the even event scores => for each event we get the cumulative gain up to that point in the partition's move sequence
            thrust::inclusive_scan_by_key(t_even_event_part, t_even_event_part + even_events_count, t_even_event_score, t_even_event_score);
            // extract the maximum idx (relative to the start of the overall array) for every partition's pair
            // => repurpose d_even_event_idx as even_event_idx[idx] -> absolute idx of the last moves-pair to apply for partitions 2*idx and 2*idx+1
            auto event_score_pair = thrust::make_zip_iterator(thrust::make_tuple(t_even_event_score, thrust::counting_iterator<uint32_t>(0)));
            CUDA_CHECK(cudaMemset(d_even_event_idx, 0xFF, (num_parts / 2) * sizeof(uint32_t)));
            auto d_event_argmax = thrust::make_transform_output_iterator(
                t_even_event_idx, // discard the "max" part of the "argmax" return tuple
                [] __device__ (auto x) { return (thrust::get<0>(x) < 0) ? UINT32_MAX : thrust::get<1>(x); }
            );
            thrust::reduce_by_key(
                t_even_event_part, t_even_event_part + even_events_count, event_score_pair,
                thrust::make_discard_iterator(), d_event_argmax,
                thrust::equal_to<uint32_t>{},
                [] __device__ (auto a, auto b) { return (thrust::get<0>(b) > thrust::get<0>(a)) ? b : a; }
            );

            // apply pairs of improving moves
            // add together the gain of equi-ranked moves between bisected partitions as the gain of the pair to swap
            {
                // launch configuration - apply move events kernels
                int threads_per_block = 256;
                int num_threads_needed = even_events_count; // 1 thread per event
                int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
                // launch - apply move events kernels
                std::cout << "Running apply move events kernels (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                apply_move_events_kernel<<<blocks, threads_per_block>>>(
                    d_even_event_idx,
                    d_even_event_part,
                    d_even_event_node,
                    d_part_even_event_offsets,
                    d_part_odd_event_offsets,
                    d_odd_event_node,
                    even_events_count,
                    d_partitions
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            CUDA_CHECK(cudaFree(d_even_event_part));
            CUDA_CHECK(cudaFree(d_even_event_score));
            CUDA_CHECK(cudaFree(d_even_event_node));
            CUDA_CHECK(cudaFree(d_odd_event_part));
            CUDA_CHECK(cudaFree(d_odd_event_score));
            CUDA_CHECK(cudaFree(d_odd_event_node));
            CUDA_CHECK(cudaFree(d_part_even_event_offsets));
            CUDA_CHECK(cudaFree(d_part_odd_event_offsets));
        }
    }

    CUDA_CHECK(cudaFree(d_moves));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_even_event_idx));
    CUDA_CHECK(cudaFree(d_odd_event_idx));

    // one final bisection to go down to 1-element partitions
    split_partitions_rand(
        d_partitions,
        num_nodes,
        num_parts,
        gen
    );

    num_parts *= 2;

    uint32_t* d_order = nullptr; // order[idx] -> node currently in position idx
    uint32_t* d_ord_part = nullptr; // ord_part[idx] -> partition of node in order[idx]

    CUDA_CHECK(cudaMalloc(&d_order, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ord_part, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_ord_part, d_partitions, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    thrust::device_ptr<uint32_t> t_order(d_order);
    thrust::device_ptr<uint32_t> t_ord_part(d_ord_part);
    thrust::sequence(t_order, t_order + num_nodes);
    thrust::sort_by_key(t_ord_part, t_ord_part + num_nodes, t_order); // this also sorts copy(d_partitions) into d_ord_part

    // fuse back partitions while internally reversing them as needed to "trap" strong connections locally inside partition pairs
    while (num_parts > 1) { // go back up the bisection tree
        std::cout << "Tree reorientation level " << level_idx << " number of partitions=" << num_parts << "\n";
        level_idx--;
        
        float* d_sibling_score = nullptr; // sibling_score[p] -> total connection strength between partition p and the sibling subtree of floor(p/2)
        CUDA_CHECK(cudaMalloc(&d_sibling_score, num_parts * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sibling_score, 0x00, num_parts * sizeof(float)));

        // compute connection strength of each partition with its parent's sibling subtree
        {
            // launch configuration - sibling tree connection strength kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_nodes; // 1 warp per event
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - sibling tree connection strength kernel
            std::cout << "Running sibling tree connection strength kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            sibling_tree_connection_strength_kernel<<<blocks, threads_per_block>>>(
                d_hedges,
                d_hedges_offsets,
                d_touching,
                d_touching_offsets,
                d_hedge_weights,
                d_order,
                d_ord_part,
                d_partitions,
                num_nodes,
                d_sibling_score
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // refold p*2 and p*2+1 back into p
        num_parts /= 2;
        thrust::transform(
            t_partitions, t_partitions + num_nodes, t_partitions,
            [] __device__ (uint32_t x) { return x >> 1; }
        );
        thrust::transform(
            t_ord_part, t_ord_part + num_nodes, t_ord_part,
            [] __device__ (uint32_t x) { return x >> 1; }
        );

        bool* d_reverse = nullptr; // reverse[p/2] -> true if the subtree of p/2 (well, ex-p/2, since we already folded it back in p) needs to have its leaves-order reversed
        CUDA_CHECK(cudaMalloc(&d_reverse, num_parts * sizeof(bool)));
        {
            // launch configuration - flag reversals kernel
            int threads_per_block = 256;
            int num_threads_needed = num_parts; // 1 thread per (half) partition
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - flag reversals kernel
            std::cout << "Running flag reversals kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            flag_reversals_kernel<<<blocks, threads_per_block>>>(
                d_sibling_score,
                num_parts,
                d_reverse
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // build offset indices over ord_part
        uint32_t* d_ord_part_offsets = nullptr; // ord_part_offsets[p] -> first index of partition p in ord_part
        CUDA_CHECK(cudaMalloc(&d_ord_part_offsets, (num_parts + 1) * sizeof(uint32_t)));
        thrust::device_ptr<uint32_t> t_ord_part_offsets(d_ord_part_offsets);
        thrust::counting_iterator<uint32_t> ord_search_begin(0);
        // NOTE: the search was "made to work" by storing p/2 inside event_part-s, hence it is enough to search from 0 to num_parts/2
        thrust::lower_bound(
            t_ord_part, t_ord_part + num_nodes,
            ord_search_begin, ord_search_begin + num_parts,
            t_ord_part_offsets
        );
        CUDA_CHECK(cudaMemcpy(d_ord_part_offsets + num_parts, &num_nodes, sizeof(uint32_t), cudaMemcpyHostToDevice));

        // apply the reversal of leaves/nodes inside each flagged subtree
        {
            // launch configuration - apply reversals kernel
            int threads_per_block = 256;
            int num_threads_needed = num_nodes; // 1 thread per (half) partition
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - apply reversals kernel
            std::cout << "Running apply reversals kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            apply_reversals_kernel<<<blocks, threads_per_block>>>(
                d_ord_part,
                d_ord_part_offsets,
                d_reverse,
                num_nodes,
                d_order
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaFree(d_sibling_score));
        CUDA_CHECK(cudaFree(d_reverse));
        CUDA_CHECK(cudaFree(d_ord_part_offsets));
    }

    // write d_order_idx as the reverse map of order
    uint32_t* d_order_idx = nullptr; // order_idx[node] -> position in the ordering for node

    CUDA_CHECK(cudaMalloc(&d_order_idx, num_nodes * sizeof(uint32_t)));
    thrust::device_ptr<uint32_t> t_order_idx(d_order_idx);
    thrust::scatter(thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(num_nodes), t_order, t_order_idx);

    CUDA_CHECK(cudaFree(d_order));
    CUDA_CHECK(cudaFree(d_ord_part));

    CURAND_CHECK(curandDestroyGenerator(gen));

    return d_order_idx;
}
