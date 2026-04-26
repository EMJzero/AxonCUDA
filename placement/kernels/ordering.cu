#include "ordering.cuh"
#include "utils_plc.cuh"
#include "utils.cuh"

// BISECTION

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


// SPLIT-COST EVALUATION

// extract into events the even partition-mapped pin-runs of a hedge whose sibling partition is also present
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// SHUFFLES OVER: d
__global__
void flag_cutnet_events_kernel(
    const uint32_t* __restrict__ part_pins,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    dim_t* __restrict__ flags
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    const dim_t my_hedge_offset = hedges_offsets[warp_id];
    const dim_t not_my_hedge_offset = hedges_offsets[warp_id + 1];
    if (not_my_hedge_offset <= my_hedge_offset + 1u) return; // empty hedge or singleton hedge cannot generate a cut event
    dim_t my_offset = my_hedge_offset + lane_id;

    // each thread in the warp reads one every WARP_SIZE pins
    for (; my_offset < not_my_hedge_offset; my_offset += WARP_SIZE) {
        const uint32_t part_pin = part_pins[my_offset];
        if ((part_pin & 1u) == 1) continue; // odd partition
        if (my_offset > my_hedge_offset && part_pins[my_offset - 1] == part_pin) continue; // not the start of the run

        dim_t run_end = my_offset + 1u;
        while (run_end < not_my_hedge_offset && part_pins[run_end] == part_pin) run_end++;
        if (run_end >= not_my_hedge_offset || part_pins[run_end] != part_pin + 1u) continue; // sibling partition absent
        flags[my_offset] = 1u;
    }
}

// emit one event per sibling partition-pair touched by a hedge, weighted by the hedge's minority pin-count across the split
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// SHUFFLES OVER: d
__global__
void cutnet_event_generation_kernel(
    const uint32_t* __restrict__ part_pins,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const dim_t* __restrict__ flags,
    const uint32_t num_hedges,
    float* __restrict__ event_weight,
    uint32_t* __restrict__ event_part
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    const dim_t my_hedge_offset = hedges_offsets[warp_id];
    const dim_t not_my_hedge_offset = hedges_offsets[warp_id + 1];
    if (not_my_hedge_offset <= my_hedge_offset + 1u) return; // empty hedge or singleton hedge cannot generate a cut event
    dim_t my_offset = my_hedge_offset + lane_id;
    const float my_weight = hedge_weights[warp_id];

    // each thread in the warp reads one every WARP_SIZE pins
    for (; my_offset < not_my_hedge_offset; my_offset += WARP_SIZE) {
        const uint32_t part_pin = part_pins[my_offset];
        if ((part_pin & 1u) == 1) continue; // odd partition
        if (my_offset > my_hedge_offset && part_pins[my_offset - 1] == part_pin) continue; // not the start of the run

        dim_t even_run_end = my_offset + 1u;
        while (even_run_end < not_my_hedge_offset && part_pins[even_run_end] == part_pin) even_run_end++;
        if (even_run_end >= not_my_hedge_offset || part_pins[even_run_end] != part_pin + 1u) continue; // sibling partition absent

        dim_t odd_run_end = even_run_end + 1u;
        while (odd_run_end < not_my_hedge_offset && part_pins[odd_run_end] == part_pin + 1u) odd_run_end++;

        const dim_t event_offset = flags[my_offset];
        const dim_t even_count = even_run_end - my_offset;
        const dim_t odd_count = odd_run_end - even_run_end;
        event_weight[event_offset] = my_weight * static_cast<float>(min(even_count, odd_count));
        event_part[event_offset] = part_pin / 2;
    }
}


// LABEL PROPAGATION

// gain calculation helper
__forceinline__ __device__
float sibling_move_gain(
    const uint32_t my_count,
    const uint32_t other_count,
    const float hedge_weight
) {
    const uint32_t curr_cost = min(my_count, other_count);
    const uint32_t moved_cost = min(my_count - 1u, other_count + 1u);
    return hedge_weight * static_cast<float>(static_cast<int32_t>(curr_cost) - static_cast<int32_t>(moved_cost));
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
    * - each node visits the pins of each touching hedge and counts how many sibling-pair pins lie on its current side vs the other
    * - the score is the exact improvement in weighted minority pin-cut if the node moves
    */

    const uint32_t my_partition = partitions[warp_id];
    const uint32_t other_partition = (my_partition & 1u) == 0 ? my_partition + 1 : my_partition - 1;

    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
    float gain = 0.0f;

    // scan touching hyperedges
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        uint32_t my_part_pins = (lane_id == 0); // include yourself before the move
        uint32_t other_part_pins = 0u;
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            const uint32_t pin = *my_hedge;
            if (pin == warp_id) continue;
            if (partitions[pin] == my_partition) // the pin is on my same side
                my_part_pins++;
            else if (partitions[pin] == other_partition) // the pin is on the other side of the bisection
                other_part_pins++;
        }

        my_part_pins = warpReduceSumLN0<uint32_t>(my_part_pins);
        other_part_pins = warpReduceSumLN0<uint32_t>(other_part_pins);
        if (lane_id == 0) {
            gain += sibling_move_gain(my_part_pins, other_part_pins, my_hedge_weight);
        }
    }

    if (lane_id == 0) {
        if (gain <= 0.0f) {
            moves[warp_id] = false;
            even_event_idx[warp_id] = 0u;
            odd_event_idx[warp_id] = 0u;
            scores[warp_id] = 0.0f;
        } else {
            moves[warp_id] = true;
            even_event_idx[warp_id] = (my_partition & 1u) == 0;
            odd_event_idx[warp_id] = (my_partition & 1u);
            scores[warp_id] = gain;
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
    float gain = 0.0f;

    // scan touching hyperedges
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        uint32_t my_part_pins = (lane_id == 0); // include yourself before your own move in the sequence
        uint32_t other_part_pins = 0u;
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            const uint32_t pin = *my_hedge;
            if (pin == my_node) continue;
            if (partitions[pin] == my_partition) {
                if (my_ranks[pin] == UINT32_MAX || my_ranks[pin] > event_id) // the pin is on my same side and didn't move before me
                    my_part_pins++;
                else // the pin was on my same side, but move before me
                    other_part_pins++;
            } else if (partitions[pin] == other_partition) {
                if (other_ranks[pin] == UINT32_MAX || other_ranks[pin] - other_part_events_offset > my_part_event_rank) // the pin is on the other side of the bisection and didn't move before me
                    other_part_pins++;
                else // the pin was on the other side of the bisection, but move before me (or -with- me)
                    my_part_pins++;
            }
        }

        my_part_pins = warpReduceSumLN0<uint32_t>(my_part_pins);
        other_part_pins = warpReduceSumLN0<uint32_t>(other_part_pins);
        if (lane_id == 0) {
            gain += sibling_move_gain(my_part_pins, other_part_pins, my_hedge_weight);
        }
    }

    if (lane_id == 0) {
        // accumulate everything in the even partition's score
        const uint32_t idx = even ? event_id : part_even_event_offsets[my_partition >> 1] + my_part_event_rank;
        atomicAdd(&even_event_score[idx], gain);
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

// for every node, if its current partition improved in split cost w.r.t. the best so far, overwrite its best partition with the current one
__global__
void update_best_partitions_kernel(
    const uint32_t* __restrict__ partitions,
    const float* __restrict__ cutnet,
    const float* __restrict__ last_best_cutnet,
    const uint32_t num_nodes,
    uint32_t* __restrict__ last_best_partitions
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    // TODO: could be ran only for nodes that actually switched sides...

    uint32_t part = partitions[tid];
    uint32_t parent_part = part / 2;
    float my_cutnet = cutnet[parent_part];
    float my_last_best_cutnet = last_best_cutnet[parent_part];

    // if your partition pair's split cost improved, update your last best partition
    if (my_last_best_cutnet > my_cutnet)
        last_best_partitions[tid] = part;
}


// TREE ORDERING

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
__global__
void apply_reversals_kernel(
    const uint32_t* __restrict__ segment, // segment[n] -> segment idx of which data[i] is part
    const uint32_t* __restrict__ offsets, // offsets[i] -> start idx of the i-th segment in "data"
    const bool* __restrict__ flag, // flag[i] -> true if the i-th segment in "data" is to be reversed
    const uint32_t size, // size of data
    uint32_t* __restrict__ data // data[n] -> segments of values being reversed
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

// compute the weighted span of each hedge
__global__
void measure_sequence_locality_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ order_idx,
    const uint32_t num_hedges,
    float* __restrict__ hedge_span
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    // TODO: upgrade to go over pins with a whole WARP

    const uint32_t* my_hedge = hedges + hedges_offsets[tid];
    const uint32_t* not_my_hedge = hedges + hedges_offsets[tid + 1];

    uint32_t min_pin_idx = UINT32_MAX;
    uint32_t max_pin_idx = 0u;

    for (; my_hedge < not_my_hedge; my_hedge++) {
        const uint32_t pin = *my_hedge;
        const uint32_t idx = order_idx[pin];
        if (idx < min_pin_idx) min_pin_idx = idx;
        if (idx > max_pin_idx) max_pin_idx = idx;
    }

    if (min_pin_idx == UINT32_MAX)
        hedge_span[tid] = 0.0f;
    else
        hedge_span[tid] = (max_pin_idx - min_pin_idx) * hedge_weights[tid];
}
