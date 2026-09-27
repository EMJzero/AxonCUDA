#include <cmath>
#include <algorithm>

#include "ordering.hpp"
#include "utils_plc.hpp"
#include "utils.hpp"

// NOTE: every kernel below runs a whole batch of multi-starts at once
// => per-multi-start arrays are one flat allocation of "batch_size" equally-sized segments, multi-start "b" owning [b*size, (b+1)*size)
// => partition ids are composite, "b*num_parts + p", which survives both the "*2" of a bisection and the ">>1" of a fold, since num_parts is always even
// => node idxs are batch-flat, hypergraph pin idxs are not, so they get rebased through 'nodes_base'

// BISECTION

// for every node, map its partition to to p*2 or p*2+1 depending on its position in the sorted (by rnd value) array
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
void split_partitions_kernel(
    const uint32_t* __restrict__ part_offsets,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    uint32_t* __restrict__ partitions // output in sorted order
) {
    // STYLE: one node per iteration!
    // NOTE: partitions are composite, so the sort already grouped every multi-start's nodes apart - nothing else to decode here
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t idx = 0; idx < batch_nodes; idx++) {
        const uint32_t part = partitions[idx];
        const uint32_t part_start = part_offsets[part];
        const uint32_t part_end = part_offsets[part + 1];
        const uint32_t part_size = part_end - part_start;
        const uint32_t rank = idx - part_start;

        // if you are ranked < ceil(count/2), go to 2*p, otherwise to 2*p+1
        // any odd element lands to p*2
        const uint32_t left_size = (part_size + 1u) / 2u;
        partitions[idx] = (rank < left_size) ? (2u * part) : (2u * part + 1u);
    }
}


// SPLIT-COST EVALUATION

// extract into events the even partition-mapped pin-runs of a hedge whose sibling partition is also present
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void flag_cutnet_events_kernel(
    const uint32_t* __restrict__ part_pins,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    const uint32_t batch_size,
    const dim_t hedges_size,
    uint32_t* __restrict__ flags
) {
    // STYLE: one hedge per iteration!
    const uint32_t batch_hedges = batch_size * num_hedges;
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t batch_hedge = 0; batch_hedge < batch_hedges; batch_hedge++) {
        // the hypergraph is shared by every multi-start, only the partitioning its pins are mapped through differs
        const uint32_t my_start = batch_hedge / num_hedges;
        const uint32_t my_hedge = batch_hedge - my_start * num_hedges;
        const uint32_t* my_part_pins = part_pins + my_start * hedges_size;
        uint32_t* my_flags = flags + my_start * hedges_size;

        const dim_t my_hedge_offset = hedges_offsets[my_hedge];
        const dim_t not_my_hedge_offset = hedges_offsets[my_hedge + 1];
        if (not_my_hedge_offset <= my_hedge_offset + 1u) continue; // empty hedge or singleton hedge cannot generate a cut event

        for (dim_t my_offset = my_hedge_offset; my_offset < not_my_hedge_offset; my_offset++) {
            const uint32_t part_pin = my_part_pins[my_offset];
            if ((part_pin & 1u) == 1) continue; // odd partition
            if (my_offset > my_hedge_offset && my_part_pins[my_offset - 1] == part_pin) continue; // not the start of the run

            dim_t run_end = my_offset + 1u;
            while (run_end < not_my_hedge_offset && my_part_pins[run_end] == part_pin) run_end++;
            if (run_end >= not_my_hedge_offset || my_part_pins[run_end] != part_pin + 1u) continue; // sibling partition absent
            my_flags[my_offset] = 1u;
        }
    }
}

// emit one event per sibling partition-pair touched by a hedge, weighted by the hedge's minority pin-count across the split
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void cutnet_event_generation_kernel(
    const uint32_t* __restrict__ part_pins,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ flags,
    const uint32_t num_hedges,
    const uint32_t batch_size,
    const dim_t hedges_size,
    float* __restrict__ event_weight,
    uint32_t* __restrict__ event_part
) {
    // STYLE: one hedge per iteration!
    const uint32_t batch_hedges = batch_size * num_hedges;
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t batch_hedge = 0; batch_hedge < batch_hedges; batch_hedge++) {
        // the hypergraph is shared by every multi-start, only the partitioning its pins are mapped through differs
        const uint32_t my_start = batch_hedge / num_hedges;
        const uint32_t my_hedge = batch_hedge - my_start * num_hedges;
        const uint32_t* my_part_pins = part_pins + my_start * hedges_size;
        const uint32_t* my_flags = flags + my_start * hedges_size;

        const dim_t my_hedge_offset = hedges_offsets[my_hedge];
        const dim_t not_my_hedge_offset = hedges_offsets[my_hedge + 1];
        if (not_my_hedge_offset <= my_hedge_offset + 1u) continue; // empty hedge or singleton hedge cannot generate a cut event
        const float my_weight = hedge_weights[my_hedge];

        for (dim_t my_offset = my_hedge_offset; my_offset < not_my_hedge_offset; my_offset++) {
            const uint32_t part_pin = my_part_pins[my_offset];
            if ((part_pin & 1u) == 1) continue; // odd partition
            if (my_offset > my_hedge_offset && my_part_pins[my_offset - 1] == part_pin) continue; // not the start of the run

            dim_t even_run_end = my_offset + 1u;
            while (even_run_end < not_my_hedge_offset && my_part_pins[even_run_end] == part_pin) even_run_end++;
            if (even_run_end >= not_my_hedge_offset || my_part_pins[even_run_end] != part_pin + 1u) continue; // sibling partition absent

            dim_t odd_run_end = even_run_end + 1u;
            while (odd_run_end < not_my_hedge_offset && my_part_pins[odd_run_end] == part_pin + 1u) odd_run_end++;

            const uint32_t event_offset = my_flags[my_offset];
            const dim_t even_count = even_run_end - my_offset;
            const dim_t odd_count = odd_run_end - even_run_end;
            event_weight[event_offset] = my_weight * static_cast<float>(std::min(even_count, odd_count));
            event_part[event_offset] = part_pin / 2;
        }
    }
}


// LABEL PROPAGATION

// minority pin-cut gain of moving a node to the other side of the bisection, before weighting it by the hedge's weight
inline int32_t sibling_move_gain(
    const uint32_t my_count,
    const uint32_t other_count
) {
    const uint32_t curr_cost = std::min(my_count, other_count);
    const uint32_t moved_cost = std::min(my_count - 1u, other_count + 1u);
    return static_cast<int32_t>(curr_cost) - static_cast<int32_t>(moved_cost);
}

// evaluates, per touching hyperedge, the exact gain from moving to the other side of the bisection, then sums it across
// every touching hyperedge
// 'classify(pin)' must return: 0u if the pin is (still) on my side, 1u if on the other side, UINT32_MAX to ignore it
// NOTE: in CUDA lane "l" evaluates the gain of the l-th hedge of every group of WARP_SIZE touching hedges, then the
//       lanes' gains are reduced with 'warpReduceSumLN0', both replayed here
template<typename Classify>
inline float touchingSiblingGain(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ my_touching,
    const uint32_t touching_count,
    const uint32_t nodes_base, // offset of the multi-start owning these pins, 0 outside of batched kernels
    Classify&& classify
) {
    float gain_lanes[WARP_SIZE] = {}; // gain_lanes[lane] -> gain accumulated by that lane
    for (uint32_t t = 0; t < touching_count; t++) {
        const uint32_t hedge_idx = my_touching[t];
        uint32_t my_part_pins = 1u; // include yourself before the move
        uint32_t other_part_pins = 0u;
        for (dim_t i = hedges_offsets[hedge_idx]; i < hedges_offsets[hedge_idx + 1]; i++) {
            const uint32_t side = classify(nodes_base + hedges[i]); // pins are node idxs local to a multi-start, rebase them
            if (side == 0u) my_part_pins++;
            else if (side == 1u) other_part_pins++;
        }
        // NOTE: in CUDA 'sibling_move_gain' also applies the hedge's weight, and nvcc fuses that product with "gain +=" into a
        //       single FFMA, hence the explicit 'fmaf'
        const uint32_t lane = LANE_OF(t);
        gain_lanes[lane] = std::fmaf(hedge_weights[hedge_idx], static_cast<float>(sibling_move_gain(my_part_pins, other_part_pins)), gain_lanes[lane]);
    }
    return lanesReduceSumLN0<float>(gain_lanes);
}

// find in which partition (between a pair that was just bisected) each node wants to stay in
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
void label_propagation_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions, // partitions[idx] -> the partition node idx is part of
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    bool* __restrict__ moves, // moves[idx] -> true if the node would like to move to the other side of the bisection
    float* __restrict__ scores // scores[idx] -> gain for node idx's move
) {
    /*
    * Idea:
    * - one node per iteration
    * - each node visits the pins of each touching hedge and counts how many sibling-pair pins lie on its current side vs the other
    * - the score is the exact improvement in weighted minority pin-cut if the node moves
    */

    // STYLE: one node per iteration!
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t node = 0; node < batch_nodes; node++) {
        const uint32_t my_start = node / num_nodes; // multi-start owning this node
        if (!active[my_start]) continue; // this multi-start already converged
        const uint32_t nodes_base = my_start * num_nodes;
        const uint32_t my_node = node - nodes_base; // node idx local to its own multi-start

        const uint32_t my_partition = partitions[node];
        const uint32_t other_partition = (my_partition & 1u) == 0 ? my_partition + 1 : my_partition - 1;

        // the hypergraph is shared by every multi-start, index it with the local node idx
        const uint32_t* my_touching = touching + touching_offsets[my_node];
        const uint32_t touching_count = touching_offsets[my_node + 1] - touching_offsets[my_node];

        const float gain = touchingSiblingGain(
            hedges, hedges_offsets, hedge_weights, my_touching, touching_count, nodes_base,
            [&](uint32_t pin) -> uint32_t {
                if (pin == node) return UINT32_MAX;
                if (partitions[pin] == my_partition) return 0u; // the pin is on my same side
                if (partitions[pin] == other_partition) return 1u; // the pin is on the other side of the bisection
                return UINT32_MAX;
            }
        );

        if (gain <= 0.0f) {
            moves[node] = false;
            scores[node] = 0.0f;
        } else {
            moves[node] = true;
            scores[node] = gain;
        }
    }
}

// transform moves into a sequence of events
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
void label_move_events_kernel(
    const bool* __restrict__ moves,
    const float* __restrict__ scores,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    uint32_t* __restrict__ even_ev_partition,
    float* __restrict__ even_ev_score,
    uint32_t* __restrict__ even_ev_node,
    uint32_t* __restrict__ odd_ev_partition,
    float* __restrict__ odd_ev_score,
    uint32_t* __restrict__ odd_ev_node
) {
    // STYLE: one node (move) per iteration!
    // NOTE: events are not compacted, node "idx" owns event slot "idx" in one of the two lists
    // => the slots it does not own carry UINT32_MAX as partition, so they sort behind every real event of their own multi-start
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t idx = 0; idx < batch_nodes; idx++) {
        even_ev_partition[idx] = UINT32_MAX;
        even_ev_score[idx] = 0.0f;
        even_ev_node[idx] = batch_nodes; // the one slot past the nodes is the reverse-map's scratch bin
        odd_ev_partition[idx] = UINT32_MAX;
        odd_ev_score[idx] = 0.0f;
        odd_ev_node[idx] = batch_nodes;

        if (!active[idx / num_nodes]) continue; // this multi-start already converged
        if (!moves[idx]) continue;

        const float score = scores[idx];
        const uint32_t part = partitions[idx];

        if ((part & 1u) == 0) {
            even_ev_partition[idx] = part >> 1; // stored as p/2
            even_ev_score[idx] = -score; // temporarily negative, to use an ascending sort as if it were descending
            even_ev_node[idx] = idx;
        } else {
            odd_ev_partition[idx] = part >> 1; // stored as (p-1)/2
            odd_ev_score[idx] = -score; // temporarily negative, to use an ascending sort as if it were descending
            odd_ev_node[idx] = idx;
        }
    }
}

// update move gains in sequence and merge them into pairs
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
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
    const uint32_t num_nodes,
    const uint32_t batch_size,
    float* __restrict__ even_event_score
) {
    /*
    * Idea:
    * - same as label_propagation_kernel, but check first if your neighbor will change partition before you (in-sequence)
    * - moreover, now we need to shuffle between the two lists of even/odd events and their relative ranks
    */

    // STYLE: one event per iteration!
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t slot_idx = 0; slot_idx < 2u * batch_nodes; slot_idx++) {
        const bool even = slot_idx < batch_nodes;
        const uint32_t event_id = even ? slot_idx : slot_idx - batch_nodes;
        // |
        const uint32_t* my_part_event_offsets = even ? part_even_event_offsets : part_odd_event_offsets;
        const uint32_t* my_ranks = even ? even_ranks : odd_ranks;
        const uint32_t* my_event_node = even ? even_event_node : odd_event_node;
        // |
        const uint32_t* other_part_event_offsets = even ? part_odd_event_offsets : part_even_event_offsets;
        const uint32_t* other_ranks = even ? odd_ranks : even_ranks;

        const uint32_t my_node = my_event_node[event_id];
        if (my_node >= batch_nodes) continue; // empty slot, no event here

        const uint32_t nodes_base = (my_node / num_nodes) * num_nodes; // multi-start owning this event

        const uint32_t my_partition = partitions[my_node];
        const uint32_t other_partition = (my_partition & 1u) == 0 ? my_partition + 1 : my_partition - 1;

        const uint32_t my_part_events_offset = my_part_event_offsets[my_partition >> 1];
        const uint32_t my_part_event_rank = event_id - my_part_events_offset;
        const uint32_t other_part_events_offset = other_part_event_offsets[other_partition >> 1];
        const uint32_t other_part_events_count = other_part_event_offsets[(other_partition >> 1) + 1] - other_part_event_offsets[other_partition >> 1];
        if (my_part_event_rank >= other_part_events_count) continue; // omit events that don't have a pair (those exceeding the minimum of the events count between the two partitions)

        // the hypergraph is shared by every multi-start, index it with the local node idx
        const uint32_t my_local_node = my_node - nodes_base;
        const uint32_t* my_touching = touching + touching_offsets[my_local_node];
        const uint32_t touching_count = touching_offsets[my_local_node + 1] - touching_offsets[my_local_node];

        const float gain = touchingSiblingGain(
            hedges, hedges_offsets, hedge_weights, my_touching, touching_count, nodes_base,
            [&](uint32_t pin) -> uint32_t {
                if (pin == my_node) return UINT32_MAX;
                if (partitions[pin] == my_partition) {
                    if (my_ranks[pin] == UINT32_MAX || my_ranks[pin] > event_id) return 0u; // same side, didn't move before me
                    return 1u; // was on my same side, but moved before me
                }
                if (partitions[pin] == other_partition) {
                    if (other_ranks[pin] == UINT32_MAX || other_ranks[pin] - other_part_events_offset > my_part_event_rank) return 1u; // other side, didn't move before me
                    return 0u; // was on the other side, but moved before me (or -with- me)
                }
                return UINT32_MAX;
            }
        );

        // accumulate everything in the even partition's score
        // NOTE: two adds land on the same slot, starting from zero, so their order does not change the float result
        const uint32_t idx = even ? event_id : part_even_event_offsets[my_partition >> 1] + my_part_event_rank;
        atomic_add<float>(&even_event_score[idx], gain);
    }
}

// apply pair swaps (handle each node of a pair at the same time, from the "even" side)
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
void apply_move_events_kernel(
    const uint32_t* __restrict__ apply_up_to, // apply_up_to[p/2] -> last absolute event idx to apply for partition p or p+1
    const uint32_t* __restrict__ even_event_part, // even_event_part[event idx] -> p/2 of the event
    const uint32_t* __restrict__ even_event_node, // even_event_node[event idx] -> node of the event
    const uint32_t* __restrict__ part_even_event_offsets, // part_even_event_offsets[p/2] -> first event idx among even events for part p
    const uint32_t* __restrict__ part_odd_event_offsets, // part_odd_event_offsets[p/2] -> first event idx among odd events for part p
    const uint32_t* __restrict__ odd_event_node, // odd_event_node[event idx] -> node of the event
    const uint32_t num_nodes,
    const uint32_t batch_size,
    uint32_t* __restrict__ partitions
) {
    // STYLE: one event per iteration!
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t my_event = 0; my_event < batch_nodes; my_event++) {
        // check if the move is to apply
        const uint32_t my_part_half = even_event_part[my_event];
        if (my_part_half == UINT32_MAX) continue; // empty slot, no event here
        const uint32_t apply_idx = apply_up_to[my_part_half];
        if (my_event > apply_idx || apply_idx == UINT32_MAX) continue;

        // check if the move exists for both even and odd events (the maximum could have been overconfident)
        const uint32_t my_part_offset = part_even_event_offsets[my_part_half];
        const uint32_t other_part_offset = part_odd_event_offsets[my_part_half];
        const uint32_t not_other_part_offset = part_odd_event_offsets[my_part_half + 1];
        const uint32_t other_event = other_part_offset + (my_event - my_part_offset);
        if (other_event >= not_other_part_offset) continue;

        const uint32_t my_node = even_event_node[my_event];
        const uint32_t other_node = odd_event_node[other_event];

        partitions[my_node] = (my_part_half << 1) + 1u; // partitions[my_node] was even by construction
        partitions[other_node] = (my_part_half << 1);
    }
}

// for every node, if its current partition improved in split cost w.r.t. the best so far, overwrite its best partition with the current one
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
void update_best_partitions_kernel(
    const uint32_t* __restrict__ partitions,
    const float* __restrict__ cutnet,
    const float* __restrict__ last_best_cutnet,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    uint32_t* __restrict__ last_best_partitions
) {
    // STYLE: one node per iteration!
    // NOTE: partitions and cutnet are both composite-indexed, so nothing needs decoding here
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < batch_nodes; node++) {
        const uint32_t part = partitions[node];
        const uint32_t parent_part = part / 2;

        // if your partition pair's split cost improved, update your last best partition
        if (last_best_cutnet[parent_part] > cutnet[parent_part])
            last_best_partitions[node] = part;
    }
}


// TREE ORDERING

// for each pair of partitions p*2 and p*2+1 in the bisection tree, evaluate its total connection weight with the sibling subtree of p
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
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
    const uint32_t batch_size,
    float* __restrict__ slot_scores // slot_scores[idx] -> strength contributed by the node in ordering slot idx, towards its partition
) {
    /*
    * Idea:
    * - one node per iteration
    * - each node visits its neighbors and if they are in the sibling tree of its parent partition, then it accumulates their connections strength in favor of its partition
    * - the higher-scoring partition gets to be near the sibling
    */

    // STYLE: one node (ordering slot) per iteration!
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t slot_idx = 0; slot_idx < batch_nodes; slot_idx++) {
        const uint32_t nodes_base = (slot_idx / num_nodes) * num_nodes; // multi-start owning this ordering slot

        const uint32_t my_node = order[slot_idx];
        const uint32_t my_part_half = ord_part[slot_idx] >> 1;
        const uint32_t sibling_part_half = (my_part_half & 1u) == 0 ? my_part_half + 1 : my_part_half - 1;

        // the hypergraph is shared by every multi-start, index it with the local node idx
        const uint32_t my_local_node = my_node - nodes_base;
        const uint32_t* my_touching = touching + touching_offsets[my_local_node];
        const uint32_t touching_count = touching_offsets[my_local_node + 1] - touching_offsets[my_local_node];
        float score_lanes[WARP_SIZE] = {}; // score_lanes[lane] -> strength accumulated by that lane

        forEachTouchingPin(
            hedges, hedges_offsets, hedge_weights, my_touching, touching_count, nodes_base,
            [&](uint32_t lane, uint32_t pin, float my_hedge_weight) {
                const uint32_t pin_part_half = partitions[pin] >> 1;
                if (pin_part_half == sibling_part_half) // the pin is in the sibling subtree
                    score_lanes[lane] += my_hedge_weight;
            }
        );

        // NOTE: one score per slot, summed per partition afterwards, so that the summation order is fixed
        slot_scores[slot_idx] = lanesReduceSumLN0<float>(score_lanes);
    }
}

// flag subtrees that need to be internally reversed
// SEQUENTIAL COMPLEXITY: p
// PARALLEL OVER: p
void flag_reversals_kernel(
    const float* __restrict__ sibling_score,
    const uint32_t num_parts,
    const uint32_t batch_size,
    bool* __restrict__ reverse
) {
    // STYLE: one (half) partition per iteration!
    // NOTE: part is a composite "b*num_parts + p", whose parity matches p's since num_parts is always even
    const uint32_t batch_parts = batch_size * num_parts;
    #pragma omp parallel for schedule(static) if(batch_parts > PARALLEL_GRAIN)
    for (uint32_t part = 0; part < batch_parts; part++) {
        const float left_score = sibling_score[part*2];
        const float right_score = sibling_score[part*2 + 1];

        const bool is_sibling_to_the_left = (part & 1u) == 1;

        reverse[part] = (is_sibling_to_the_left && left_score < right_score) || (!is_sibling_to_the_left && left_score > right_score);
    }
}

// retire the multi-starts left with no strictly improving balanced prefix
// SEQUENTIAL COMPLEXITY: p
// PARALLEL OVER: batch_size
void labelprop_activity_kernel(
    const uint32_t* __restrict__ apply_up_to, // apply_up_to[p/2] -> last absolute event idx to apply for partition p or p+1
    const uint32_t num_part_pairs, // partition pairs per multi-start, that is num_parts/2
    const uint32_t batch_size,
    uint8_t* __restrict__ active // active[start] -> 0 once that multi-start converged
) {
    // STYLE: one multi-start per iteration!
    #pragma omp parallel for schedule(static)
    for (uint32_t my_start = 0; my_start < batch_size; my_start++) {
        if (!active[my_start]) continue; // this multi-start already converged
        const uint32_t* my_apply_up_to = apply_up_to + my_start * num_part_pairs;
        bool improving = false;
        for (uint32_t pair = 0; pair < num_part_pairs && !improving; pair++)
            improving = my_apply_up_to[pair] != UINT32_MAX;
        if (!improving) active[my_start] = 0u;
    }
}

// reverse elements in each flagged segment
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
void apply_reversals_kernel(
    const uint32_t* __restrict__ segment, // segment[n] -> segment idx of which data[i] is part
    const uint32_t* __restrict__ offsets, // offsets[i] -> start idx of the i-th segment in "data"
    const bool* __restrict__ flag, // flag[i] -> true if the i-th segment in "data" is to be reversed
    const uint32_t size, // size of data
    uint32_t* __restrict__ data // data[n] -> segments of values being reversed
) {
    // STYLE: one node per iteration!
    #pragma omp parallel for schedule(static) if(size > PARALLEL_GRAIN)
    for (uint32_t idx = 0; idx < size; idx++) {
        const uint32_t seg = segment[idx];
        if (!flag[seg]) continue;

        const uint32_t seg_begin = offsets[seg];
        const uint32_t seg_end = offsets[seg + 1];
        const uint32_t seg_half_size = (seg_end - seg_begin) >> 1;
        const uint32_t seg_idx = idx - seg_begin;
        if (seg_idx >= seg_half_size) continue;

        std::swap(data[idx], data[seg_end - seg_idx - 1]);
    }
}

// compute the weighted span of each hedge
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void measure_sequence_locality_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ order_idx,
    const uint32_t num_hedges,
    float* __restrict__ hedge_span
) {
    // STYLE: one hedge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
        uint32_t min_pin_idx = UINT32_MAX;
        uint32_t max_pin_idx = 0u;

        for (dim_t i = hedges_offsets[hedge_idx]; i < hedges_offsets[hedge_idx + 1]; i++) {
            const uint32_t idx = order_idx[hedges[i]];
            min_pin_idx = std::min(min_pin_idx, idx);
            max_pin_idx = std::max(max_pin_idx, idx);
        }

        if (min_pin_idx == UINT32_MAX)
            hedge_span[hedge_idx] = 0.0f;
        else
            hedge_span[hedge_idx] = (max_pin_idx - min_pin_idx) * hedge_weights[hedge_idx];
    }
}
