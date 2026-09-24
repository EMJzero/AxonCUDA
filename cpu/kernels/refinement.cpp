#include <bit>
#include <vector>

#include "refinement.hpp"
#include "constants.hpp"
#include "utils.hpp"

// NOTE: in CUDA per-partition counters are updated atomically from every hedge; here each thread counts into its own
//       partition counters, and flushes them into the shared ones once, at the end of the loop

// add each thread's partition counters to the shared ones
static inline void flush_partition_counters(const std::vector<int64_t> &local, uint32_t* __restrict__ counters) {
    for (uint32_t part = 0; part < local.size(); part++)
        if (local[part] != 0) atomic_add<uint32_t>(&counters[part], (uint32_t)local[part]);
}

// for each hyperedge, count how many of its pins are in each partition
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void pins_per_partition_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ pins_per_partitions, // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
    uint32_t* __restrict__ partitions_inbound_sizes // partitions_inbound_sizes[part] -> number of inbound_pins_per_partitions for 'part' that are not zero => will be incorrect as of here (also including outbounds)
) {
    #pragma omp parallel
    {
        std::vector<int64_t> my_inbound_sizes(num_partitions, 0);

        // STYLE: one hedge per iteration!
        // NOTE: each iteration owns its hedge's row, no atomics needed
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
            uint32_t *my_pins_per_partitions = pins_per_partitions + static_cast<dim_t>(hedge_idx) * num_partitions;
            for (dim_t i = hedges_offsets[hedge_idx]; i < hedges_offsets[hedge_idx + 1]; i++) {
                const uint32_t part = partitions[hedges[i]];
                if (my_pins_per_partitions[part]++ == 0) my_inbound_sizes[part]++;
            }
        }

        flush_partition_counters(my_inbound_sizes, partitions_inbound_sizes);
    }
}

// for each hyperedge, flag the partitions holding its pins in the bitmaps, and count them
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void sparse_pins_per_partition_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t num_hedges,
    const uint32_t ppp_per_hedge, // ppp_per_hedge = ceil(num_partitions / 64) [note: 64 = BITMAP_CAPACITY]
    bitmap* __restrict__ ppp_offsets // ppp_offsets[hedge-idx * ppp_per_hedge + part-idx / 64] -> bitmap to access the pin count for all (hedge, part / 64), ... (hedge, part / 64 + 63) pairs
) {
    // STYLE: one hedge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
        bitmap *my_ppp_offsets = ppp_offsets + static_cast<dim_t>(hedge_idx) * ppp_per_hedge;
        for (dim_t i = hedges_offsets[hedge_idx]; i < hedges_offsets[hedge_idx + 1]; i++) {
            const uint32_t part = partitions[hedges[i]];
            my_ppp_offsets[part >> BITMAP_CAPLOG].flg |= 1ull << (part & (BITMAP_CAPACITY - 1u));
        }
        for (uint32_t b = 0; b < ppp_per_hedge; b++)
            my_ppp_offsets[b].cnt = std::popcount(my_ppp_offsets[b].flg);
    }
}

// given the sparse pins-per-partiton bitmaps, write the actual ppp counters in the segmented array
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void sparse_pins_per_partition_write_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const bitmap* __restrict__ ppp_offsets, // ppp_offsets[hedge-idx * ppp_per_hedge + part-idx / 64] -> bitmap to access the pin count for all (hedge, part / 64), ... (hedge, part / 64 + 63) pairs
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    const uint32_t ppp_per_hedge, // ppp_per_hedge = ceil(num_partitions / 64) [note: 64 = BITMAP_CAPACITY]
    uint32_t* __restrict__ ppp, // ppp[...] -> number of pins of partition p owned by hedge e
    uint32_t* __restrict__ partitions_incident_sizes // partitions_incident_sizes[part] -> number of pins-per-partitions for 'part' that are not zero
) {
    const sparse_ppp accessor { ppp_offsets, ppp, ppp_per_hedge, num_partitions };

    #pragma omp parallel
    {
        std::vector<int64_t> my_incident_sizes(num_partitions, 0);

        // STYLE: one hedge per iteration!
        // NOTE: each iteration owns its hedge's segment, no atomics needed
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
            for (dim_t i = hedges_offsets[hedge_idx]; i < hedges_offsets[hedge_idx + 1]; i++) {
                const uint32_t part = partitions[hedges[i]];
                if ((*accessor.at(hedge_idx, part))++ == 0) my_incident_sizes[part]++;
            }
        }

        flush_partition_counters(my_incident_sizes, partitions_incident_sizes);
    }
}

// for each hyperedge, remove its sources from the pins per partition counts
// SEQUENTIAL COMPLEXITY: e
// PARALLEL OVER: e
template <typename PPP>
void inbound_pins_per_partition_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    const PPP inbound_pins_per_partitions, // (hedge idx, partition idx) -> number of inbound pins of that partition owned by this hedge
    uint32_t* __restrict__ partitions_inbound_sizes // partitions_inbound_sizes[part] -> number of inbound_pins_per_partitions for 'part' that are not zero
) {
    #pragma omp parallel
    {
        std::vector<int64_t> my_inbound_sizes(num_partitions, 0);

        // STYLE: one hedge per iteration!
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
            const dim_t hedge_start_idx = hedges_offsets[hedge_idx];
            for (dim_t i = hedge_start_idx; i < hedge_start_idx + srcs_count[hedge_idx]; i++) {
                const uint32_t src_part = partitions[hedges[i]];
                if ((*inbound_pins_per_partitions.at(hedge_idx, src_part))-- == 1) my_inbound_sizes[src_part]--;
            }
        }

        flush_partition_counters(my_inbound_sizes, partitions_inbound_sizes);
    }
}

// find moves of nodes from one partition to another that yield a positive gain
// SEQUENTIAL COMPLEXITY: n*h*partitions
// PARALLEL OVER: n
template <typename PPP>
void fm_refinement_gains_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const PPP pins_per_partitions, // (hedge idx, partition idx) -> number of pins of that partition owned by this hedge
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ partitions_sizes,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t randomizer,
    const uint32_t discount, // by how much to overshoot the size constraint when proposing moves
    const bool encourage_all_moves, // if true, even moves that don't fully disconnect an hyperedge receive a gain inversely proportional to how many pins remain
    uint32_t* __restrict__ moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx
    float* __restrict__ scores // scores[idx] -> gain for move in position idx
) {
    /*
    * Idea:
    * - one node per iteration
    * - histogram of losses (one bin per partition), filled with one scan of the touching hyperedges
    * - pick the best partition in each lane, PART_HIST_SIZE*WARP_SIZE partitions at a time, then the best among lanes
    *
    * NOTE: in CUDA each lane of the warp handles its own partitions, and ties between equal gains are broken by hashes within
    *       a lane, but by partition id between lanes: lanes are replayed here only to break ties the same way
    */

    #pragma omp parallel
    {
        std::vector<float> loss(num_partitions); // loss[part] -> weight of hedges that get connected to 'part' if the node moves there

        // STYLE: one node per iteration!
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t node_id = 0; node_id < num_nodes; node_id++) {
            const uint32_t my_partition = partitions[node_id];
            const uint32_t my_size = nodes_sizes[node_id];

            const uint32_t* my_touching = touching + touching_offsets[node_id];
            const uint32_t my_touching_count = (uint32_t)(touching_offsets[node_id + 1] - touching_offsets[node_id]);

            // handle the current partition first with its own scan of touching hyperedges
            // NOTE: summed lane by lane, then reduced along the same tree as 'warpReduceSum'
            float saving_lanes[WARP_SIZE] = {};
            for (uint32_t t = 0; t < my_touching_count; t++) {
                const uint32_t actual_hedge_idx = my_touching[t];
                const uint32_t my_pin_per_partition = pins_per_partitions.get(actual_hedge_idx, my_partition);
                const float my_hedge_weight = hedge_weights[actual_hedge_idx];
                // hedge connected to my partition: gain the hedge's weight iff moving would disconnect it from my partition (I am its last pin left there)
                if (!encourage_all_moves && my_pin_per_partition == 1)
                    saving_lanes[LANE_OF(t)] += my_hedge_weight;
                // VARIANT: give a little push to nodes leaving a partition with not just one, but few pins left for an hedge
                if (encourage_all_moves && my_pin_per_partition >= 1)
                    saving_lanes[LANE_OF(t)] += my_hedge_weight / (my_pin_per_partition * my_pin_per_partition);
            }
            const float saving = lanesReduceSum<float>(saving_lanes);

            // scan touching hyperedges
            // NOTE: interpret this as "for each hedge, see if you moving to a certain partition is something that they like or not"
            for (uint32_t part = 0; part < num_partitions; part++)
                loss[part] = 0.0f;
            for (uint32_t t = 0; t < my_touching_count; t++) {
                // hedge not yet connected to the partition: pay the hedge's weight iff moving there connects it to the new partition (I would become its first pin there)
                pins_per_partitions.add_where_empty(my_touching[t], hedge_weights[my_touching[t]], loss.data());
            }

            // every lane agrees on those, after each reduction between lanes...
            float best_gain = -FLT_MAX;
            uint32_t best_move = UINT32_MAX;

            // handle PART_HIST_SIZE*WARP_SIZE partitions at a time, that is partitions_per_lane per lane
            for (uint32_t curr_base_part = 0; curr_base_part < num_partitions; curr_base_part += PART_HIST_SIZE*WARP_SIZE) {
                // each lane handles, at once, min(PART_HIST_SIZE, partitions_per_lane) partitions, each partition is handled by exactly one lane
                const uint32_t partitions_to_handle = std::min(num_partitions - curr_base_part, PART_HIST_SIZE*WARP_SIZE); // ... to handle over all lanes
                const uint32_t partitions_per_lane = (partitions_to_handle + WARP_SIZE - 1) / WARP_SIZE; // ceiled
                const uint32_t lanes_with_one_less_partition = partitions_per_lane*WARP_SIZE - partitions_to_handle;

                // reduce max inside each lane
                float lane_gain[WARP_SIZE];
                uint32_t lane_move[WARP_SIZE];
                for (uint32_t lane_id = 0; lane_id < WARP_SIZE; lane_id++) {
                    const uint32_t my_part_count = partitions_per_lane - (lane_id >= WARP_SIZE - lanes_with_one_less_partition ? 1 : 0);
                    float my_best_gain = best_gain;
                    uint32_t my_best_move = best_move;
                    for (uint32_t p = 0; p < my_part_count; p++) {
                        const uint32_t part = curr_base_part + lane_id + p * WARP_SIZE;
                        const float gain = saving - loss[part];
                        // => pseudo-random tie-break via hashes
                        if (part != my_partition && partitions_sizes[part] + my_size - (my_size / discount) <= max_nodes_per_part && (gain > my_best_gain || gain == my_best_gain && hash_uint32(part + randomizer) > hash_uint32(my_best_move + randomizer))) {
                            my_best_gain = gain;
                            my_best_move = part;
                        }
                    }
                    lane_gain[lane_id] = my_best_gain;
                    lane_move[lane_id] = my_best_move;
                }

                // reduce max between lanes, the maximum by gain, then by partition id (as 'warpReduceArgMax')
                best_gain = lane_gain[0];
                best_move = lane_move[0];
                for (uint32_t lane_id = 1; lane_id < WARP_SIZE; lane_id++) {
                    if (lane_gain[lane_id] > best_gain || lane_gain[lane_id] == best_gain && lane_move[lane_id] > best_move) {
                        best_gain = lane_gain[lane_id];
                        best_move = lane_move[lane_id];
                    }
                }
            }

            moves[node_id] = best_move;
            scores[node_id] = best_gain;
        }
    }
}

// find the gain of each move under the HP that all higher-score moves have been applied
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
template <typename PPP>
void fm_refinement_cascade_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ move_ranks, // move_ranks[node_idx] -> i (ranking by score) of the move proposed by the idx node
    const uint32_t* __restrict__ moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx (DO NOT SORT)
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const PPP pins_per_partitions, // (hedge idx, partition idx) -> number of pins of that partition owned by this hedge
    const uint32_t num_nodes,
    const bool encourage_all_moves,
    float* __restrict__ scores // scores[move_ranks[node_idx]] -> gain for node idx's move
) {
    /*
    * Idea (from HyperG):
    * - moves are sorted from the highest to lowest score
    * - greedy assumption: all moves with a higher score get applied
    * - therefore, for each move, recompute its score (gain) like in "fm_refinement_gains_kernel", but now assuming
    *   each node with a higher score changed partition to the one specified by the move
    * - write the new score in place of the previous one for each move, this then enables a scan to find the sequence of
    *   "moves as if applied in isolation" that yields the highest total gain when applied all together
    *
    * NOTE: re-evaluate here EVERY move, even negative-gain ones, because after applying all previous moves, they may become positive-gained!
    */

    // STYLE: one node (move) per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t node_id = 0; node_id < num_nodes; node_id++) {
        const uint32_t my_move_part = moves[node_id];
        // no need to update invalid moves
        if (my_move_part == UINT32_MAX) continue;

        const uint32_t my_partition = partitions[node_id];
        const uint32_t my_move_rank = move_ranks[node_id];

        float score = 0.0f;

        // scan touching hyperedges
        for (dim_t t = touching_offsets[node_id]; t < touching_offsets[node_id + 1]; t++) {
            const uint32_t actual_hedge_idx = touching[t];
            const float my_hedge_weight = hedge_weights[actual_hedge_idx];
            const uint32_t my_curr_part_counter = pins_per_partitions.get(actual_hedge_idx, my_partition);
            const uint32_t my_move_part_counter = pins_per_partitions.get(actual_hedge_idx, my_move_part);
            int32_t my_curr_part_counter_delta = 0;
            int32_t my_move_part_counter_delta = 0;
            for (dim_t i = hedges_offsets[actual_hedge_idx]; i < hedges_offsets[actual_hedge_idx + 1]; i++) {
                const uint32_t pin = hedges[i];
                if (move_ranks[pin] < my_move_rank) { // speculation: better-ranked move -> applied
                    // NOTE: invalid moves should all have a lower score and thus a higher rank than all others, never being see here
                    const uint32_t new_pin_partition = moves[pin];
                    const uint32_t prev_pin_partition = partitions[pin];
                    if (new_pin_partition == my_partition)
                        my_curr_part_counter_delta++;
                    else if (new_pin_partition == my_move_part)
                        my_move_part_counter_delta++;
                    if (prev_pin_partition == my_partition)
                        my_curr_part_counter_delta--;
                    else if (prev_pin_partition == my_move_part)
                        my_move_part_counter_delta--;
                }
            }
            // gain the hedge's weight iff moving would disconnect the hedge from my partition (I am its last pin left there)
            const uint32_t true_curr_part_counter = my_curr_part_counter + my_curr_part_counter_delta;
            if (!encourage_all_moves && true_curr_part_counter == 1)
                score += my_hedge_weight;
            // pay the hedge's weight iff moving there connects the hedge to the new partition (I would become its first pin there)
            if (my_move_part_counter + my_move_part_counter_delta == 0)
                score -= my_hedge_weight;
            // VARIANT: give a little push to nodes leaving a partition with not just one, but few pins left for an hedge
            if (encourage_all_moves && true_curr_part_counter >= 1)
                score += my_hedge_weight / (true_curr_part_counter * true_curr_part_counter);
        }

        scores[my_move_rank] = score;
    }
}

// apply moves with a positive gain
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
void fm_refinement_apply_kernel(
    const uint32_t* __restrict__ moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx (DO NOT SORT)
    const uint32_t* __restrict__ move_ranks, // move_ranks[node_idx] -> i (ranking by score) of the move proposed by the idx node
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ nodes_pins,
    const uint32_t num_nodes,
    const uint32_t num_good_moves, // idx + 1 of the maximum in the updated scores
    uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    uint32_t* __restrict__ partitions_sizes,
    uint32_t* __restrict__ partitions_pins
) {
    // STYLE: one node (move) per iteration!
    #pragma omp parallel for schedule(static) if(num_nodes > PARALLEL_GRAIN)
    for (uint32_t node_id = 0; node_id < num_nodes; node_id++) {
        // stop at the last gain-increasing move
        if (move_ranks[node_id] >= num_good_moves || moves[node_id] == UINT32_MAX) continue;

        const uint32_t my_partition = partitions[node_id];
        const uint32_t my_move_part = moves[node_id];

        // update partition sizes
        atomic_sub<uint32_t>(&partitions_sizes[my_partition], nodes_sizes[node_id]);
        atomic_add<uint32_t>(&partitions_sizes[my_move_part], nodes_sizes[node_id]);

        // update partition pins
        atomic_sub<uint32_t>(&partitions_pins[my_partition], nodes_pins[node_id]);
        atomic_add<uint32_t>(&partitions_pins[my_move_part], nodes_pins[node_id]);

        // update my partition
        partitions[node_id] = my_move_part;
    }
}

// transform moves into a sequence of size-altering events for capacity constraint checks, two per move
// => events are written in rank order, so that a stable sort by partition sorts them by (partition, rank)
// NOTE: the (partition, rank) key of an event does not depend on the quantity being accounted for, hence sizes and
//       inbound pins share one key array and one sort
void build_size_events_kernel(
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ node_of_rank, // node_of_rank[rank] -> node whose move has that rank
    const uint32_t* __restrict__ moving_ranks, // moving_ranks[idx] -> rank of the idx-th valid move (in rank order)
    const uint32_t* __restrict__ partitions,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ nodes_pins,
    const uint32_t num_moving,
    uint32_t* __restrict__ ev_partition,
    uint32_t* __restrict__ ev_index,
    int32_t* __restrict__ ev_delta,
    int32_t* __restrict__ ev_pins_delta
) {
    // STYLE: one (valid) move per iteration!
    #pragma omp parallel for schedule(static) if(num_moving > PARALLEL_GRAIN)
    for (uint32_t idx = 0; idx < num_moving; idx++) {
        const uint32_t rank = moving_ranks[idx];
        const uint32_t node = node_of_rank[rank];
        const int32_t size = static_cast<int32_t>(nodes_sizes[node]);
        const int32_t pins = static_cast<int32_t>(nodes_pins[node]);

        // first event: node leaves its current partition
        const dim_t e0 = 2ull * idx;
        // second event: node enters its destination partition
        const dim_t e1 = e0 + 1;

        ev_partition[e0] = partitions[node];
        ev_index[e0] = rank;
        ev_delta[e0] = -size;
        ev_pins_delta[e0] = -pins;

        ev_partition[e1] = moves[node];
        ev_index[e1] = rank;
        ev_delta[e1] = size;
        ev_pins_delta[e1] = pins;
    }
}

// mark moves that are valid points in the sequence w.r.t. an additive per-partition constraint
// NOTE: "sizes" here is any additive quantity, this kernel serves both nodes sizes and nodes pins
void flag_size_events_kernel(
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const int32_t* __restrict__ ev_delta,
    const uint32_t* __restrict__ partitions_sizes,
    const dim_t num_events,
    const uint32_t max_per_part,
    int32_t* __restrict__ valid_moves // initialized with 0s
) {
    /*
    * Idea:
    * - for each move, compute how many partitions it brings to be invalid or it brings back to a valid state
    * - then compute the number of invalid partitions at each point in time as the prefix sum of the number going from ok to not-ok (+1) and not-ok to ok (-1)
    */

    // STYLE: one event per iteration!
    #pragma omp parallel for schedule(static) if(num_events > PARALLEL_GRAIN)
    for (dim_t ev = 0; ev < num_events; ev++) {
        const uint32_t part = ev_partition[ev];
        const uint32_t rank = ev_index[ev];

        const int32_t base_size = static_cast<int32_t>(partitions_sizes[part]);
        const int32_t curr_size = base_size + ev_delta[ev];
        const int32_t max_size = static_cast<int32_t>(max_per_part);
        const int32_t new_excess = std::max(curr_size - max_size, 0); // by how much we now exceed the constraint

        const uint32_t pred_part = ev > 0 ? ev_partition[ev - 1] : UINT32_MAX; // partition acted upon by the event before this one
        const int32_t old_excess = std::max(pred_part != part ? base_size - max_size : base_size + ev_delta[ev - 1] - max_size, 0); // by how much we previously were exceeding the constraint

        atomic_add<int32_t>(&valid_moves[rank], new_excess - old_excess); // accumulate how much we recovered from constraint violations with this move (valid -> valid = 0, invalid -> valid = <0, valid -> invalid = >0)
    }
}

// for every move, generate two events for every inbound hedge, one removing it front the src, one adding it back
// => events are written in rank order, so that a stable sort by (partition, hedge) sorts them by (partition, hedge, rank)
// SEQUENTIAL COMPLEXITY: n*h (h -> inbound only)
// PARALLEL OVER: n
void build_hedge_events_kernel(
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ node_of_rank, // node_of_rank[rank] -> node whose move has that rank
    const uint32_t* __restrict__ moving_ranks, // moving_ranks[idx] -> rank of the idx-th valid move (in rank order)
    const uint32_t* __restrict__ partitions,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const dim_t* __restrict__ ev_offsets, // ev_offsets[idx] -> first event of the idx-th valid move (in rank order)
    const uint32_t num_moving,
    uint32_t* __restrict__ ev_partition,
    uint32_t* __restrict__ ev_index,
    uint32_t* __restrict__ ev_hedge,
    int32_t* __restrict__ ev_delta
) {
    // STYLE: one (valid) move per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK) if(num_moving > PARALLEL_GRAIN)
    for (uint32_t idx = 0; idx < num_moving; idx++) {
        const uint32_t rank = moving_ranks[idx];
        const uint32_t node = node_of_rank[rank];
        const uint32_t src_part = partitions[node];
        const uint32_t dst_part = moves[node];
        const uint32_t* inbound = touching + touching_offsets[node];
        dim_t ev = ev_offsets[idx];
        for (uint32_t i = 0; i < inbound_count[node]; i++, ev += 2) {
            const uint32_t hedge = inbound[i];
            // first event: hedge does not touches one less time the node's current partition
            ev_partition[ev] = src_part;
            ev_index[ev] = rank;
            ev_hedge[ev] = hedge;
            ev_delta[ev] = -1;
            // second event: hedge touches one more time the node's destination partition
            ev_partition[ev + 1] = dst_part;
            ev_index[ev + 1] = rank;
            ev_hedge[ev + 1] = hedge;
            ev_delta[ev + 1] = +1;
        }
    }
}

// for every inbound hedge event that adds/removes an inbound hedge to a partition, count a new inbound size event
template <typename PPP>
void count_inbound_size_events_kernel(
    const PPP partitions_inbound_counts, // this is (inbound) pins_per_partition, index it by (hedge_idx, partition_idx)
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_hedge,
    const int32_t* __restrict__ ev_delta,
    const dim_t num_events,
    dim_t* __restrict__ inbound_size_events_offsets // init. to zero
) {
    // STYLE: one event per iteration!
    #pragma omp parallel for schedule(static) if(num_events > PARALLEL_GRAIN)
    for (dim_t ev = 0; ev < num_events; ev++) {
        const uint32_t part = ev_partition[ev];
        const uint32_t hedge = ev_hedge[ev];
        const uint32_t init_hedge_inbound_count = partitions_inbound_counts.get(hedge, part);

        uint32_t prev_hedge_inbound_count = init_hedge_inbound_count;
        if (ev > 0 && ev_partition[ev - 1] == part && ev_hedge[ev - 1] == hedge) // if the previous sum was about the same hedge as mine, consider its updated count in the sequence
            prev_hedge_inbound_count += ev_delta[ev - 1];
        const uint32_t curr_hedge_inbound_count = init_hedge_inbound_count + ev_delta[ev];

        inbound_size_events_offsets[ev + 1] = (prev_hedge_inbound_count == 0 && curr_hedge_inbound_count > 0 || prev_hedge_inbound_count > 0 && curr_hedge_inbound_count == 0) ? 1 : 0; // +1 to do an inclusive scan and keep the final count
    }
}

// for every inbound hedge event that adds/removes an inbound hedge to a partition, create a new inbound size event
template <typename PPP>
void build_inbound_size_events_kernel(
    const PPP partitions_inbound_counts, // this is (inbound) pins_per_partition, index it by (hedge_idx, partition_idx)
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const uint32_t* __restrict__ ev_hedge,
    const int32_t* __restrict__ ev_delta,
    const dim_t* __restrict__ inbound_size_events_offsets,
    const dim_t num_events,
    uint32_t* __restrict__ new_ev_partition,
    uint32_t* __restrict__ new_ev_index,
    int32_t* __restrict__ new_ev_delta
) {
    // STYLE: one event per iteration!
    #pragma omp parallel for schedule(static) if(num_events > PARALLEL_GRAIN)
    for (dim_t ev = 0; ev < num_events; ev++) {
        // NOTE: the counting kernel already flagged which events produce a new one
        if (inbound_size_events_offsets[ev + 1] == inbound_size_events_offsets[ev]) continue;
        const uint32_t part = ev_partition[ev];
        const uint32_t hedge = ev_hedge[ev];
        const uint32_t init_hedge_inbound_count = partitions_inbound_counts.get(hedge, part);
        const uint32_t curr_hedge_inbound_count = init_hedge_inbound_count + ev_delta[ev];

        const dim_t new_ev_offset = inbound_size_events_offsets[ev];
        new_ev_partition[new_ev_offset] = part;
        new_ev_index[new_ev_offset] = ev_index[ev];
        new_ev_delta[new_ev_offset] = curr_hedge_inbound_count > 0 ? 1 : -1; // 0 -> >0 is an addition, >0 -> 0 a removal
    }
}

// mark moves that are valid points in the sequence w.r.t. inbound constraints
void flag_inbound_events_kernel(
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const int32_t* __restrict__ ev_delta,
    const uint32_t* __restrict__ partitions_inbound_sizes, // partitions_inbound_sizes[part] = size of the inbound set for part
    const dim_t num_events,
    int32_t* __restrict__ valid_moves // initialized with 0s
) {
    /*
    * HP: always start from a VALID state
    * How: very much like 'flag_size_events_kernel'
    */

    // STYLE: one event per iteration!
    #pragma omp parallel for schedule(static) if(num_events > PARALLEL_GRAIN)
    for (dim_t ev = 0; ev < num_events; ev++) {
        const uint32_t part = ev_partition[ev];
        const uint32_t rank = ev_index[ev];

        const int32_t base_size = static_cast<int32_t>(partitions_inbound_sizes[part]);
        const int32_t curr_size = base_size + ev_delta[ev];
        const int32_t max_size = static_cast<int32_t>(max_inbound_per_part);
        const bool is_valid = curr_size <= max_size; // true iff after this event the partition's inbound set size is valid

        const uint32_t pred_part = ev > 0 ? ev_partition[ev - 1] : UINT32_MAX; // partition acted upon by the event before this one
        const bool was_valid = pred_part != part || base_size + ev_delta[ev - 1] <= max_size; // true iff before this event the partition's inbound set size is valid

        if (was_valid && !is_valid) // this event made the partition invalid -> track a +1 in invalid partitions as of this event
            atomic_add<int32_t>(&valid_moves[rank], 1);
        if (!was_valid && is_valid) // this event made the partition valid -> track a -1 in invalid partitions as of this event
            atomic_sub<int32_t>(&valid_moves[rank], 1);
    }
}

// straight up compute inbound set sizes from hedges
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void inbound_sets_size_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ partitions_inbound_sizes
) {
    #pragma omp parallel
    {
        std::vector<uint32_t> seen(num_partitions, UINT32_MAX); // seen[part] -> last hedge whose destinations included 'part'
        std::vector<int64_t> my_inbound_sizes(num_partitions, 0);

        // STYLE: one hedge per iteration!
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
            for (dim_t i = hedges_offsets[hedge_idx] + srcs_count[hedge_idx]; i < hedges_offsets[hedge_idx + 1]; i++) {
                const uint32_t part = partitions[hedges[i]];
                if (seen[part] == hedge_idx) continue;
                seen[part] = hedge_idx;
                my_inbound_sizes[part]++;
            }
        }

        flush_partition_counters(my_inbound_sizes, partitions_inbound_sizes);
    }
}


// EXPLICIT INSTANTIATIONS

#define INSTANTIATE_PPP_KERNELS(PPP) \
    template void inbound_pins_per_partition_kernel<PPP>(const uint32_t* __restrict__, const dim_t* __restrict__, const uint32_t* __restrict__, const uint32_t* __restrict__, const uint32_t, const uint32_t, const PPP, uint32_t* __restrict__); \
    template void fm_refinement_gains_kernel<PPP>(const uint32_t* __restrict__, const dim_t* __restrict__, const float* __restrict__, const uint32_t* __restrict__, const PPP, const uint32_t* __restrict__, const uint32_t* __restrict__, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const bool, uint32_t* __restrict__, float* __restrict__); \
    template void fm_refinement_cascade_kernel<PPP>(const uint32_t* __restrict__, const dim_t* __restrict__, const uint32_t* __restrict__, const dim_t* __restrict__, const float* __restrict__, const uint32_t* __restrict__, const uint32_t* __restrict__, const uint32_t* __restrict__, const PPP, const uint32_t, const bool, float* __restrict__); \
    template void count_inbound_size_events_kernel<PPP>(const PPP, const uint32_t* __restrict__, const uint32_t* __restrict__, const int32_t* __restrict__, const dim_t, dim_t* __restrict__); \
    template void build_inbound_size_events_kernel<PPP>(const PPP, const uint32_t* __restrict__, const uint32_t* __restrict__, const uint32_t* __restrict__, const int32_t* __restrict__, const dim_t* __restrict__, const dim_t, uint32_t* __restrict__, uint32_t* __restrict__, int32_t* __restrict__);

INSTANTIATE_PPP_KERNELS(dense_ppp)
INSTANTIATE_PPP_KERNELS(sparse_ppp)
