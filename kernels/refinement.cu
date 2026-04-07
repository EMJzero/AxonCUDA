#include "refinement.cuh"
#include "constants.cuh"
#include "utils.cuh"

// for each hyperedge, count how many of its pins are in each partition
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void pins_per_partition_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ pins_per_partitions, // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
    uint32_t* __restrict__ partitions_inbound_sizes // partitions_inbound_sizes[part] -> number of inbound_pins_per_partitions for 'part' that are not zero => will be incorrect as of here (also including outbounds)
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /*
    * TODO upgrade:
    * - one hedge per warp -> go over the hedge in // in the warp
    * - shared memory histogram per hyperedge
    */

    /*
    * TODO alternative upgrade:
    * - one node per warp -> go over touching in // in the warp
    * - shared memory histogram per partition
    * => take the code from 'inbound_pins_per_partition_kernel' as of commit 'f803463'
    */

    const dim_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    uint32_t *my_pins_per_partitions = pins_per_partitions + static_cast<dim_t>(tid) * num_partitions;

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t part = partitions[*curr];
        const uint32_t prev = atomicAdd(&my_pins_per_partitions[part], 1);
        if (prev == 0) atomicAdd(&partitions_inbound_sizes[part], 1);
    }
}

// for each hyperedge, remove its sources from the pins per partition counts
// SEQUENTIAL COMPLEXITY: e
// PARALLEL OVER: e
__global__
void inbound_pins_per_partition_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ inbound_pins_per_partitions, // inbound_pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of inbound pins of that partition owned by this hedge
    uint32_t* __restrict__ partitions_inbound_sizes // partitions_inbound_sizes[part] -> number of inbound_pins_per_partitions for 'part' that are not zero
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    const dim_t hedge_start_idx = hedges_offsets[tid];
    const uint32_t *hedge_start = hedges + hedge_start_idx;
    const uint32_t hedge_srcs_count = srcs_count[tid];
    const uint32_t *hedge_srcs_end = hedge_start + hedge_srcs_count;
    uint32_t *my_inbound_pins_per_partitions = inbound_pins_per_partitions + static_cast<dim_t>(tid) * num_partitions;

    for (const uint32_t* curr = hedge_start; curr < hedge_srcs_end; curr++) {
        const uint32_t src_part = partitions[*curr];
        const uint32_t prev = atomicSub(&my_inbound_pins_per_partitions[src_part], 1);
        if (prev == 1) atomicSub(&partitions_inbound_sizes[src_part], 1);
    }
}

// find moves of nodes from one partition to another that yield a positive gain
// SEQUENTIAL COMPLEXITY: n*h*partitions
// PARALLEL OVER: n
// SHUFFLES OVER: partitions
__global__
void fm_refinement_gains_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t* __restrict__ pins_per_partitions, // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ partitions_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t randomizer,
    const uint32_t discount, // by how much to overshoot the size constraint when proposing moves
    const bool encourage_all_moves, // if true, even moves that don't fully disconnect an hyperedge receive a gain inversely proportional to how many pins remain
    // NOTE: we repurpose the arrays allocated for the "candidates kernel" for those!
    uint32_t* __restrict__ moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx
    float* __restrict__ scores // scores[idx] -> gain for move in position idx
) { 
    // STYLE: one node per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;
    
    /*
    * Idea:
    * - one node per warp
    * - different part of the histogram (different partitions - one bin per partition) in each thread's registers
    * - repeat enough times for finite-sized (PART_HIST_SIZE) histogram parts among threads to cover all partitions
    * - scan hyperedges, specifically their pins per hedge (with manual caching in shared memory?) enough times to have each partition once in the histogram
    * - maximum bin inside each thread, then warp primitives to find the maximum bin per node
    *
    * Upgrade: since in "pins_per_partitions", each hedge has an entry for each partition, and they are all always in the same order, we can just
    *          assign a thread in each warp to a few partitions, and always make it see those! No need for the whole histogram of all partitions
    *          per thread where each bin is then reduced! Just each thread in the warp is in charge of num_partitions/warp_size partitions!
    *
    * TODO: this kernel could undergo the same multi-candidate upgrade as "pairs"! Tho here it would be much harder with the moves sorting mechanism...
    * 
    * TODO: full conversion to shared memory histogram...
    *
    * TODO: like in HyperG, we could repurpose neighbors to keep a list of neighboring hedges to each node (maybe one-hot encoded), and thus
    *       not build the full histogram, but build it only for those neighboring partitions...
    *
    * NOTE: no need for FIXED POINT here, since we don't need symmetry nor other invariants!
    */
    
    const uint32_t my_partition = partitions[warp_id];
    const uint32_t my_size = nodes_sizes[warp_id];
    
    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
    //uint32_t touching_count = touching_offsets[warp_id + 1] - touching_offsets[warp_id];
    float loss[PART_HIST_SIZE]; // make sure this fits in registers (no spill) !! Store here only the score, the partition id can be inferred!
    float saving = 0.0f;
    
    // all threads in the warp should agree on those...
    float best_gain = -FLT_MAX;
    uint32_t best_move = UINT32_MAX;

    // handle the current partition first with its own scan of touching hyperedges
    for (const uint32_t* hedge_idx = my_touching + lane_id; hedge_idx < not_my_touching; hedge_idx += WARP_SIZE) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        const uint32_t* my_pin_per_partition = pins_per_partitions + static_cast<dim_t>(actual_hedge_idx) * num_partitions;
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        // hedge connected to my partition: gain the hedge's weight iff moving would disconnect it from my partition (I am its last pin left there)
        if (!encourage_all_moves && my_pin_per_partition[my_partition] == 1)
            saving += my_hedge_weight;
        // VARIANT: give a little push to nodes leaving a partition with not just one, but few pins left for an hedge
        if (encourage_all_moves && my_pin_per_partition[my_partition] >= 1)
            saving += my_hedge_weight / (my_pin_per_partition[my_partition] * my_pin_per_partition[my_partition]);
    }

    saving = warpReduceSum<float>(saving);
    
    // handle PART_HIST_SIZE*WARP_SIZE partitions at a time, that is partitions_per_thread per thread in the warp
    for (uint32_t curr_base_part = 0; curr_base_part < num_partitions; curr_base_part += PART_HIST_SIZE*WARP_SIZE) {
        // each thread handles, at once, min(PART_HIST_SIZE, partitions_per_thread) partitions, each partition is handled by exactly one thread per warp
        const uint32_t partitions_to_handle = min(num_partitions - curr_base_part, PART_HIST_SIZE*WARP_SIZE); // ... to handle over the whole warp
        const uint32_t partitions_per_thread = (partitions_to_handle + WARP_SIZE - 1) / WARP_SIZE; // ceiled
        const uint32_t threads_with_one_less_partition = partitions_per_thread*WARP_SIZE - partitions_to_handle;
        const uint32_t my_part_count = partitions_per_thread - (lane_id >= WARP_SIZE - threads_with_one_less_partition ? 1 : 0);

        // clear per-thread local histograms
        for (uint32_t p = 0; p < my_part_count; p++)
            loss[p] = 0.0f;

        // scan touching hyperedges
        // NOTE: interpret this as "for each hedge, see if you moving to a certain partition is something that they like or not"
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            const uint32_t actual_hedge_idx = *hedge_idx;
            const uint32_t* my_pin_per_partition = pins_per_partitions + static_cast<dim_t>(actual_hedge_idx) * num_partitions;
            const float my_hedge_weight = hedge_weights[actual_hedge_idx];
            // each thread in the warp reads partitions_per_thread counters
            for (uint32_t p = 0; p < my_part_count; p++) {
                const uint32_t part = curr_base_part + lane_id + p * WARP_SIZE;
                // hedge not yet connected to the partition: pay the hedge's weight iff moving there connects it to the new partition (I would become its first pin there)
                if (my_pin_per_partition[part] == 0) loss[p] += my_hedge_weight;
                // VARIANT: pt2
                //loss[p] += my_hedge_weight / ((my_pin_per_partition[part] + 1) * (my_pin_per_partition[part] + 1));
            }
        }

        // reduce max inside each threads
        for (uint32_t p = 0; p < my_part_count; p++) {
            const float gain = saving - loss[p];
            const uint32_t part = curr_base_part + lane_id + p * WARP_SIZE;
            //const float gain = (saving - loss[p]) * (partitions_sizes[part] + my_size <= max_nodes_per_part ? 1.0f : 0.8f);
            // MAYBE: could anticipate the constraint check! E.g. at the beginning of the iteration, entirely removing some partitions from the histogram,
            //        but this will require keeping a list of active partitions, since the indices won't suffice anymore...
            // => pseudo-random tie-break via hashes
            if (part != my_partition && partitions_sizes[part] + my_size - (my_size / discount) <= max_nodes_per_part && (gain > best_gain || gain == best_gain && hash_uint32(part + randomizer) > hash_uint32(best_move + randomizer))) {
            //if (part != my_partition && (gain > best_gain || gain == best_gain && hash_uint32(part) > hash_uint32(best_move))) {
                best_gain = gain;
                best_move = part;
            }
        }

        // reduce max between threads
        bin<float> max = warpReduceMax<float>(best_gain, best_move);
        best_gain = max.val;
        best_move = max.payload; // yeah, "node" should be called "partition" here, but this way we repurpose the struct...
    }

    if (lane_id == 0) {
        moves[warp_id] = best_move;
        scores[warp_id] = best_gain;
    }
}

// find the gain of each move under the HP that all higher-score moves have been applied
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
// SHUFFLES OVER: d
__global__
void fm_refinement_cascade_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* hedge_weights,
    // CHOOSE: either rank nodes by their score and pass "move_ranks" or pass "scores" and sort on the fly, with the node id as a tie-breaker
    // CHOICE: sorted scores and move_ranks, because we need to keep scores in their current (sorted) order even after updating them
    const uint32_t* __restrict__ move_ranks, // move_ranks[node_idx] -> i (ranking by score) of the move proposed by the idx node
    const uint32_t* __restrict__ moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx (DO NOT SORT)
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t* __restrict__ pins_per_partitions, // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const bool encourage_all_moves,
    float* __restrict__ scores // scores[move_ranks[node_idx]] -> gain for node idx's move
) {
    // STYLE: one node (move) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;

    /*
    * Idea (from HyperG):
    * - moves are sorted from the highest to lowest score
    * - greedy assumption: all moves with a higher score get applied
    * - therefore, for each move, recompute its score (gain) like in "fm_refinement_gains_kernel", but now assuming
    *   each node with a higher score changed partition to the one specified by the move
    * - write the new score in place of the previous one for each move, this then enables a scan to find the sequence of
        "moves as if applied in isolation" that yields the highest total gain when applied all together
    *
    * NOTE: we assume that the # of partitions is close to the avg. hedge cardinality, from which iterating over pins_per_partitions is just as efficient as it would be to iterate over hedges!
    *
    * NOTE: no need for FIXED POINT here, since we don't need neither symmetry nor invariants!
    *
    * NOTE: re-evaluate here EVERY move, even negative-gain ones, because after applying all previous moves, they may become positive-gained!
    *
    * MAYBE: do NOT read "my_move_part_counter" and "my_move_part_counter", but for every pin you see read its "partitions[pin]" and from it
    *        just recompute them while we are at it; this allow in-place updates to my_pin_per_partition without waiting for the global sync
    */
    
    float score = 0.0f;
    
    //const uint32_t my_move_score = scores[warp_id];
    const uint32_t my_move_part = moves[warp_id];
    // no need to update invalid moves
    if (my_move_part == UINT32_MAX) return;
    
    const uint32_t my_partition = partitions[warp_id];
    const uint32_t my_move_rank = move_ranks[warp_id];
    
    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
    //uint32_t touching_count = touching_offsets[warp_id + 1] - touching_offsets[warp_id];
    
    // scan touching hyperedges
    // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        // NOTE: this is not a warp-sync kernel, so using shuffles here to share data looses time, it's better to exploit caches with redundant reads!
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        const dim_t part_row_offset = static_cast<dim_t>(actual_hedge_idx) * num_partitions;
        const uint32_t my_curr_part_counter = pins_per_partitions[part_row_offset + my_partition];
        const uint32_t my_move_part_counter = pins_per_partitions[part_row_offset + my_move_part];
        int32_t my_curr_part_counter_delta = 0;
        int32_t my_move_part_counter_delta = 0;
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            uint32_t pin = *my_hedge;
            if (move_ranks[pin] < my_move_rank) { // speculation: better-ranked move -> applied
                // NOTE: invalid moves should all have a lower score and thus a higher rank than all others, never being see here
                uint32_t new_pin_partition = moves[pin];
                uint32_t prev_pin_partition = partitions[pin];
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
        // VVVV
        // NOTE: do not use the variant AT ALL when you work on k-way !!
        // ^^^^
        // gain the hedge's weight iff moving would disconnect the hedge from my partition (I am its last pin left there)
        const uint32_t true_curr_part_counter = my_curr_part_counter + warpReduceSumLN0<int32_t>(my_curr_part_counter_delta);
        if (!encourage_all_moves && true_curr_part_counter == 1)
            score += my_hedge_weight;
        // pay the hedge's weight iff moving there connects the hedge to the new partition (I would become its first pin there)
        if (my_move_part_counter + warpReduceSumLN0<int32_t>(my_move_part_counter_delta) == 0)
            score -= my_hedge_weight;
        // VARIANT: give a little push to nodes leaving a partition with not just one, but few pins left for an hedge
        if (encourage_all_moves && true_curr_part_counter >= 1)
            score += my_hedge_weight / (true_curr_part_counter * true_curr_part_counter);
        //const uint32_t true_move_part_counter = my_move_part_counter + warpReduceSumLN0<int32_t>(my_move_part_counter_delta);
        //score -= my_hedge_weight / ((true_move_part_counter + 1) * (true_move_part_counter + 1));
    }

    if (lane_id == 0)
        scores[my_move_rank] = score;
}

// apply moves with a positive gain
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
__global__
void fm_refinement_apply_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx (DO NOT SORT)
    const uint32_t* __restrict__ move_ranks, // move_ranks[node_idx] -> i (ranking by score) of the move proposed by the idx node
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t num_good_moves, // idx + 1 of the maximum in the updated scores
    uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    uint32_t* __restrict__ partitions_sizes
    //uint32_t* pins_per_partitions // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
) {
    // STYLE: one node (move) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    // stop at the last gain-increasing move
    // TODO: remove "moves[tid] == UINT32_MAX", it's redundant and here "just in case", invalid moves should always be outside of num_good_moves
    if (move_ranks[tid] >= num_good_moves || moves[tid] == UINT32_MAX) return;

    const uint32_t my_partition = partitions[tid];
    const uint32_t my_move_part = moves[tid];
    const uint32_t my_size = nodes_sizes[tid];

    // update partition sizes
    atomicSub(&partitions_sizes[my_partition], my_size);
    atomicAdd(&partitions_sizes[my_move_part], my_size);

    // update my partition
    partitions[tid] = my_move_part;

    /*const uint32_t* my_touching = touching + touching_offsets[tid];
    const uint32_t* not_my_touching = touching + touching_offsets[tid + 1];

    // scan touching hyperedges and update pins_per_partitions counts
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        atomicAdd(&pins_per_partitions[actual_hedge_idx * num_partitions + my_partition], -1);
        atomicAdd(&pins_per_partitions[actual_hedge_idx * num_partitions + my_move_part], 1);
    }*/
}

// transform moves into a sequence of size-altering events for capacity constraint checks
__global__
void build_size_events_kernel(
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ ranks,
    const uint32_t* __restrict__ partitions,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_nodes,
    uint32_t* __restrict__ ev_partition,
    uint32_t* __restrict__ ev_index,
    int32_t* __restrict__ ev_delta
) {
    // STYLE: one node (move) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    const int32_t size = static_cast<int32_t>(nodes_sizes[tid]);
    const uint32_t dst_part = moves[tid];
    const uint32_t rank = ranks[tid];
    
    // first event: node leaves its current partition
    const uint32_t e0 = 2 * tid;
    // second event: node enters its destination partition
    const uint32_t e1 = e0 + 1;
    
    // create no events for invalid moves
    if (dst_part == UINT32_MAX) {
        ev_partition[e0] = UINT32_MAX;
        ev_index[e0] = rank;
        ev_delta[e0] = 0;

        ev_partition[e1] = UINT32_MAX;
        ev_index[e1] = rank;
        ev_delta[e1] = 0;
        return;
    }

    ev_partition[e0] = partitions[tid];
    ev_index[e0] = rank;
    ev_delta[e0] = -size;

    ev_partition[e1] = dst_part;
    ev_index[e1] = rank;
    ev_delta[e1] = size;
}

// mark moves that are valid points in the sequence w.r.t. size constraints
__global__
void flag_size_events_kernel(
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const int32_t* __restrict__ ev_delta,
    const uint32_t* __restrict__ partitions_sizes,
    const uint32_t num_events,
    int32_t* __restrict__ valid_moves // initialized with 0s
) {
    // STYLE: one event per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    /*
    * Idea:
    * - for each move, compute how many partitions it brings to be invalid or it brings back to a valid state
    * - then compute the number of invalid partitions at each point in time as the prefix sum of the number going from ok to not-ok (+1) and not-ok to ok (-1)
    *
    * How:
    * - one thread per event
    * - all flags start at 0
    * - add to the flag how much you violated constraints by with this move, or subtract by how much you recovered
    * - if the initial state is valid, only having a final count of zero makes a move valid
    * - if the initial state is invalid, the more negative the count, the more we get close to a valid state
    */

    const uint32_t part = ev_partition[tid];
    const uint32_t rank = ev_index[tid];

    // dispose of invalid moves
    if (part == UINT32_MAX) {
        valid_moves[rank] = 1;
        return;
    }

    // TODO: those type casts are kinda dangerous...
    const int32_t base_size = static_cast<int32_t>(partitions_sizes[part]);
    const int32_t curr_size = base_size + ev_delta[tid];
    const int32_t max_size = static_cast<int32_t>(max_nodes_per_part);
    const int32_t new_excess = max(curr_size - max_size, 0); // by how much we now exceed the constraint

    const uint32_t pred_part = tid > 0 ? ev_partition[tid - 1] : UINT32_MAX; // partition acted upon by the event before this one
    const int32_t old_excess = max(pred_part != part ? base_size - max_size : base_size + ev_delta[tid - 1] - max_size, 0); // by how much we previously were exceeding the constraint

    atomicAdd(&valid_moves[rank], new_excess - old_excess); // accumulate how much we recovered from constraint violations with this move (valid -> valid = 0, invalid -> valid = <0, valid -> invalid = >0)
}

// for every move, generate two events for every inbound hedge, one removing it front the src, one adding it back
// SEQUENTIAL COMPLEXITY: n*h (h -> inbound only)
// PARALLEL OVER: n
__global__
void build_hedge_events_kernel(
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ ranks,
    const uint32_t* __restrict__ partitions,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t num_nodes,
    uint32_t* __restrict__ ev_partition, // init. to UINT32_MAX (to spot invalid ones later)
    uint32_t* __restrict__ ev_index,
    uint32_t* __restrict__ ev_hedge,
    int32_t* __restrict__ ev_delta
) {
    // STYLE: one node (move) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;
    
    /*
    * Idea:
    * - let each move write to an offset given by "touching_offsets", this will leave some blank events at the end of each // S;G
    *   move's, but we can easily filter those out later by identifying them from the UINT32_MAX ev_partition...
    */

    const uint32_t dst_part = moves[warp_id];
    if (dst_part == UINT32_MAX) return;
    const uint32_t src_part = partitions[warp_id];
    const uint32_t my_rank = ranks[warp_id];

    uint32_t *my_ev_partition = ev_partition + 2 * touching_offsets[warp_id] + 2 * lane_id;
    uint32_t *my_ev_index = ev_index + 2 * touching_offsets[warp_id] + 2 * lane_id;
    uint32_t *my_ev_hedge = ev_hedge + 2 * touching_offsets[warp_id] + 2 * lane_id;
    int32_t *my_ev_delta = ev_delta + 2 * touching_offsets[warp_id] + 2 * lane_id;

    const uint32_t* inbound = touching + touching_offsets[warp_id];
    const uint32_t my_inbound_count = inbound_count[warp_id];
    for (uint32_t i = lane_id; i < my_inbound_count; i += WARP_SIZE) {
        uint32_t hedge = inbound[i];
        // first event: hedge does not touches one less time the node's current partition
        my_ev_partition[0] = src_part;
        my_ev_index[0] = my_rank;
        my_ev_hedge[0] = hedge;
        my_ev_delta[0] = -1;
        // second event: hedge touches one more time the node's destination partition
        my_ev_partition[1] = dst_part;
        my_ev_index[1] = my_rank;
        my_ev_hedge[1] = hedge;
        my_ev_delta[1] = +1;
        
        my_ev_partition += 2 * WARP_SIZE;
        my_ev_index += 2 * WARP_SIZE;
        my_ev_hedge += 2 * WARP_SIZE;
        my_ev_delta += 2 * WARP_SIZE;
    }
}

// for every inbound hedge event that adds/removes an inbound hedge to a partition, count a new inbound size event
__global__
void count_inbound_size_events_kernel(
    const uint32_t* __restrict__ partitions_inbound_counts, // this is pins_per_partition, index it as [hedge_idx*num_partitions + partition_idx]
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const uint32_t* __restrict__ ev_hedge,
    const int32_t* __restrict__ ev_delta,
    const uint32_t num_events,
    const uint32_t num_partitions,
    dim_t* inbound_size_events_offsets // init. to zero
) {
    // STYLE: one event per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    const uint32_t part = ev_partition[tid];

    // dispose of invalid events
    if (part == UINT32_MAX) return;

    const uint32_t hedge = ev_hedge[tid];
    const uint32_t init_hedge_inbound_count = partitions_inbound_counts[static_cast<dim_t>(hedge) * num_partitions + part];

    uint32_t prev_hedge_inbound_count = init_hedge_inbound_count;
    if (tid > 0 && ev_partition[tid - 1] == part && ev_hedge[tid - 1] == hedge) // if the previous sum was about the same hedge as mine, consider its updated count in the sequence
        prev_hedge_inbound_count += ev_delta[tid - 1];
    uint32_t curr_hedge_inbound_count = init_hedge_inbound_count + ev_delta[tid];

    if (prev_hedge_inbound_count == 0 && curr_hedge_inbound_count > 0 || prev_hedge_inbound_count > 0 && curr_hedge_inbound_count == 0)
        inbound_size_events_offsets[tid + 1] = 1; // +1 to do an inclusive scan and keep the final count
}

// for every inbound hedge event that adds/removes an inbound hedge to a partition, create a new inbound size event
__global__
void build_inbound_size_events_kernel(
    const uint32_t* __restrict__ partitions_inbound_counts, // this is pins_per_partition, index it as [hedge_idx*num_partitions + partition_idx]
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const uint32_t* __restrict__ ev_hedge,
    const int32_t* __restrict__ ev_delta,
    const dim_t* inbound_size_events_offsets,
    const uint32_t num_events,
    const uint32_t num_partitions,
    uint32_t* __restrict__ new_ev_partition,
    uint32_t* __restrict__ new_ev_index,
    int32_t* __restrict__ new_ev_delta
) {
    // STYLE: one event per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    // TODO: could avoid repeating the check (and all the associated memory accesses) by storing a +1/-1/0 flag between this kernel and the previous counting one

    const uint32_t part = ev_partition[tid];

    // dispose of invalid events
    if (part == UINT32_MAX) return;

    const uint32_t hedge = ev_hedge[tid];
    const uint32_t init_hedge_inbound_count = partitions_inbound_counts[static_cast<dim_t>(hedge) * num_partitions + part];

    uint32_t prev_hedge_inbound_count = init_hedge_inbound_count;
    if (tid > 0 && ev_partition[tid - 1] == part && ev_hedge[tid - 1] == hedge) // if the previous sum was about the same hedge as mine, consider its updated count in the sequence
        prev_hedge_inbound_count += ev_delta[tid - 1];
    uint32_t curr_hedge_inbound_count = init_hedge_inbound_count + ev_delta[tid];

    if (prev_hedge_inbound_count == 0 && curr_hedge_inbound_count > 0) {
        const uint32_t new_ev_offset = inbound_size_events_offsets[tid];
        new_ev_partition[new_ev_offset] = part;
        new_ev_index[new_ev_offset] = ev_index[tid];
        new_ev_delta[new_ev_offset] = 1;
    } else if (prev_hedge_inbound_count > 0 && curr_hedge_inbound_count == 0) {
        const uint32_t new_ev_offset = inbound_size_events_offsets[tid];
        new_ev_partition[new_ev_offset] = part;
        new_ev_index[new_ev_offset] = ev_index[tid];
        new_ev_delta[new_ev_offset] = -1;
    }
}

// mark moves that are valid points in the sequence w.r.t. inbound constraints
__global__
void flag_inbound_events_kernel(
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const int32_t* __restrict__ ev_delta,
    const uint32_t* __restrict__ partitions_inbound_sizes, // partitions_inbound_sizes[part] = size of the inbound set for part
    const dim_t num_events,
    int32_t* __restrict__ valid_moves // initialized with 0s
) {
    // STYLE: one event per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    /*
    * Idea:
    * - for each move, compute how many partitions it brings to be invalid or it brings back to a valid state
    * - then compute the number of invalid partitions at each point in time as the prefix sum of the number going from ok to not-ok (+1) and not-ok to ok (-1)
    * 
    * HP: always start from a VALID state
    * 
    * How: very much like 'flag_size_events_kernel'
    */

    const uint32_t part = ev_partition[tid];
    const uint32_t rank = ev_index[tid];

    // dispose of invalid moves
    if (part == UINT32_MAX) {
        valid_moves[rank] = 1;
        return;
    }

    const int32_t base_size = static_cast<int32_t>(partitions_inbound_sizes[part]);
    const int32_t curr_size = base_size + ev_delta[tid];
    const int32_t max_size = static_cast<int32_t>(max_inbound_per_part);
    const bool is_valid = curr_size <= max_size; // true iff after this event the partition's inbound set size is valid

    const uint32_t pred_part = tid > 0 ? ev_partition[tid - 1] : UINT32_MAX; // partition acted upon by the event before this one
    const bool was_valid = pred_part != part || base_size + ev_delta[tid - 1] <= max_size; // true iff before this event the partition's inbound set size is valid

    if (was_valid && !is_valid) // this event made the partition invalid -> track a +1 in invalid partitions as of this event
        atomicAdd(&valid_moves[rank], 1);
    if (!was_valid && is_valid) // this event made the partition invalid -> track a -1 in invalid partitions as of this event
        atomicSub(&valid_moves[rank], 1);
}


// SPARSE PINS-PER-PARTITION VARIANT

// retrieval:
// - if your flag bit (the one part-idx from the least-significant one) is zero, return zero
// - otherwise, count how many ones are in less-significant positions than part-idx, add that to the base count, and that's your offset to go fetch
// TODO: could <maybe> cache a few entries of ppp_offsets per warp/block?
__device__ __forceinline__ uint32_t get_ppp(const bitmap* ppp_offsets, const uint32_t* ppp, const uint32_t ppp_per_hedge, const uint32_t hedge_idx, const uint32_t part_idx) {
    const bitmap *my_ppp_offsets = ppp_offsets + static_cast<dim_t>(hedge_idx) * ppp_per_hedge;
    const dim_t ppp_bitmap_idx = part_idx >> BITMAP_CAPLOG; // aka: part / BITMAP_CAPACITY
    bitmap ppp_bitmap = my_ppp_offsets[ppp_bitmap_idx];
    const uint64_t bitmap_part_idx = part_idx & (BITMAP_CAPACITY - 1u); // aka: part % BITMAP_CAPACITY
    const uint64_t ppp_bitmask = 1ull << bitmap_part_idx; // put a 1 in position part % BITMAP_CAPACITY, all other bits are zero
    if ((ppp_bitmap.flg & ppp_bitmask) == 0ull) return 0u; // no pins
    //const uint64_t ppp_left_side_bitmask = ~(UINT64_MAX >> (part_idx & (BITMAP_CAPACITY - 1u))); // put all 1s left of position part % BITMAP_CAPACITY, followed by zeros
    //return ppp[ppp_bitmap.cnt + __popcll(ppp_bitmap.flg & ppp_left_side_bitmask)];
    //return ppp[ppp_bitmap.cnt + __popcll(ppp_bitmap.flg >> (BITMAP_CAPACITY - bitmap_part_idx))];
    return ppp[ppp_bitmap.cnt + (bitmap_part_idx == 0 ? 0u : __popcll(ppp_bitmap.flg << (BITMAP_CAPACITY - bitmap_part_idx)))];
}

// for each hyperedge, count how many of its pins are in each partition
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// SHUFFLES OVER: d
__global__
void sparse_pins_per_partition_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t num_hedges,
    const uint32_t ppp_per_hedge, // ppp_per_hedge = ceil(num_partitions / 64) [note: 64 = BITMAP_CAPACITY]
    bitmap* __restrict__ ppp_offsets // ppp_offsets[hedge-idx * ppp_per_hedge + part-idx / 64] -> bitmap to access the pin count for all (hedge, part / 64), ... (hedge, part / 64 + 63) pairs
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the sample to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    const dim_t hedge_start_idx = hedges_offsets[warp_id], hedge_end_idx = hedges_offsets[warp_id + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    bitmap *my_ppp_offsets = ppp_offsets + static_cast<dim_t>(warp_id) * ppp_per_hedge;

    for (const uint32_t* curr = hedge_start + lane_id; curr < hedge_end; curr += WARP_SIZE) {
        const uint32_t part = partitions[*curr];
        // now all thread in the warp set their "bitmap.flg" of my_ppp_offsets to 1 in parallel
        // => since multiple threads might want to write the same 64bit word, make them agree on a single update per word
        const dim_t ppp_bitmap_idx = part >> BITMAP_CAPLOG; // aka: part / BITMAP_CAPACITY
        const uint64_t ppp_bitmask = 1ull << (part & (BITMAP_CAPACITY - 1u)); // put a 1 in position part % BITMAP_CAPACITY, all other bits are zero
        const unsigned active = __activemask();
        // lanes targeting the same 64+64-bit word cooperate
        const unsigned peers = __match_any_sync(active, ppp_bitmap_idx);
        // OR-reduce all bit requests for the same word across peers
        const uint64_t combined_mask = __reduce_or_sync(peers, ppp_bitmask);
        const int leader = __ffs(peers) - 1;
        if (static_cast<int>(lane_id) == leader) {
            // exactly one writer per bitmap per iteration
            bitmap* ppp_offset_ptr = my_ppp_offsets + ppp_bitmap_idx;
            const bitmap old_bitmap = *ppp_offset_ptr;
            bitmap updated_bitmap;
            uint64_t new_bits = combined_mask & ~old_bitmap.flg;
            updated_bitmap.cnt = old_bitmap.cnt + __popcll(new_bits);
            updated_bitmap.flg = old_bitmap.flg | combined_mask;
            *ppp_offset_ptr = updated_bitmap;
            // add the number of added 1s to the incident hedges count
        }
    }
}

// given the sparse pins-per-partiton bitmaps, write the actual ppp counters in the segmented array
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// SHUFFLES OVER: d
__global__
void sparse_pins_per_partition_write_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const bitmap* __restrict__ ppp_offsets, // ppp_offsets[hedge-idx * ppp_per_hedge + part-idx / 64] -> bitmap to access the pin count for all (hedge, part / 64), ... (hedge, part / 64 + 63) pairs
    const uint32_t num_hedges,
    const uint32_t ppp_per_hedge, // ppp_per_hedge = ceil(num_partitions / 64) [note: 64 = BITMAP_CAPACITY]
    uint32_t* __restrict__ ppp, // ppp[ppp_offsets[e*num_partitions+p/64].cnt + bits-at-one-before-the(p%64)th-in(ppp_offsets[e*num_partitions+p/64].flg)] -> pins count held by hedge e in partition p
    uint32_t* __restrict__ partitions_incident_sizes // partitions_incident_sizes[part] -> number of pins-per-partitions for 'part' that are not zero
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the sample to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    const dim_t hedge_start_idx = hedges_offsets[warp_id], hedge_end_idx = hedges_offsets[warp_id + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    const bitmap *my_ppp_offsets = ppp_offsets + static_cast<dim_t>(warp_id) * ppp_per_hedge;

    for (const uint32_t* curr = hedge_start + lane_id; curr < hedge_end; curr += WARP_SIZE) {
        const uint32_t part = partitions[*curr];
        const dim_t ppp_bitmap_idx = part >> BITMAP_CAPLOG; // aka: part / BITMAP_CAPACITY
        const uint64_t bitmap_part_idx = part & (BITMAP_CAPACITY - 1u); // aka: part % BITMAP_CAPACITY
        bitmap ppp_bitmap = my_ppp_offsets[ppp_bitmap_idx];
        const uint32_t prev = atomicAdd(&ppp[ppp_bitmap.cnt + (bitmap_part_idx == 0 ? 0u : __popcll(ppp_bitmap.flg << (BITMAP_CAPACITY - bitmap_part_idx)))], 1u);
        if (prev == 0) atomicAdd(&partitions_incident_sizes[part], 1);
    }
}

// subtract outbound pins from ppp's counters
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// SHUFFLES OVER: d
__global__
void sparse_inbound_pins_per_partition_update_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ partitions,
    const bitmap* __restrict__ ppp_offsets,
    const uint32_t num_hedges,
    const uint32_t ppp_per_hedge,
    uint32_t* __restrict__ in_ppp, // in_ppp[ppp_offsets[... same as ppp] -> inbound pins count held by hedge e in partition p
    uint32_t* __restrict__ partitions_inbound_sizes // partitions_inbound_sizes[part] -> number of inbound-pins-per-partitions for 'part' that are not zero
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the sample to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    const dim_t hedge_src_start_idx = hedges_offsets[warp_id];
    const dim_t hedge_src_end_idx = hedge_src_start_idx + srcs_count[warp_id];
    const uint32_t *hedge_src_start = hedges + hedge_src_start_idx, *hedge_src_end = hedges + hedge_src_end_idx;
    const bitmap *my_ppp_offsets = ppp_offsets + static_cast<dim_t>(warp_id) * ppp_per_hedge;

    for (const uint32_t* curr = hedge_src_start + lane_id; curr < hedge_src_end; curr += WARP_SIZE) {
        const uint32_t part = partitions[*curr];
        const dim_t ppp_bitmap_idx = part >> BITMAP_CAPLOG; // aka: part / BITMAP_CAPACITY
        const uint64_t bitmap_part_idx = part & (BITMAP_CAPACITY - 1u); // aka: part % BITMAP_CAPACITY
        bitmap ppp_bitmap = my_ppp_offsets[ppp_bitmap_idx];
        const uint32_t prev = atomicSub(&in_ppp[ppp_bitmap.cnt + (bitmap_part_idx == 0 ? 0u : __popcll(ppp_bitmap.flg << (BITMAP_CAPACITY - bitmap_part_idx)))], 1u);
        if (prev == 1) atomicSub(&partitions_inbound_sizes[part], 1);
    }
}

// see "fm_refinement_gains_kernel"
__global__
void fm_refinement_gains_sparse_ppp_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const bitmap* __restrict__ ppp_offsets, // ppp_offsets[hedge-idx * ppp_per_hedge + part-idx / 64] -> bitmap to access the pin count for all (hedge, part / 64), ... (hedge, part / 64 + 63) pairs
    const uint32_t* __restrict__ ppp, // ppp[ppp_offsets[e*num_partitions+p/64].cnt + bits-at-one-before-the(p%64)th-in(ppp_offsets[e*num_partitions+p/64].flg)] -> pins count held by hedge e in partition p
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ partitions_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t ppp_per_hedge,
    const uint32_t randomizer,
    const uint32_t discount, // by how much to overshoot the size constraint when proposing moves
    const bool encourage_all_moves, // if true, even moves that don't fully disconnect an hyperedge receive a gain inversely proportional to how many pins remain
    // NOTE: we repurpose the arrays allocated for the "candidates kernel" for those!
    uint32_t* __restrict__ moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx
    float* __restrict__ scores // scores[idx] -> gain for move in position idx
) { 
    // STYLE: one node per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;
    
    const uint32_t my_partition = partitions[warp_id];
    const uint32_t my_size = nodes_sizes[warp_id];
    
    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
    //uint32_t touching_count = touching_offsets[warp_id + 1] - touching_offsets[warp_id];
    float loss[PART_HIST_SIZE]; // make sure this fits in registers (no spill) !! Store here only the score, the partition id can be inferred!
    float saving = 0.0f;
    
    // all threads in the warp should agree on those...
    float best_gain = -FLT_MAX;
    uint32_t best_move = UINT32_MAX;

    // handle the current partition first with its own scan of touching hyperedges
    for (const uint32_t* hedge_idx = my_touching + lane_id; hedge_idx < not_my_touching; hedge_idx += WARP_SIZE) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        const uint32_t my_pin_per_partition = get_ppp(ppp_offsets, ppp, ppp_per_hedge, actual_hedge_idx, my_partition);
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        // hedge connected to my partition: gain the hedge's weight iff moving would disconnect it from my partition (I am its last pin left there)
        if (!encourage_all_moves && my_pin_per_partition == 1)
            saving += my_hedge_weight;
        // VARIANT: give a little push to nodes leaving a partition with not just one, but few pins left for an hedge
        if (encourage_all_moves && my_pin_per_partition >= 1)
            saving += my_hedge_weight / (my_pin_per_partition * my_pin_per_partition);
    }

    saving = warpReduceSum<float>(saving);
    
    // handle PART_HIST_SIZE*WARP_SIZE partitions at a time, that is partitions_per_thread per thread in the warp
    for (uint32_t curr_base_part = 0; curr_base_part < num_partitions; curr_base_part += PART_HIST_SIZE*WARP_SIZE) {
        // each thread handles, at once, min(PART_HIST_SIZE, partitions_per_thread) partitions, each partition is handled by exactly one thread per warp
        const uint32_t partitions_to_handle = min(num_partitions - curr_base_part, PART_HIST_SIZE*WARP_SIZE); // ... to handle over the whole warp
        const uint32_t partitions_per_thread = (partitions_to_handle + WARP_SIZE - 1) / WARP_SIZE; // ceiled
        const uint32_t threads_with_one_less_partition = partitions_per_thread*WARP_SIZE - partitions_to_handle;
        const uint32_t my_part_count = partitions_per_thread - (lane_id >= WARP_SIZE - threads_with_one_less_partition ? 1 : 0);

        // clear per-thread local histograms
        for (uint32_t p = 0; p < my_part_count; p++)
            loss[p] = 0.0f;

        // scan touching hyperedges
        // NOTE: interpret this as "for each hedge, see if you moving to a certain partition is something that they like or not"
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            const uint32_t actual_hedge_idx = *hedge_idx;
            const float my_hedge_weight = hedge_weights[actual_hedge_idx];
            const bitmap *my_ppp_offsets = ppp_offsets + static_cast<dim_t>(actual_hedge_idx) * ppp_per_hedge;
            // each thread in the warp reads partitions_per_thread counters
            for (uint32_t p = 0; p < my_part_count; p++) {
                const uint32_t part = curr_base_part + lane_id + p * WARP_SIZE;
                const dim_t ppp_bitmap_idx = part >> BITMAP_CAPLOG; // aka: part / BITMAP_CAPACITY
                bitmap ppp_bitmap = my_ppp_offsets[ppp_bitmap_idx];
                uint32_t my_pin_per_partition = 0u;
                const uint64_t ppp_bitmask = 1ull << (part & (BITMAP_CAPACITY - 1u)); // put a 1 in position part % BITMAP_CAPACITY, all other bits are zero
                if ((ppp_bitmap.flg & ppp_bitmask) != 0ull) { // >0 pins
                    const uint64_t bitmap_part_idx = part & (BITMAP_CAPACITY - 1u); // aka: part % BITMAP_CAPACITY
                    my_pin_per_partition = ppp[ppp_bitmap.cnt + (bitmap_part_idx == 0 ? 0u : __popcll(ppp_bitmap.flg << (BITMAP_CAPACITY - bitmap_part_idx)))];
                }
                // hedge not yet connected to the partition: pay the hedge's weight iff moving there connects it to the new partition (I would become its first pin there)
                if (my_pin_per_partition == 0) loss[p] += my_hedge_weight;
            }
        }

        // reduce max inside each threads
        for (uint32_t p = 0; p < my_part_count; p++) {
            const float gain = saving - loss[p];
            const uint32_t part = curr_base_part + lane_id + p * WARP_SIZE;
            // => pseudo-random tie-break via hashes
            if (part != my_partition && partitions_sizes[part] + my_size - (my_size / discount) <= max_nodes_per_part && (gain > best_gain || gain == best_gain && hash_uint32(part + randomizer) > hash_uint32(best_move + randomizer))) {
                best_gain = gain;
                best_move = part;
            }
        }

        // reduce max between threads
        bin<float> max = warpReduceMax<float>(best_gain, best_move);
        best_gain = max.val;
        best_move = max.payload; // yeah, "node" should be called "partition" here, but this way we repurpose the struct...
    }

    if (lane_id == 0) {
        moves[warp_id] = best_move;
        scores[warp_id] = best_gain;
    }
}

// see "fm_refinement_cascade_kernel"
__global__
void fm_refinement_cascade_sparse_ppp_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t *inbound_count,
    const float* hedge_weights,
    const uint32_t* __restrict__ move_ranks, // move_ranks[node_idx] -> i (ranking by score) of the move proposed by the idx node
    const uint32_t* __restrict__ moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx (DO NOT SORT)
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const bitmap* __restrict__ ppp_offsets, // ppp_offsets[hedge-idx * ppp_per_hedge + part-idx / 64] -> bitmap to access the pin count for all (hedge, part / 64), ... (hedge, part / 64 + 63) pairs
    const uint32_t* __restrict__ ppp, // ppp[ppp_offsets[e*num_partitions+p/64].cnt + bits-at-one-before-the(p%64)th-in(ppp_offsets[e*num_partitions+p/64].flg)] -> pins count held by hedge e in partition p
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t ppp_per_hedge,
    const bool encourage_all_moves,
    float* __restrict__ scores, // scores[move_ranks[node_idx]] -> gain for node idx's move
    dim_t* __restrict__ size_events_offsets, // size_events_offsets[node_idx] -> set to "1" if the node has a valid move
    dim_t* __restrict__ inbound_events_offsets // inbound_events_offsets[node idx] -> set to the node's "inbound set size", if it has a valid move
) {
    // STYLE: one node (move) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;
    
    float score = 0.0f;
    
    const uint32_t my_move_part = moves[warp_id];
    // no need to update invalid moves
    if (my_move_part == UINT32_MAX) return;
    
    const uint32_t my_partition = partitions[warp_id];
    const uint32_t my_move_rank = move_ranks[warp_id];
    
    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
    
    // scan touching hyperedges
    // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        // NOTE: this is not a warp-sync kernel, so using shuffles here to share data looses time, it's better to exploit caches with redundant reads!
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        const uint32_t my_curr_part_counter = get_ppp(ppp_offsets, ppp, ppp_per_hedge, actual_hedge_idx, my_partition);
        const uint32_t my_move_part_counter = get_ppp(ppp_offsets, ppp, ppp_per_hedge, actual_hedge_idx, my_move_part);
        int32_t my_curr_part_counter_delta = 0;
        int32_t my_move_part_counter_delta = 0;
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            uint32_t pin = *my_hedge;
            if (move_ranks[pin] < my_move_rank) { // speculation: better-ranked move -> applied
                // NOTE: invalid moves should all have a lower score and thus a higher rank than all others, never being see here
                uint32_t new_pin_partition = moves[pin];
                uint32_t prev_pin_partition = partitions[pin];
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
        const uint32_t true_curr_part_counter = my_curr_part_counter + warpReduceSumLN0<int32_t>(my_curr_part_counter_delta);
        if (!encourage_all_moves && true_curr_part_counter == 1)
            score += my_hedge_weight;
        // pay the hedge's weight iff moving there connects the hedge to the new partition (I would become its first pin there)
        if (my_move_part_counter + warpReduceSumLN0<int32_t>(my_move_part_counter_delta) == 0)
            score -= my_hedge_weight;
        // VARIANT: give a little push to nodes leaving a partition with not just one, but few pins left for an hedge
        if (encourage_all_moves && true_curr_part_counter >= 1)
            score += my_hedge_weight / (true_curr_part_counter * true_curr_part_counter);
    }

    if (lane_id == 0) {
        scores[my_move_rank] = score;
        size_events_offsets[warp_id] = 1;
        inbound_events_offsets[warp_id] = inbound_count[warp_id];
    }
}

// see "build_size_events_kernel"
__global__
void build_size_events_sparse_kernel(
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ ranks,
    const uint32_t* __restrict__ partitions,
    const uint32_t* __restrict__ nodes_sizes,
    const dim_t* __restrict__ size_ev_offsets,
    const uint32_t num_nodes,
    uint32_t* __restrict__ ev_partition,
    uint32_t* __restrict__ ev_index,
    int32_t* __restrict__ ev_delta
) {
    // STYLE: one node (move) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    const uint32_t dst_part = moves[tid];
    // create no events for invalid moves
    if (dst_part == UINT32_MAX) return;
    const int32_t size = static_cast<int32_t>(nodes_sizes[tid]);
    const uint32_t rank = ranks[tid];

    // first event: node leaves its current partition
    const uint32_t e0 = 2 * size_ev_offsets[tid];
    // second event: node enters its destination partition
    const uint32_t e1 = e0 + 1;

    ev_partition[e0] = partitions[tid];
    ev_index[e0] = rank;
    ev_delta[e0] = -size;

    ev_partition[e1] = dst_part;
    ev_index[e1] = rank;
    ev_delta[e1] = size;
}

// see "build_hedge_events_kernel"
__global__
void build_hedge_events_sparse_kernel(
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ ranks,
    const uint32_t* __restrict__ partitions,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const dim_t* __restrict__ inbound_ev_offsets,
    const uint32_t num_nodes,
    uint32_t* __restrict__ ev_partition, // init. to UINT32_MAX (to spot invalid ones later)
    uint32_t* __restrict__ ev_index,
    uint32_t* __restrict__ ev_hedge,
    int32_t* __restrict__ ev_delta
) {
    // STYLE: one node (move) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;

    const uint32_t dst_part = moves[warp_id];
    if (dst_part == UINT32_MAX) return;
    const uint32_t src_part = partitions[warp_id];
    const uint32_t my_rank = ranks[warp_id];

    const dim_t ev_offset = inbound_ev_offsets[warp_id];
    uint32_t *my_ev_partition = ev_partition + 2 * ev_offset + 2 * lane_id;
    uint32_t *my_ev_index = ev_index + 2 * ev_offset + 2 * lane_id;
    uint32_t *my_ev_hedge = ev_hedge + 2 * ev_offset + 2 * lane_id;
    int32_t *my_ev_delta = ev_delta + 2 * ev_offset + 2 * lane_id;

    const uint32_t* inbound = touching + touching_offsets[warp_id];
    const uint32_t my_inbound_count = inbound_count[warp_id];
    for (uint32_t i = lane_id; i < my_inbound_count; i += WARP_SIZE) {
        uint32_t hedge = inbound[i];
        // first event: hedge does not touches one less time the node's current partition
        my_ev_partition[0] = src_part;
        my_ev_index[0] = my_rank;
        my_ev_hedge[0] = hedge;
        my_ev_delta[0] = -1;
        // second event: hedge touches one more time the node's destination partition
        my_ev_partition[1] = dst_part;
        my_ev_index[1] = my_rank;
        my_ev_hedge[1] = hedge;
        my_ev_delta[1] = +1;
        
        my_ev_partition += 2 * WARP_SIZE;
        my_ev_index += 2 * WARP_SIZE;
        my_ev_hedge += 2 * WARP_SIZE;
        my_ev_delta += 2 * WARP_SIZE;
    }
}

// see "count_inbound_size_events_kernel"
__global__
void count_inbound_size_events_sparse_ppp_kernel(
    const bitmap* __restrict__ ppp_offsets,
    const uint32_t* __restrict__ in_ppp,
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const uint32_t* __restrict__ ev_hedge,
    const int32_t* __restrict__ ev_delta,
    const uint32_t num_events,
    const uint32_t num_partitions,
    const uint32_t ppp_per_hedge,
    dim_t* inbound_size_events_offsets // init. to zero
) {
    // STYLE: one event per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    const uint32_t part = ev_partition[tid];

    // dispose of invalid events
    if (part == UINT32_MAX) return;

    const uint32_t hedge = ev_hedge[tid];
    const uint32_t init_hedge_inbound_count = get_ppp(ppp_offsets, in_ppp, ppp_per_hedge, hedge, part);

    uint32_t prev_hedge_inbound_count = init_hedge_inbound_count;
    if (tid > 0 && ev_partition[tid - 1] == part && ev_hedge[tid - 1] == hedge) // if the previous sum was about the same hedge as mine, consider its updated count in the sequence
        prev_hedge_inbound_count += ev_delta[tid - 1];
    uint32_t curr_hedge_inbound_count = init_hedge_inbound_count + ev_delta[tid];

    if (prev_hedge_inbound_count == 0 && curr_hedge_inbound_count > 0 || prev_hedge_inbound_count > 0 && curr_hedge_inbound_count == 0)
        inbound_size_events_offsets[tid + 1] = 1; // +1 to do an inclusive scan and keep the final count
}

// see "build_inbound_size_events_kernel"
__global__
void build_inbound_size_events_sparse_ppp_kernel(
    const bitmap* __restrict__ ppp_offsets,
    const uint32_t* __restrict__ in_ppp,
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const uint32_t* __restrict__ ev_hedge,
    const int32_t* __restrict__ ev_delta,
    const dim_t* inbound_size_events_offsets,
    const uint32_t num_events,
    const uint32_t num_partitions,
    const uint32_t ppp_per_hedge,
    uint32_t* __restrict__ new_ev_partition,
    uint32_t* __restrict__ new_ev_index,
    int32_t* __restrict__ new_ev_delta
) {
    // STYLE: one event per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    const uint32_t part = ev_partition[tid];

    // dispose of invalid events
    if (part == UINT32_MAX) return;

    const uint32_t hedge = ev_hedge[tid];
    const uint32_t init_hedge_inbound_count = get_ppp(ppp_offsets, in_ppp, ppp_per_hedge, hedge, part);

    uint32_t prev_hedge_inbound_count = init_hedge_inbound_count;
    if (tid > 0 && ev_partition[tid - 1] == part && ev_hedge[tid - 1] == hedge) // if the previous sum was about the same hedge as mine, consider its updated count in the sequence
        prev_hedge_inbound_count += ev_delta[tid - 1];
    uint32_t curr_hedge_inbound_count = init_hedge_inbound_count + ev_delta[tid];

    if (prev_hedge_inbound_count == 0 && curr_hedge_inbound_count > 0) {
        const uint32_t new_ev_offset = inbound_size_events_offsets[tid];
        new_ev_partition[new_ev_offset] = part;
        new_ev_index[new_ev_offset] = ev_index[tid];
        new_ev_delta[new_ev_offset] = 1;
    } else if (prev_hedge_inbound_count > 0 && curr_hedge_inbound_count == 0) {
        const uint32_t new_ev_offset = inbound_size_events_offsets[tid];
        new_ev_partition[new_ev_offset] = part;
        new_ev_index[new_ev_offset] = ev_index[tid];
        new_ev_delta[new_ev_offset] = -1;
    }
}

// straight up compute inbound set sizes from hedges
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// SHUFFLES OVER: d
__global__
void inbound_sets_size_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ pp_map,
    uint32_t* __restrict__ partitions_inbound_sizes
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the sample to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    const dim_t hedge_start_idx = hedges_offsets[warp_id] + srcs_count[warp_id], hedge_end_idx = hedges_offsets[warp_id + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    uint32_t *my_pp_map = pp_map + static_cast<dim_t>(warp_id) * ((num_partitions + 31u) / 32u);

    for (const uint32_t* curr = hedge_start + lane_id; curr < hedge_end; curr += WARP_SIZE) {
        const uint32_t part = partitions[*curr];
        // => since multiple threads might want to write the same 32bit word, make them agree on a single update per word
        const dim_t pp_map_idx = part >> 5u; // aka: part / 2**5
        const uint32_t pp_map_bit = 1u << (part & 31u); // aka: put a 1 in position part % 32
        const unsigned active = __activemask();
        // lanes targeting the same 64+64-bit word cooperate
        const unsigned peers = __match_any_sync(active, pp_map_idx);
        // OR-reduce all bit requests for the same word across peers
        const uint64_t combined_mask = __reduce_or_sync(peers, pp_map_bit);
        const int leader = __ffs(peers) - 1;
        uint32_t prev;
        if (static_cast<int>(lane_id) == leader) {
            // exactly one writer per map per iteration
            uint32_t* pp_ptr = my_pp_map + pp_map_idx;
            const uint32_t old_map = *pp_ptr;
            const uint32_t updated_bitmap = old_map | combined_mask;
            *pp_ptr = updated_bitmap;
            prev = __shfl_sync(peers, old_map, leader);
        } else {
            prev = __shfl_sync(peers, 0u, leader);
        }
        if ((prev & pp_map_bit) == 0) atomicAdd(&partitions_inbound_sizes[part], 1u);
    }
}