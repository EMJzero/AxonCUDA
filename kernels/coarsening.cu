#include "coarsening.cuh"
#include "constants.cuh"
#include "utils.cuh"

// find the best neighbor for each node to stay with (edge-coarsening)
// SEQUENTIAL COMPLEXITY: n*h*d + n*(# neighbors)*h
// PARALLEL OVER: n
// SHUFFLES OVER: h*d (neighbors)
__global__
void candidates_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    uint32_t* __restrict__ pairs,
    uint32_t* __restrict__ scores
) {
    // STYLE: one node per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;

    /*
    * Idea:
    * - one node per warp
    * - histogram (one bin per neighbor) in each shared memory
    * - iterate hyperedges (with caching in shared memory - either passive or automatic) once per histogram part
    * - warp primitives to both broadcast reads among threads and find the maximum bin
    *
    * It is 100% possible for two neighbors of a node to partake, with that node, in the same hedges! And this makes them peer pairing candidates.
    * => Use fixed point, not floats, because we need associativity to find those peer candidates!
    * => In case of a tie, deterministically update the best by lower ID! This is the invariant that was lost in the parallel construction of neighborhoods: the order of neighbors!
    *
    * This kernel must give a symmetry invariant, if one node sees a candidate with score "s", then that candidate must also see this node as an option with score "s"!
    */

    /*
    * Idea, dramatic coarsening speedup:
    * - we need to keep connections symmetric -> ids-based and permutation-invariant "deterministic_noise"!
    * - during pairs construction, nudge up or down the hedge’s weight based on a very fast hash of the two node ids involved (order of the node ids must not matter)
    * 
    * TODO: simpler/faster hash, e.g. we could sum the node ids and pick the last 4 bits, adding those to the fixed point weight?
    */

    const uint32_t* my_neighbors = neighbors + neighbors_offsets[warp_id];
    const uint32_t* not_my_neighbors = neighbors + neighbors_offsets[warp_id + 1];
    uint32_t neighbors_count = (uint32_t)(neighbors_offsets[warp_id + 1] - neighbors_offsets[warp_id]);
    extern __shared__ uint32_t histogram[];
    uint32_t* histogram_node = histogram + 3 * HIST_SIZE * (threadIdx.x / WARP_SIZE);
    uint32_t* histogram_score = histogram + 3 * HIST_SIZE * (threadIdx.x / WARP_SIZE) + HIST_SIZE;
    uint32_t* histogram_inbound = histogram + 3 * HIST_SIZE * (threadIdx.x / WARP_SIZE) + 2 * HIST_SIZE;

    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t my_touching_count = touching_offsets[warp_id + 1] - touching_offsets[warp_id];
    const uint32_t my_inbound_count = inbound_count[warp_id];

    const uint32_t my_size = nodes_sizes[warp_id];

    // all threads in the warp should agree on those...
    // TODO: only keep these in lane 0!
    uint32_t best_score[MAX_CANDIDATES];
    thr_init<uint32_t>(best_score, MAX_CANDIDATES, 0);
    uint32_t best_neighbor[MAX_CANDIDATES];
    thr_init<uint32_t>(best_neighbor, MAX_CANDIDATES, UINT32_MAX);

    // handle HIST_SIZE neighbors at a time
    for (; my_neighbors < not_my_neighbors; my_neighbors += HIST_SIZE) {
        // load the first HIST_SIZE neighbors and setup per-thread local histograms, each thread reads and prepares a neighbor
        // TODO: maybe if you have an invalid node, sample another one to fill its place in the histogram?
        uint32_t inserted = 0;
        for (uint32_t nb = lane_id; nb < HIST_SIZE; nb += WARP_SIZE) {
            uint32_t curr_neighbor = UINT32_MAX;
            if (nb < neighbors_count) {
                curr_neighbor = my_neighbors[nb];
                 // skip incompatible neighbors due to size constraints
                if (my_size + nodes_sizes[curr_neighbor] <= max_nodes_per_part) {
                    histogram_node[nb] = curr_neighbor;
                    inserted++;
                } else
                    histogram_node[nb] = UINT32_MAX;
            } else
                histogram_node[nb] = UINT32_MAX;
        }
        // reduce and sync warp after filling histograms
        inserted = warpReduceSum<uint32_t>(inserted);

        // sort the histogram by node-id to then rely on binary search
        wrp_bitonic_sort<uint32_t, HIST_SIZE>(histogram_node);
        __syncwarp();

        // add a little bit of symmetric deterministic noise
        for (uint32_t nb = lane_id; nb < inserted; nb += WARP_SIZE) {
            // NOTE: 'inserted' and the sort already ensure that we only see valid histogram entries
            uint32_t curr_neighbor = histogram_node[nb];
            histogram_score[nb] = deterministic_noise<DETERMINISTIC_SCORE_NOISE>(curr_neighbor, warp_id);
            histogram_inbound[nb] = inbound_count[curr_neighbor];
        }

        // iterate over touching hyperedges
        // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
        for (uint32_t hedge_idx = 0u; hedge_idx < my_touching_count; hedge_idx++) {
            dim_t my_hedge_offset, not_my_hedge_offset;
            uint32_t my_hedge_weight, my_hedge_src_count;
            if (lane_id == 0) {
                const uint32_t actual_hedge_idx = my_touching[hedge_idx];
                my_hedge_offset = hedges_offsets[actual_hedge_idx];
                not_my_hedge_offset = hedges_offsets[actual_hedge_idx + 1];
                my_hedge_weight = (uint32_t)(hedge_weights[actual_hedge_idx]*FIXED_POINT_SCALE);
                my_hedge_src_count = srcs_count[actual_hedge_idx];
            }
            my_hedge_offset = __shfl_sync(0xFFFFFFFF, my_hedge_offset, 0); // each thread in the warp reads one every WARP_SIZE pins
            not_my_hedge_offset = __shfl_sync(0xFFFFFFFF, not_my_hedge_offset, 0);
            my_hedge_weight = __shfl_sync(0xFFFFFFFF, my_hedge_weight, 0);
            my_hedge_src_count = __shfl_sync(0xFFFFFFFF, my_hedge_src_count, 0);
            const uint32_t* my_hedge = hedges + my_hedge_offset;
            const dim_t my_hedge_size = not_my_hedge_offset - my_hedge_offset;
            for (uint32_t i = lane_id; i < my_hedge_size; i += WARP_SIZE) {
                uint32_t pin = my_hedge[i];
                // update local histogram
                const uint32_t hist_idx = binary_search<uint32_t, true>(histogram_node, inserted, pin);
                // NOTE: no atomic needed => all threads in the warp advance in sync and there are not duplicates in hedges, no two threads will ever write the same bin!
                if (hist_idx != UINT32_MAX) {
                    // normalize hedge weight over size
                    histogram_score[hist_idx] += my_hedge_weight / my_hedge_size;
                    if (i >= my_hedge_src_count && hedge_idx < my_inbound_count) // the pin is a destination and the hedge is an inbound-to-me one
                        histogram_inbound[hist_idx]--;
                }
            }
        }

        // warp sync after computing scores
        __syncwarp();

        // delete candidates that would lead to invalid clusters
        // and penalize neighbors by your combined size (symmetric)
        for (uint32_t nb = lane_id; nb < inserted; nb += WARP_SIZE) {
            if (histogram_inbound[nb] + my_inbound_count > max_inbound_per_part) {
                histogram_node[nb] = UINT32_MAX;
                histogram_score[nb] = 0u;
            } /* else if (histogram_node[nb] != UINT32_MAX) {
                uint32_t neigh_size = nodes_sizes[histogram_node[nb]];
                //histogram_score[nb] /= my_size + neigh_size;
                histogram_score[nb] = (uint32_t)((float)histogram_score[nb] * (1 + 1/(float)(my_size + neigh_size)));
                //histogram_score[nb] -= min(my_size + neigh_size, DETERMINISTIC_SCORE_NOISE);
                //if (histogram_score[nb] == 0) histogram_score[nb] = deterministic_noise<DETERMINISTIC_SCORE_NOISE>(histogram_node[nb], warp_id);
            } */
        }

        // sort the histogram by lowest score first (empty bins have score = 0), highest id first as a tiebreaker
        // TODO: maybe replace with a warp-parallel max per candidate iteration below (dunno tho, the max requires a warp-parallel read of the whole histogram)
        //wrp_bitonic_sort_by_key<uint32_t, uint32_t, HIST_SIZE>(histogram_score, histogram_node);
        //__syncwarp();

        // extract the global maximum(s)
        for (int32_t candidate = inserted - 1; candidate >= 0; candidate--) {
            // warp sync on updated histogram (after removing invalid candidates and after removing the last extracted max)
            __syncwarp();
            uint32_t curr_neighbor = UINT32_MAX;
            uint32_t curr_score = 0u;
            uint32_t best_hist_idx = 0u;
            for (uint32_t nb = lane_id; nb < inserted; nb += WARP_SIZE) {
                const uint32_t nb_neighbor = histogram_node[nb];
                const uint32_t nb_score = histogram_score[nb];
                if (nb_score > curr_score || nb_score == curr_score && nb_neighbor > curr_neighbor) { // tie-breaking here too
                    curr_neighbor = nb_neighbor;
                    curr_score = nb_score;
                    best_hist_idx = nb;
                }
            }
            bin<uint32_t> max_neighbor = warpReduceMax<uint32_t>(curr_score, curr_neighbor); // this also tie-breaks
            if (max_neighbor.payload == curr_neighbor) {
                histogram_node[best_hist_idx] = UINT32_MAX;
                histogram_score[best_hist_idx] = 0u;
            }
            curr_neighbor = max_neighbor.payload;
            if (curr_neighbor == UINT32_MAX) break;
            curr_score = max_neighbor.val;
            if (curr_score < best_score[candidates_count - 1]) break;

            // get the best 'candidates_count' candidates out of the histogram
            for (uint32_t i = 0; i < candidates_count; i++) {
                // tie-breaker: higher id node wins; invariant: partial neighbors order
                if (curr_score > best_score[i] || curr_score == best_score[i] && curr_neighbor > best_neighbor[i]) {
                    for (uint32_t j = candidates_count - 1; j > i; j--) {
                        best_score[j] = best_score[j - 1];
                        best_neighbor[j] = best_neighbor[j - 1];
                    }
                    best_score[i] = curr_score;
                    best_neighbor[i] = curr_neighbor;
                    break;
                }
            }
        }
        
        neighbors_count -= HIST_SIZE;
    }

    if (lane_id == 0) {
        for (uint32_t i = 0; i < candidates_count; i++) {
            pairs[warp_id * candidates_count + i] = best_neighbor[i];
            scores[warp_id * candidates_count + i] = best_score[i]; // stay fixed point for now!
        }
    }
}

// create groups of at most MAX_GROUP_SIZE ( =2 for now ) nodes, highest score first
// SEQUENTIAL COMPLEXITY: n*log n
// PARALLEL OVER: n
__global__
void grouping_kernel(
    const uint32_t* __restrict__ pairs, // pairs[idx] is the partner idx wants to be grouped with (UINT32_MAX if undefined)
    const uint32_t* __restrict__ scores, // scores[idx] is the strenght with which idx wants to be grouped with pairs[idx] (0 if undefined)
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_nodes,
    const uint32_t num_repeats,
    const uint32_t candidates_count,
    slot* __restrict__ group_slots, // initialized with -1 on the id
    dp_score* __restrict__ dp_scores, // dynamic programming alternating scores, initialize to 0s
    uint32_t* __restrict__ groups // uninitialized, final group id of each node (non-zero based for now)
) {
    // SETUP FOR GLOBAL SYNC
    cg::grid_group grid = cg::this_grid();

    // STYLE: 'num_repeats' node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tcount = gridDim.x * blockDim.x;

    /*
    * Logic: a walk up (and down) the tree for grouping!
    *
    * => big HP: by construction, a node will always propose a pair with score equal or higher than that of pairs formulated
    *            by others towards him. This HP leaves no room for cycles (longer than two - pairs pointing one to the other)!
    *            The "pairs" build a tree with "roots" that are pairs pointing to each other!
    * => in practice, this HP needs to be slightly stronger, imposing the same score to never happen twice, except on the same edge...
    * 
    * Thus:
    * - one thread per node, all doing this walk up the tree in parallel
    * - go to your target, write (atomically) your ID in its group and write your score IIF your score is higher than what is written there
    * - go to you target's target, write (atomically) your ID in its group and write your target's score IIF it is higher than what is written there
    * - continue this chain until you reach either a node with no target, or a node pointing back
    * - synch once every thread finished the walk
    * - now descend backwards (exactly the same path) where you came from:
    * - if the node you were on one step back still contained the ID of the current node you are on, lock the pair by writing the same ID in the
    *   current node (lit. look one step up the ladder and see if you are still connected)
    * - otherwise, simply continue and you (on the next step) or someone else will lock the current node
    *
    * Nice thing: once a thread descended past a node, you are 100% certain that it knows whether that node already has a pair or not, because every
    *             thread is walking back from the root, and deterministical builds the entire chain required to take all decisions it needs to.
    *             Thus, if one thread passed "before me", it would have 100% taken the same decisions, so we end up writing the same things!
    *
    * Note: if at the end you find a pair pointing one to the other, your root becomes the second node that points back...
    * => this is a choke for atomic operations of a shit ton of threads that followed the same tree! Trivialize the solution of the root!
    *
    * For multiple slots, this can be iterated after deleting the pairs that were already used! Leaving some nodes pointing to "nothing".
    *
    * Required invariant: symmetric connectivity and each node always pick the first, strongest connection (ignore hedge direction when selecting pairs)
    *   => you get a tree with single roots or 2-cycles as roots
    *
    * Could the upward walk be done in a single shot, no path required? Yes! But then you'd loose the ability to deterministially walk back and group by score!
    *
    * Note: the only true requirement is that each thread starts going backward from a point that is guaranteed to determine its first decision correctly
    *       (and failing a max gives this, since you know you will never claim the next node, and so do the other stopping conditions for the upward walk,
    *       the roots). Every decision that follows the first, if the first is correct, is deterministic and identical for all threads!
    *       This means it is fine for threads to non-deterministically pass through some atomic max and continue if they came first!
    *       => the downward pass is designed to "wash out" the fact that multiple threads may have climbed past the same node! The fact that some threads
    *          climbed "too far" and some stopped early wouldn’t change the logical decisions, only how many redundant times those decisions are re-applied.
    *
    * Note: the downward walk could be replaced with a second upward walk, no need to track "path", but requires re-reading pairs...
    */

    /*
    * Beyond just pairs, true multi-group upgrade:
    * - use the "id" from the first slot as the final group reference
    * - going upward, each node has MAX_GROUP_SIZE slots, you claim the first free one or the first one you beat:
    *   - if you find a free slot, atomically settle there
    *   - if your "id" is already there, break (someone else "handled you" already) and continue upward
    *   - if you atomically beat someone, now repeat the process on the next slot, with your candidate becoming the guy you beat (essentially, go down the ladder,
    *     atomically), if you exceed the MAX_GROUP_SIZE slots, ditch the candidate and break
    *     => this should be possible with just atomic_max_on_slot while retrieving the previous slot value, keeping slots sorted from highest to lowest score
    *   - in both above cases, after you wrote your "id" in your target, you attempt to write there also the "id" of the previous node in your path (that is
    *     the previous node you visited while going upward, that currently points to you) with the method, and repeat for up to MAX_GROUP_SIZE-1 nodes going back
    *     - optimization to not be exponential in MAX_GROUP_SIZE: nodes before you will always have a score lower than yours, so you can start searching a slot
    *       for them starting from where you wrote your own "id", and stop (not even continue back in the path) as soon as you finish the available target slots
    *   - mutual pairs take each other's first slot, then proceed to agree on the MAX_GROUP_SIZE-1 best slots among all of theirs (slots are always sorted, use
    *     a routine similar to merge-sort's merge), assuming each also received already the MAX_GROUP_SIZE-1 attempts to slot a node from those that lead to it
    * - going downward:
    *  - if your entry is still there in your target, start copying all its slot over yours, as to propagate downward the assembled group's information
    * Important: at the end of this, all nodes of a group must have IDENTICAL slots (all slots), then the minimum id in the slots will be used to tag the group.
    *
    * TODO: for now, nodes_sizes is not used, because we assume pairs are already filtered by the candidates kernel, however allowing larger groups requries
    *       introducing constraint checks here as well...
    *
    * Beyond just the first choice:
    * - let the candidates kernel return the top-k candidates (sorted) for each node, rather thank just pairs
    * - run grouping like normal (even with multiple slots), but also propagate scores during the downward walk
    * - synch after finishing the downward walk, then repeat the whole up-and-down with the next set of pairs; as they will all have lower scores, none of such
    *   pairs will end up overwriting existing pairings (assumption: a grouping pass only creates optimal pairings)
    *   - immediately return for nodes that already fulfilled their previous pair
    *
    * Note: cycles should still be impossible if I lock previous pairs! Otherwise, looking at all "second pairs" would allow for cycles longer than 2!
    * Note: the global synch means that if a thread returns, the others are deadlocked? Check this! Otherwise just rerurn after the downward path
    *
    * Necessity, this is a cooperative kernel, it must be fully instantiated, even if it means each thread must handle multiple nodes:
    * - multiple nodes per thread (passed as an argument)
    * - compute the ceiling of how many nodes each simultaneously loaded thread would need to handle, from that infer back the exact number of threads,
    *   and run them, each thread repeats the upward and downward walk once for every node it was assigned
    * - the effective PATH_SIZE available to nodes is thus reduced by the number of nodes per thread, but luckly at the beginning, when there are many nodes,
    *   paths are also at their shortest...
    * 
    * From a greedy technique, to a true maximum matching!
    * Exact maximum weighted matching with dynamic programming:
    * - accumulate the score "w/out" on your target
    * - update your next (that of the previous target) score "with" to your score plus the accumulated "w/out" from your children
    * - update your next score "w/out" to "with" the accumulated "w/out" from your children, minus the w/out of the children holding the max / slot (may not be "you"), plus the same children's "with" score
    * => we need to track / accumulate both "with" and "w/out" scores per node as we go up...
    * - the last thread going up the path, as before sees and consolidate the right sums (no early break now)
    * => now each node is claimed by the one such that, if the match is formed, the resulting subtree hash maximum score, accumulating alternating tree costs up through each branch, until the root
    * 
    * TODO: let only threads on leaves do the walks...
    */

    int32_t path_length[MAX_MATCHING_REPEATS];

    uint32_t path[PATH_SIZE];
    const uint32_t actual_path_size = PATH_SIZE / num_repeats;
    uint32_t completed_repeats = 0u;

    for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
        const uint32_t curr_tid = tid + repeat * tcount;
        // initialize yourself to a one-node group, unless someone already claimed you
        if (curr_tid < num_nodes) atomicCAS(&reinterpret_cast<unsigned long long*>(group_slots)[curr_tid], pack_slot(0u, UINT32_MAX), pack_slot(0u, curr_tid));
    }

    for (uint32_t i = 0; i < candidates_count; i++) {
        // repeat for every node that you need to handle
        for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
            const uint32_t curr_tid = tid + repeat * tcount;

            // if you are not a valid node
            if (curr_tid >= num_nodes) break;
            // if you formed a pair, stop
            if (completed_repeats & (1u << repeat)) continue;

            uint32_t* curr_path = path + actual_path_size * repeat;
            int32_t curr_path_length = 0;

            uint32_t current = curr_tid; // current node in the tree of pairs
            uint32_t target = pairs[curr_tid * candidates_count + i]; // target node of "current"
            // total score in the traversed subtree assuming the current node will be paired with its targed
            // initialized to the value with which "current" points to "target"
            uint32_t score_with = scores[curr_tid * candidates_count + i];
            // total score in the traversed subtree assuming the current node will NOT be paired with its targed
            uint32_t score_wout = 0u;

            // go up the tree
            bool outcome = false;
            if (target != UINT32_MAX) {
                outcome = true;
                uint32_t target_target = pairs[target * candidates_count + i]; // target node of "target"
                while (current != target_target || current > target) { // break the two-cycle, always go up to the node of lowest id in the root pair
                    const uint32_t prev_target_score_wout_sum = atomicAdd(&dp_scores[target].with, score_wout); // dp_scores[...].with contains the sum of childrens' wouts!
                    const uint32_t target_score_wout_sum = prev_target_score_wout_sum + score_wout;
                    slot prev_slot;
                    // NOTE: if, by bad luck, the fixed-point score is the same, the id is used as a tie-breaker
                    outcome = atomic_max_on_slot_ret(group_slots, target, current, score_with - score_wout, prev_slot); // atomic are done with the GAIN from the wout->with transition, wins the subtree with the highest gain if given the target
                    if (prev_slot.score == UINT32_MAX) break; // alternative root: a node locked in a previous round
                    //const uint32_t holder_id = outcome ? current : prev_slot.id;
                    const uint32_t holder_with_minus_wout = outcome ? score_with - score_wout : prev_slot.score;
                    // DEBUG: prevent cycles longer than 2!
                    /*bool die = false;
                    for (uint32_t j = 0; j < curr_path_length; j++) {
                        if (curr_path[j] == target) {
                            die = true;
                            printf("Broke cycle between %d -> %d!\n", current, target);
                            break;
                        }
                    }
                    if (die) break;*/
                    assert(curr_path_length < actual_path_size);
                    curr_path[curr_path_length++] = target;
                    current = target;
                    target = target_target;
                    if (target == UINT32_MAX) break; // alternative root: a node with no target
                    target_target = pairs[target_target * candidates_count + i]; // if this goes to -1, stop after the next iteration, as to still handle the current "target" that will be the "root"
                    score_wout = target_score_wout_sum + /* + with[holder] - wout[holder] */ holder_with_minus_wout; // if the next node up the tree won't get paired, then his score is that of the best children's "with", and the other "w/out"
                    score_with = target_score_wout_sum + scores[current * candidates_count + i]; // if the next node up the tree will get paired, then his score is his own candidate score, and all children's "w/out"
                }
                // no need to handle root pair, since we already broke the pair and made the lower-id node the sole root
                // => the result does not change, since the lower-id node will be contended between the gains of the two subtrees
                // handle "root(s)" as a pair of nodes pointing to each other from the POV of the lower-id node of the two
                //if (current == target_target && current < target) {
                    // tie back to your target (other root node) iff there is more to gain
                    //const uint32_t with_sum = score_with + atomicAdd(&dp_scores[target].with, 0); // sum of the two roots' with dp_score and their score
                    //const uint32_t holder_with_minus_wout = get_slot(group_slots, target).score;
                    //const uint32_t wout_sum = score_wout + atomicAdd(&dp_scores[target].with, 0) + holder_with_minus_wout; // sum of the two root's wout dp_scores
                    //if (with_sum > wout_sum) ...
                //}
            }

            path_length[repeat] = curr_path_length;
        }

        // global synch
        grid.sync();

        // repeat for every node that you need to handle
        for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
            const uint32_t curr_tid = tid + repeat * tcount;

            // if you are not a valid node
            if (curr_tid >= num_nodes) break;
            // if you formed a pair, stop
            if (completed_repeats & (1u << repeat)) continue;

            uint32_t* curr_path = path + actual_path_size * repeat;
            int32_t curr_path_length = path_length[repeat];
            path_length[repeat] = 0;

            // go down the tree
            // it the root had no target, path[path_length - 1] is the root, if the root was a mutual-poiting pair, path[path_length - 1] is the first encountered node of the pair
            uint32_t current, target;
            for (curr_path_length = curr_path_length - 2; curr_path_length >= 0; curr_path_length--) {
                // NOTE: on the first iteration, coming from the upward walk, "current" is the last node that got its slot updated
                target = curr_path[curr_path_length + 1]; // this is "who current would have liked to be with", the node one step back up the ladder towards root
                current = curr_path[curr_path_length];
                if (group_slots[target].id == current) {
                    // TODO: maybe writing only after reading and checking if someone else already wrote can spare cache coherence chaos - concurrent writes are still fine
                    group_slots[current].id = current;
                    // NOTE: lock groups by setting scores to the maximum
                    group_slots[current].score = UINT32_MAX;
                    group_slots[target].score = UINT32_MAX;
                    // NOTE: after a successful link the next node down the path has already lost its target, so we might as well skip it
                    curr_path_length--;
                }
            }
            // the path did not include the "curr_tid" node, handle it here iff you did not happen to group path[0] with path[1] during the last iteration above (leading to path_length == -2)
            if (curr_path_length == -1) {
                target = curr_path[0];
                current = curr_tid;
                if (group_slots[target].id == current) {
                    group_slots[current].id = current;
                    // lock groups by setting scores to the maximum
                    group_slots[current].score = UINT32_MAX;
                    group_slots[target].score = UINT32_MAX;
                }
            }
        }
        
        // global synch
        grid.sync();

        // anyone not yet selected (score != UINT32_MAX), set your score to 0 to enable the next pairing round, then sync again and continue
        for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
            const uint32_t curr_tid = tid + repeat * tcount;
            if (curr_tid >= num_nodes) break;
            // if you formed a pair, stop
            if (completed_repeats & (1u << repeat)) continue;
            if (group_slots[curr_tid].score == UINT32_MAX) {
                completed_repeats |= (1u << repeat);
                continue;
            }
            group_slots[curr_tid].score = 0;
            dp_scores[curr_tid] = (dp_score){ 0u, 0u };
        }

        // global synch
        grid.sync();
    }

    // write inside "groups" the minimum id among each node's slots, used to identify its group, eventually, zero-base those ids
    for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
        const uint32_t curr_tid = tid + repeat * tcount;
        if (curr_tid >= num_nodes) break;
        groups[curr_tid] = group_slots[curr_tid].id;
    }
}