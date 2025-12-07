#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "utils.cuh"

namespace cg = cooperative_groups;

// REMEMBER: "const" means the data pointed to is not modified, not the pointer itself!

// count the number of distinct neighbors of each node (to then perform a scan and find offsets where to write the neighborhoods)
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
__global__
void neighborhoods_count_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t num_nodes,
    uint32_t* neighbors_offsets // here filled as counters of "how many neighbors per node" -> then do a prefix sum for the offsets
) {
    /*
    * Idea: BIN pattern of count->scan->scatter
    */

    // STYLE: one node per block, one touching hedge per warp, distinct neighbors in shared memory!
    const uint32_t node_id = blockIdx.x;
    if (node_id >= num_nodes) return;

    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // local per block - coincides with the touching hedge to handle
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const uint32_t* my_touching = touching + touching_offsets[node_id];
    //const uint32_t* not_my_touching = touching + touching_offsets[node_id + 1];
    const uint32_t my_touching_count = touching_offsets[node_id + 1] - touching_offsets[node_id];
    
    // hash-set for deduplication
    __shared__ uint32_t dedupe[SM_MAX_DEDUPE_BUFFER_SIZE];
    __shared__ uint32_t seen_distinct_total;
    uint32_t seen_distinct = 0;
    
    // initialize shared memory
    sm_init(dedupe, SM_MAX_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    if (threadIdx.x == 0)
        seen_distinct_total = 0;
    __syncthreads();
    
    // can return only after helping initializing shared memory
    if (warp_id >= my_touching_count) return;
    
    // TOOD: could optimize by iterating directly on pointers
    for (uint32_t touching_hedge_idx = warp_id; touching_hedge_idx < my_touching_count; touching_hedge_idx += warps_per_block) { // the block loops over touching hedges
        if (touching_hedge_idx < my_touching_count) {
            const uint32_t my_hedge_idx = my_touching[touching_hedge_idx];
            const uint32_t* my_hedge = hedges + hedges_offsets[my_hedge_idx];
            //const uint32_t* not_my_hedge = hedges + hedges_offsets[my_hedge_idx + 1];
            const uint32_t my_hedge_size = hedges_offsets[my_hedge_idx + 1] - hedges_offsets[my_hedge_idx];
            for (uint32_t node_idx = lane_id; node_idx < my_hedge_size; node_idx += WARP_SIZE) { // the warp loops over hedge pins
                if (node_idx < my_hedge_size) {
                    uint32_t neighbor = my_hedge[node_idx];
                    if (neighbor != node_id && sm_hashset_insert(dedupe, SM_MAX_DEDUPE_BUFFER_SIZE, neighbor))
                        seen_distinct++;
                }
            }
        }
    }

    // reduce distinct counts per-warp
    seen_distinct = warpReduceSum<uint32_t>(seen_distinct);

    // accumulate counts per-block
    if (lane_id == 0)
        atomicAdd(&seen_distinct_total, seen_distinct);
    __syncthreads();

    if (threadIdx.x == 0)
        neighbors_offsets[node_id] = seen_distinct_total;
}

// compute the distinct neighbors of each node (again) and write them at the pre-computed offsets in global memory
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
__global__
void neighborhoods_scatter_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t num_nodes,
    const uint32_t* neighbors_offsets,
    uint32_t* neighbors
) {
    // STYLE: one node per block, one touching hedge per warp, distinct neighbors in shared memory!
    const uint32_t node_id = blockIdx.x;
    if (node_id >= num_nodes) return;

    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // local per block - coincides with the touching hedge to handle
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const uint32_t* my_touching = touching + touching_offsets[node_id];
    //const uint32_t* not_my_touching = touching + touching_offsets[node_id + 1];
    const uint32_t my_touching_count = touching_offsets[node_id + 1] - touching_offsets[node_id];

    uint32_t* my_neighbors = neighbors + neighbors_offsets[node_id];

    // hash-set for deduplication
    __shared__ uint32_t dedupe[SM_MAX_DEDUPE_BUFFER_SIZE];
    __shared__ uint32_t seen_distinct_total;
    
    // initialize shared memory
    sm_init(dedupe, SM_MAX_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    if (threadIdx.x == 0)
        seen_distinct_total = 0;
    __syncthreads();

    // can return only after helping initializing shared memory
    if (warp_id >= my_touching_count) return;

    // TOOD: could optimize by iterating directly on pointers
    for (uint32_t touching_hedge_idx = warp_id; touching_hedge_idx < my_touching_count; touching_hedge_idx += warps_per_block) { // the block loops over touching hedges
        if (touching_hedge_idx < my_touching_count) {
            const uint32_t my_hedge_idx = my_touching[touching_hedge_idx];
            const uint32_t* my_hedge = hedges + hedges_offsets[my_hedge_idx];
            //const uint32_t* not_my_hedge = hedges + hedges_offsets[my_hedge_idx + 1];
            const uint32_t my_hedge_size = hedges_offsets[my_hedge_idx + 1] - hedges_offsets[my_hedge_idx];
            for (uint32_t node_idx = lane_id; node_idx < my_hedge_size; node_idx += WARP_SIZE) { // the warp loops over hedge pins
                if (node_idx < my_hedge_size) {
                    uint32_t neighbor = my_hedge[node_idx];
                    if (neighbor != node_id && sm_hashset_insert(dedupe, SM_MAX_DEDUPE_BUFFER_SIZE, neighbor)) {
                        uint32_t offset = atomicAdd(&seen_distinct_total, 1); // returns the value as it was before the increment (reserving that idx)
                        my_neighbors[offset] = neighbor;
                    }
                }
            }
        }
    }

    // TODO: could remove this, but for now keep it, just to make sure the hash-set is deterministic
    __syncthreads();
    if (threadIdx.x == 0)
        assert(neighbors_offsets[node_id + 1] - neighbors_offsets[node_id] == seen_distinct_total);
}


// find the best neighbor for each node to stay with (edge-coarsening)
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
// SHUFFLES OVER: h*d (neighbors)
__global__
void candidates_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* neighbors,
    const uint32_t* neighbors_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    uint32_t* pairs,
    float* scores
) {
    // STYLE: one node per warp!
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;

    /*
    * Idea:
    * - one node per warp
    * - same part of the histogram (one bin per neighbor) in each thread's registers
    * - scan hyperedges (with caching in shared memory - either passive or automatic) once per histogram part
    * - warp primitives to both reduce each bin and find the maximum bin
    *
    * It is 100% possible for two neighbors of a node to partake, with that node, in the same hedges! And this makes them peer pairing candidates.
    * => Use fixed point, not floats, because we need associativity to find those peer candidates!
    * => In case of a tie, deterministically update the best by lower ID! This is the invariant that was lost in the parallel construction of neighborhoods: the order of neighbors!
    *
    * This kernel must give a symmetry invariant, if one node sees a candidate with score "s", then that candidate must also see this node as an option with score "s"!
    */

    const uint32_t* my_neighbors = neighbors + neighbors_offsets[warp_id];
    const uint32_t* not_my_neighbors = neighbors + neighbors_offsets[warp_id + 1];
    uint32_t neighbors_count = neighbors_offsets[warp_id + 1] - neighbors_offsets[warp_id];
    bin histogram[HIST_SIZE]; // make sure this fits in registers (no spill) !!

    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];

    // all threads in the warp should agree on those...
    uint32_t best_score[MAX_CANDIDATES] = {0};
    uint32_t best_neighbor[MAX_CANDIDATES] = {UINT32_MAX};

    // handle HIST_SIZE neighbors at a time
    for (; my_neighbors < not_my_neighbors; my_neighbors += HIST_SIZE) {
        // load the first HIST_SIZE neighbors and setup per-thread local histograms
        for (uint32_t nb = 0; nb < HIST_SIZE; nb++) {
            if (nb < neighbors_count) {
                const uint32_t my_neighbor = my_neighbors[nb];
                histogram[nb].node = my_neighbor;
                //if (my_neighbor == UINT32_MAX) // reached blank neighbors area left by coarsening
                //    neighbors_count = 0;
            } else
                histogram[nb].node = UINT32_MAX;
            histogram[nb].score = 0;
        }

        // TODO: shared memory caching of hyperedges!!

        // scan touching hyperedges
        // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            const uint32_t actual_hedge_idx = *hedge_idx;
            const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
            my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
            const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
            const uint32_t my_hedge_weight = (uint32_t)(hedge_weights[actual_hedge_idx]*FIXED_POINT_SCALE);
            for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
                uint32_t pin = UINT32_MAX - 1;
                if (my_hedge < not_my_hedge)
                    pin = *my_hedge;
                // update local histogram
                for (uint32_t nb = 0; nb < HIST_SIZE; nb++) {
                    if (pin == histogram[nb].node)
                        histogram[nb].score += my_hedge_weight;
                }
                // DEBUG: do we have all neighbors?
                /*bool ok = false;
                for (const uint32_t *nb = neighbors + neighbors_offsets[warp_id]; nb < not_my_neighbors; nb++) {
                    if (pin == *nb || pin == UINT32_MAX - 1 || pin == warp_id) {
                        ok = true;
                        break;
                    }
                }
                if (!ok) printf("Missing some neighbors (node %d, neighbor %d)!\n", warp_id, pin);*/
            }
        }

        // tie-breaker: lower id node wins; invariant: partial neighbors order

        /*
        * Idea, dramatic coarsening speedup:
        * - we need to keep connections symmetric
        * - during pairs construction, nudge up or down the hedge’s weight based on a very fast hash of the two node ids involved (order of the node ids must not matter)
        * => we could sum the node ids and pick the last 4 bits, adding those to the fixed point weight?
        */

        // reduce local histograms between threads (each thread will see the full histogram)
        // TODO: could stop at min(HIST_SIZE, neighbors_count) if we don't set neighbors_count to 0 as a stopping condition earliers...
        for (uint32_t nb = 0; nb < HIST_SIZE; nb++)
            histogram[nb].score = warpReduceSum<uint32_t>(histogram[nb].score);

        // reduce max in histogram between threads (each thread grabs a different bin)
        for (uint32_t nb = lane_id; nb < HIST_SIZE; nb += WARP_SIZE) {
            // get the best MAX_CANDIDATES candidates out of the histogram
            uint32_t curr_neighbor = histogram[nb].node;
            uint32_t curr_score = histogram[nb].score;
            for (uint32_t i = 0; i < MAX_CANDIDATES; i++) {
                bin max = warpReduceMax(curr_score, curr_neighbor);
                // TODO: only "lane_id = 0" actually needs to do all of this...
                for (uint32_t j = 0; j < MAX_CANDIDATES; j++) {
                    if (max.score > best_score[j] || max.score == best_score[j] && max.node < best_neighbor[j]) {
                        for (uint32_t t = MAX_CANDIDATES - 1; t > j; t--) {
                            best_score[t] = best_score[t - 1];
                            best_neighbor[t] = best_neighbor[t - 1];
                        }
                        best_score[j] = max.score;
                        best_neighbor[j] = max.node;
                        break;
                    }
                }
                if (curr_neighbor == max.node) {
                    curr_neighbor = UINT32_MAX;
                    curr_score = 0;
                }
            }
        }

        neighbors_count -= HIST_SIZE;
    }

    if (lane_id == 0) {
        for (uint32_t i = 0; i < MAX_CANDIDATES; i++) {
            pairs[warp_id * MAX_CANDIDATES + i] = best_neighbor[i];
            scores[warp_id * MAX_CANDIDATES + i] = ((float)best_score[i])/FIXED_POINT_SCALE;
        }
    }
}

// create groups of at most MAX_GROUP_SIZE nodes, highest score first
// TODO: currently MAX_GROUP_SIZE is ignored, and groups are of at most 2!
// SEQUENTIAL COMPLEXITY: n*log n
// PARALLEL OVER: n
__global__
void grouping_kernel(
    const uint32_t* pairs, // pairs[idx] is the partner idx wants to be grouped with
    const float* scores, // pairs[idx] is the partner idx wants to be grouped with
    const uint32_t num_nodes,
    slot* group_slots, // initialized with -1 on the id
    uint32_t* groups // uninitialized, final group id of each node (non-zero based for now)
) {
    // SETUP FOR GLOBAL SYNC
    cg::grid_group grid = cg::this_grid();

    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) {
        for (uint32_t i = 0; i < 2*MAX_CANDIDATES; i++)
            grid.sync();
        return;
    }

    int32_t path_length = 0;
    uint32_t path[PATH_SIZE];

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
    */

    /*
    * Beyond just pairs, true multi-group upgrade:
    * - use the "id" from the first slot as the final group reference
    * - going upward, each node has MAX_GROUP_SIZE slots, you claim the first free one or the first one you beat:
    *   - if you find a free slot, atomically settle there
    *   - if your "id" is already there, break (someone else "handled you" already)
    *   - if you atomically beat someone, now repeat the process on the next slot, with your candidate becoming the guy you beat (essentially, go down the ladder,
    *     atomically), if you exceed the MAX_GROUP_SIZE slots, ditch the candidate and break
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
    * Beyond just the first choice:
    * - let the candidates kernel return the top-k candidates (sorted) for each node, rather thank just pairs
    * - run grouping like normal (even with multiple slots), but also propagate scores during the downward walk
    * - synch after finishing the downward walk, then repeat the whole up-and-down with the next set of pairs; as they will all have lower scores, none of such
    *   pairs will end up overwriting existing pairings (assumption: a grouping pass only creates optimal pairings)
    *   - immediately return for nodes that already fulfilled their previous pair
    *
    * Note: cycles should still be impossible if I lock previous pairs! Otherwise, looking at all "second pairs" would allow for cycles longer than 2!
    * Note: the global synch means that if a thread returns, the others are deadlocked? Check this! Otherwise just rerurn after the downward path
    */

    // initialize yourself to a one-node group, unless someone already claimed you
    atomicCAS(&group_slots[tid*MAX_GROUP_SIZE].id, UINT32_MAX, tid);

    for (uint32_t i = 0; i < MAX_CANDIDATES; i++) {
        // if you formed a pair, stop
        if (group_slots[tid*MAX_GROUP_SIZE].score == UINT32_MAX) {
            grid.sync();
            grid.sync();
            continue;
        }

        uint32_t current = tid; // current node in the tree of pairs
        uint32_t target = pairs[tid * MAX_CANDIDATES + i]; // target node of "current"
        uint32_t score = (uint32_t)(scores[tid * MAX_CANDIDATES + i]*FIXED_POINT_SCALE); // score with which "current" points to "target"
        
        // go up the tree
        bool outcome = false;
        if (target != UINT32_MAX) {
            uint32_t target_target = pairs[target * MAX_CANDIDATES + i]; // target node of "target"
            while (current != target_target) {
                // NOTE: if, by bad luck, the fixed-point score is the same, the id is used as a tie-breaker
                outcome = atomic_max_on_slot(group_slots, target*MAX_GROUP_SIZE, current, score);
                // NOTE: if the atomic fails you are surely not the maximum, thus you can stop here and not even bother continuing (we then start going down without even looking again at this target)
                if (!outcome) break;
                // DEBUG: prevent cycles longer than 2!
                /*bool die = false;
                for (uint32_t j = 0; j < path_length; j++) {
                    if (path[j] == target) {
                        die = true;
                        printf("Broke cycle between %d -> %d!\n", current, target);
                        break;
                    }
                }
                if (die) break;*/
                assert(path_length < PATH_SIZE);
                path[path_length++] = target;
                current = target;
                target = target_target;
                if (target == UINT32_MAX) break; // alternative root: a node with no target
                target_target = pairs[target_target * MAX_CANDIDATES + i]; // if this goes to -1, stop after the next iteration, as to still handle the current "target" that will be the "root"
                score = (uint32_t)(scores[current * MAX_CANDIDATES + i]*FIXED_POINT_SCALE);
            }
        }
        // handle "root(s)" as a pair of nodes pointing to each other
        // NOTE: concurrent writes are fine, anyone seing those two nodes will write the same thing: "this is a group" or "would you get married already!?"
        if (outcome) { // "&& target != UINT32_MAX" is implied by "outcome" not being false
            const uint32_t lowest_id = min(current, target);
            group_slots[current*MAX_GROUP_SIZE].id = lowest_id;
            group_slots[current*MAX_GROUP_SIZE].score = score;
            group_slots[target*MAX_GROUP_SIZE].id = lowest_id;
            group_slots[target*MAX_GROUP_SIZE].score = score;
        }

        // global synch
        grid.sync();

        // go down the tree
        // it the root had no target, path[path_length - 1] is the root, if the root was a mutual-poiting pair, path[path_length - 1] is the first encountered node of the pair
        for (path_length = path_length - 2; path_length >= 0; path_length--) {
            // NOTE: on the first iteration, coming from the upward walk, "current" is the last node that got its slot updated
            target = current; // this is "who current would have liked to be with", the node one step back up the ladder towards root
            current = path[path_length];
            if (group_slots[target*MAX_GROUP_SIZE].id == current) {
                // TODO: maybe writing only after reading and checking if someone else already wrote can spare cache coherence chaos - concurrent writes are still fine
                group_slots[current*MAX_GROUP_SIZE].id = current;
                // NOTE: lock groups by setting scores to the maximum
                group_slots[current*MAX_GROUP_SIZE].score = UINT32_MAX;
                group_slots[target*MAX_GROUP_SIZE].score = UINT32_MAX;
                // NOTE: after a successful link the next node down the path has already lost its target, so we might as well skip it
                path_length--;
            }
        }
        // the path did not include the "tid" node, handle it here iff you did not happen to group path[0] with path[1] during the last iteration above (leading to path_length == -2)
        if (path_length == -1) {
            target = current;
            current = tid;
            if (group_slots[target*MAX_GROUP_SIZE].id == current) {
                group_slots[current*MAX_GROUP_SIZE].id = current;
                 // lock groups by setting scores to the maximum
                group_slots[current*MAX_GROUP_SIZE].score = UINT32_MAX;
                group_slots[target*MAX_GROUP_SIZE].score = UINT32_MAX;
            }
        }
        
        // global synch
        grid.sync();

        path_length = 0;
    }

    // write inside "groups" the minimum id among each node's slots, used to identify its group, eventually, zero-base those ids
    uint32_t min_id = UINT32_MAX;
    for (uint32_t i = 0; i < MAX_GROUP_SIZE; i++)
        min_id = min(group_slots[tid*MAX_GROUP_SIZE + i].id, min_id);
    groups[tid] = min_id;
}

// count how many distinct new pins are in each hedge
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void apply_coarsening_hedges_count(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t num_hedges,
    const uint32_t* groups, // groups[node idx] -> new group/node id
    uint32_t* coarse_hedges_offsets // coarse_hedges_offsets[hedge idx] -> count of distinct groups among its pins
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /*
    * Idea:
    * - first deduplicate hedges to count their new, distinct, nodes
    * - scan the counts per hedge to get the new offsets
    * - repeat the deduplication and write (scatter) each new hedge to its offset
    *
    * TODO, upgrade options:
    * - instead of three kernels to count distinct nodes, scan offsets, and scatter distinct nodes, can't we do all in one?
    * - for now we just do atomics towards global memory, because nodes are too many for shared memory, eventually:
    *   => use the shared memory hash-map to keep a "first-come-first-served" set of counters in shared memory,
    *     and reditect additional entries to global counters, then atomically increment global memory
    */

    const uint32_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t pins[MAX_DEDUPE_BUFFER_SIZE];
    // TODO: maybe "memset(pins, 0xFF, sizeof(pins))" is faster?
    lm_init(pins, MAX_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t pin = groups[*curr]; // read and map pin to its new id
        bool not_seen = lm_hashset_insert(pins, MAX_DEDUPE_BUFFER_SIZE, pin);
        if (not_seen) {
            distinct++;
        }
    }

    // leave the first entry to be 0 (offset of the first hedge)
    coarse_hedges_offsets[tid + 1] = distinct;
}

// write the new distinct pins of hedges
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void apply_coarsening_hedges_scatter(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t num_hedges,
    const uint32_t* groups, // groups[node idx] -> new group/node id
    const uint32_t* coarse_hedges_offsets, // coarse_hedges_offsets[hedge idx] -> count of distinct groups among its pins
    uint32_t* coarse_hedges
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    // Idea: second part of "apply_coarsening_hedges_count" -> scatter

    const uint32_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    const uint32_t coarse_hedge_start_idx = coarse_hedges_offsets[tid];
    const uint32_t coarse_hedge_size = coarse_hedges_offsets[tid + 1] - coarse_hedges_offsets[tid];
    uint32_t *coarse_hedge_start = coarse_hedges + coarse_hedge_start_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t pins[MAX_DEDUPE_BUFFER_SIZE];
    lm_init(pins, MAX_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t pin = groups[*curr]; // read and map pin to its new id
        bool not_seen = lm_hashset_insert(pins, MAX_DEDUPE_BUFFER_SIZE, pin);
        if (not_seen) {
            assert(distinct < coarse_hedge_size);
            coarse_hedge_start[distinct] = pin;
            distinct++;
        }
    }
}

// count how many distinct neighbors there are in each group
// SEQUENTIAL COMPLEXITY: n*d*h (in reality there are <<d*h neighbors per node)
// PARALLEL OVER: n
__global__
void apply_coarsening_neighbors_count(
    const uint32_t* neighbors,
    const uint32_t* neighbors_offsets,
    const uint32_t* groups, // groups[node id] -> node's group id
    const uint32_t* ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const uint32_t* ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    uint32_t* coarse_neighbors_offsets // group id -> count of distinct neighbors
) {
    // STYLE: one group (new node) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_groups) return;

    /*
    * Idea:
    * - first translate to their new id and deduplicate neighbors inside each group to count their new, distinct, nodes
    * - scan the counts per group to get the new offsets
    * - repeat the deduplication and write (scatter) each new neighbor (neighboring group id) to its offset
    *
    * NOTE: by doing this, instead of rebuilding neighborhoods from scratch, we have fewer neighbors to deduplicate,
    *       since most were already handled at the level above! While this runs we keep allocated both the old and new sets...
    *
    * TODO: compare this with rebuilding neighborhoods from scratch, to see if it is faster! (it is worth the memory investment)
    */

    /*
    * TODO:
    * - the neighbors of multiple nodes combined easily exceed 8k, sets are so large that not even grouping nodes makes their merged version shrink...
    *   => we can continously use larger and larger local buffers..
    * - upgrade this to be like the initial construction of neighbors, using shared memory and warps
    * - use the one-block per group or one-warp per group method and move the hash set in SM, also then have warps read neighbors in //
    */

    const uint32_t ungroups_start_idx = ungroups_offsets[tid], ungroups_end_idx = ungroups_offsets[tid + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t new_neighbors[MAX_LARGE_DEDUPE_BUFFER_SIZE];
    lm_init(new_neighbors, MAX_LARGE_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

    // for every original node in the group, go over its touching set
    for (const uint32_t* ungroup = ungroups_start; ungroup < ungroups_end; ungroup++) {
        const uint32_t node = *ungroup;
        const uint32_t* my_neighbors = neighbors + neighbors_offsets[node];
        const uint32_t* not_my_neighbors = neighbors + neighbors_offsets[node + 1];
        for (const uint32_t* curr_neighbor = my_neighbors; curr_neighbor < not_my_neighbors; curr_neighbor++) {
            const uint32_t new_neighbor = groups[*curr_neighbor]; // translate to group id
            if (tid == new_neighbor)
                continue;
            bool not_seen = lm_hashset_insert(new_neighbors, MAX_LARGE_DEDUPE_BUFFER_SIZE, new_neighbor);
            if (not_seen)
                distinct++;
        }
    }

    // leave the first entry to be 0 (offset of the first set)
    coarse_neighbors_offsets[tid + 1] = distinct;
}

// write distinct neighbors
// SEQUENTIAL COMPLEXITY: n*d*h (in reality there are <<d*h neighbors per node)
// PARALLEL OVER: n
__global__
void apply_coarsening_neighbors_scatter(
    const uint32_t* neighbors,
    const uint32_t* neighbors_offsets,
    const uint32_t* groups, // groups[node id] -> node's group id
    const uint32_t* ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const uint32_t* ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const uint32_t* coarse_neighbors_offsets, // group id -> count of distinct neighbors
    uint32_t* coarse_neighbors
) {
    // STYLE: one group (new node) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_groups) return;

    // Idea: second part of "apply_coarsening_neighbors_count" -> scatter

    const uint32_t ungroups_start_idx = ungroups_offsets[tid], ungroups_end_idx = ungroups_offsets[tid + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    const uint32_t coarse_neighbors_start_idx = coarse_neighbors_offsets[tid];
    const uint32_t coarse_neighbors_size = coarse_neighbors_offsets[tid + 1] - coarse_neighbors_offsets[tid];
    uint32_t *coarse_neighbors_start = coarse_neighbors + coarse_neighbors_start_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t new_neighbors[MAX_LARGE_DEDUPE_BUFFER_SIZE];
    lm_init(new_neighbors, MAX_LARGE_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

    // for every original node in the group, go over its touching set
    for (const uint32_t* ungroup = ungroups_start; ungroup < ungroups_end; ungroup++) {
        const uint32_t node = *ungroup;
        const uint32_t* my_neighbors = neighbors + neighbors_offsets[node];
        const uint32_t* not_my_neighbors = neighbors + neighbors_offsets[node + 1];
        for (const uint32_t* curr_neighbor = my_neighbors; curr_neighbor < not_my_neighbors; curr_neighbor++) {
            const uint32_t new_neighbor = groups[*curr_neighbor]; // translate to group id
            if (tid == new_neighbor)
                continue;
            bool not_seen = lm_hashset_insert(new_neighbors, MAX_LARGE_DEDUPE_BUFFER_SIZE, new_neighbor);
            if (not_seen) {
                assert(distinct < coarse_neighbors_size);
                coarse_neighbors_start[distinct] = new_neighbor;
                distinct++;
            }
        }
    }
}

// count how many hedges touch each node
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void apply_coarsening_touching_count(
    const uint32_t* hedges, // already coarsened as of here, thus contain group ids!
    const uint32_t* hedges_offsets,
    const uint32_t num_hedges,
    uint32_t* coarse_touching_offsets // here filled as counters of "how many hedges per node" -> then do a prefix sum for the offsets
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /*
    * Idea:
    * - first coarsen hedge, then going over coarse hedge and counting the occurrencies of each group (new node) should be
    *   faster than just deduplicating touching hedges between nodes in the same group
    * - scan the touching counts per group (new node) to get the new offsets
    * - OLD METHOD: repeat the above while deduplicating per-group and writing (scatter) each hedge to its offset in the new touching sets
    * - NEW METHOD: change the way the scatter operates, assign one thread per group, the thread goes over the nodes
    *   in the group via the "ungroups" and "ungroups_offsets" structures, deduplicates hedges and writes them in the touching set

    * TODO, upgrade options:
    * - for now we just do atomics towards global memory, because nodes are too many for shared memory, eventually:
    *   => use the shared memory hash-map to keep a "first-come-first-served" set of counters in shared memory,
    *     and reditect additional entries to global counters, then atomically increment global memory
    */

    const uint32_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;

    // TODO: initialize shared memory
    // in utils.cuh : #define SM_MAX_HASHMAP_SIZE 4096u ?
    //__shared__ hashmap_entry hashmap[SM_MAX_HASHMAP_SIZE];
    //sm_init(hashmap, SM_MAX_HASHMAP_SIZE*2, HASH_EMPTY);

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t pin = *curr; // already a group id
        atomicAdd(&coarse_touching_offsets[pin + 1], 1); // leave the first entry to be 0 (offset of the first set)
    }
}

// write distinct touching hedges
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
__global__
void apply_coarsening_touching_scatter(
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t* ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const uint32_t* ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const uint32_t* coarse_touching_offsets, // group id -> count of distinct touching hedges
    uint32_t* coarse_touching
) {
    // STYLE: one group (new node) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_groups) return;

    // Idea: second part of "apply_coarsening_touching_scatter" -> scatter
    // Alternative version: one thread per hedge and dedupe inside each touching set (by linearly going over it, with a CAS at the end)

    const uint32_t ungroups_start_idx = ungroups_offsets[tid], ungroups_end_idx = ungroups_offsets[tid + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    const uint32_t coarse_touching_start_idx = coarse_touching_offsets[tid];
    const uint32_t coarse_touching_size = coarse_touching_offsets[tid + 1] - coarse_touching_offsets[tid];
    uint32_t *coarse_touching_start = coarse_touching + coarse_touching_start_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t new_touching[MAX_DEDUPE_BUFFER_SIZE];
    lm_init(new_touching, MAX_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

    // for every original node in the group, go over its touching set
    for (const uint32_t* ungroup = ungroups_start; ungroup < ungroups_end; ungroup++) {
        const uint32_t node = *ungroup;
        const uint32_t* my_touching = touching + touching_offsets[node];
        const uint32_t* not_my_touching = touching + touching_offsets[node + 1];
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            const uint32_t actual_hedge_idx = *hedge_idx;
            bool not_seen = lm_hashset_insert(new_touching, MAX_DEDUPE_BUFFER_SIZE, actual_hedge_idx);
            if (not_seen) {
                assert(distinct < coarse_touching_size);
                coarse_touching_start[distinct] = actual_hedge_idx;
                distinct++;
            }
        }
    }
}


// write to each node the partition of its group
__global__
void apply_uncoarsening_partitions(
    const uint32_t* groups, // groups[node id] -> node's group
    const uint32_t* coarse_partitions, // coarse_partitions[group id] -> group's partition
    const uint32_t num_nodes,
    uint32_t* partitions // partitions[node id] -> group's partition
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    // Idea: gather, each node reads its group, and from it its partition

    const uint32_t my_group = groups[tid];
    const uint32_t my_partition = coarse_partitions[my_group];
    partitions[tid] = my_partition;
}

// for each hyperedge, count how many of its pins are in each partition
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void pins_per_partition_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* pins_per_partitions // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /*
    * TODO Upgrade:
    * - one hedge per warp
    * - shared memory histogram per hyperedge
    */

    const uint32_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    uint32_t *my_pins_per_partitions = pins_per_partitions + tid * num_partitions;

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t part = partitions[*curr];
        atomicAdd(&my_pins_per_partitions[part], 1);
    }
}

// find moves of nodes from one partition to another that yield a positive gain
// SEQUENTIAL COMPLEXITY: n*h*partitions
// PARALLEL OVER: n
// SHUFFLES OVER: partitions
__global__
void fm_refinement_gains_kernel(
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t* partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t* pins_per_partitions, // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    // NOTE: we repurpose the arrays allocated for the "candidates kernel" for those!
    uint32_t* moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx
    float* scores // scores[idx] -> gain for move in position idx
) { 
    // STYLE: one node per warp!
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;
    
    /*
    * Idea:
    * - one node per warp
    * - same part of the histogram (one bin per partition) in each thread's registers
    * - scan hyperedges, specifically their pins per hedge (with caching in shared memory - either passive or automatic) once per histogram part
    * - warp primitives to both reduce each bin and find the maximum bin
    *
    * TODO: this kernel could undergo the same multi-candidate upgrade as "pairs"! Tho here it would be much harder with the moves sorting mechanism...
    *
    * TODO: like in HyperG, we could repurpose neighbors to keep a list of neighboring hedges to each node (maybe one-hot encoded), and thus
    *       not build the full histogram, but build it only for those neighboring partitions...
    *
    * NOTE: no need for FIXED POINT here, since we don't need neither symmetry nor invariants!
    */

    const uint32_t my_partition = partitions[warp_id];

    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
    //uint32_t touching_count = touching_offsets[warp_id + 1] - touching_offsets[warp_id];
    float histogram[HIST_SIZE]; // make sure this fits in registers (no spill) !! Store here only the score, the partition id can be inferred!

    // all threads in the warp should agree on those...
    uint32_t best_score = 0;
    uint32_t best_move = UINT32_MAX;

    // handle HIST_SIZE partitions at a time
    for (uint32_t curr_base_part = 0; curr_base_part < num_partitions; curr_base_part += HIST_SIZE) {
        // clear per-thread local histograms
        for (uint32_t p = 0; p < HIST_SIZE; p++)
            histogram[p] = 0.0f;

        // TODO: shared memory caching of pins_per_partitions!!

        // scan touching hyperedges
        // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
        // NOTE: interpret this as "for each hedge, see if you moving to a certain partition is something that they like or not"
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            const uint32_t actual_hedge_idx = *hedge_idx;
            const uint32_t* my_pin_per_partition = pins_per_partitions + actual_hedge_idx * num_partitions;
            my_pin_per_partition += lane_id; // each thread in the warp reads one every WARP_SIZE counters
            const float my_hedge_weight = hedge_weights[actual_hedge_idx];
            for (uint32_t p = 0; p < HIST_SIZE; p += WARP_SIZE) {
                if (curr_base_part + p < num_partitions) {
                    // Option 1: the gain is the weighted connections count with another partition <minus> the weighted connections count with the current one
                    /*uint32_t counter = my_pin_per_partition[curr_base_part + p];
                    if (curr_base_part + p == my_partition)
                        counter--; // exclude your presence in the hyperedge from the count for your partition
                    // update local histogram
                    histogram[p] += counter*my_hedge_weight;*/
                    // Option 2: the gain is the sum the weights of hedges such that moving to a certain partition the hedge (same as HyperG)
                    uint32_t counter = my_pin_per_partition[curr_base_part + p];
                    float gain = 0.0f;
                    // hedge connected to my partition: gain the hedge's weight iff moving would disconnect it from my partition (I am its last pin left there)
                    if (counter == 1 && curr_base_part + p == my_partition)
                        gain = my_hedge_weight;
                    // hedge not yet connected to the partition: pay the hedge's weight iff moving there connects it to the new partition (I would become its first pin there)
                    else if (counter == 0 && curr_base_part + p != my_partition)
                        gain = -my_hedge_weight;
                    // update local histogram
                    histogram[p] += gain;
                }
            }
        }

        // reduce local histograms between threads (each thread will see the full histogram)
        // TODO: could use the stopping condition "p < HIST_SIZE && curr_base_part + p < num_partitions", but would lose unrolling...
        for (uint32_t p = 0; p < HIST_SIZE; p++)
            histogram[p] = warpReduceSum<float>(histogram[p]);

        // reduce max in histogram between threads (each thread grabs a different bin)
        for (uint32_t p = lane_id; p < HIST_SIZE; p += WARP_SIZE) {
            bin max = warpReduceMax(histogram[p], p);
            if (max.score > best_score || max.score == best_score && max.node < best_move) {
                best_score = max.score;
                best_move = max.node; // yeah, "node" should be called "partition" here, but this way we repurpose the struct...
            }
        }
    }

    if (lane_id == 0) {
        moves[warp_id] = best_move;
        scores[warp_id] = best_score; // write even if negative since scores are uninitialized, we filter later!
    }
}

// find the gain of each move under the HP that all higher-score moves have been applied
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
// SHUFFLES OVER: d
__global__
void fm_refinement_cascade_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    // CHOOSE: either rank nodes by their score and pass "move_ranks" or pass "scores" and sort on the fly, with the node id as a tie-breaker
    // CHOICE: sorted scores and move_ranks, because we need to keep scores in their current (sorted) order even after updating them
    const uint32_t* move_ranks, // move_ranks[node_idx] -> i (ranking by score) of the move proposed by the idx node
    const uint32_t* moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx (DO NOT SORT)
    const uint32_t* partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t* pins_per_partitions, // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    float* scores // scores[move_ranks[node_idx]] -> gain for node idx's move
) {
    // STYLE: one node (move) per warp!
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
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

    const uint32_t my_partition = partitions[warp_id];
    const uint32_t my_move_rank = move_ranks[warp_id];
    //const uint32_t my_move_score = scores[warp_id];
    const uint32_t my_move_part = moves[warp_id];

    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
    //uint32_t touching_count = touching_offsets[warp_id + 1] - touching_offsets[warp_id];

    // scan touching hyperedges
    // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        const uint32_t my_curr_part_counter = pins_per_partitions[actual_hedge_idx * num_partitions + my_partition];
        const uint32_t my_move_part_counter = pins_per_partitions[actual_hedge_idx * num_partitions + my_move_part];
        uint32_t my_curr_part_counter_delta = 0;
        uint32_t my_move_part_counter_delta = 0;
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            if (my_hedge < not_my_hedge) {
                uint32_t pin = *my_hedge;
                // Option 1: sorting scores to build and pass the ranking of moves
                if (move_ranks[pin] < my_move_rank) { // speculation: better-ranked move -> applied
                // Option 2: compare scores on the fly and tie-break with the node ids
                //if (scores[pin] > my_move_score || scores[pin] == my_move_score && pin < warp_id) { // speculation: better-ranked move -> applied
                // NOTE: MUST use option (1) because we need to keep the original sorting of the scores even after updating them!
                    uint32_t new_pin_partition = moves[pin];
                    uint32_t prev_pin_partition = moves[pin];
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
        }
        // Option 1: the gain is the weighted connections count with another partition <minus> the weighted connections count with the current one
        //score -= my_hedge_weight*(my_curr_part_counter + warpReduceSumLN0<uint32_t>(my_curr_part_counter_delta));
        //score += my_hedge_weight*(my_move_part_counter + warpReduceSumLN0<uint32_t>(my_move_part_counter_delta));
        // Option 2: the gain is the sum the weights of hedges such that moving to a certain partition the hedge (same as HyperG)
        // gain the hedge's weight iff moving would disconnect the hedge from my partition (I am its last pin left there)
        if (my_curr_part_counter + warpReduceSumLN0<uint32_t>(my_curr_part_counter_delta) == 1)
            score += my_hedge_weight;
        // pay the hedge's weight iff moving there connects the hedge to the new partition (I would become its first pin there)
        if (my_move_part_counter + warpReduceSumLN0<uint32_t>(my_move_part_counter_delta) == 0)
            score -= my_hedge_weight;
    }

    if (lane_id == 0 && score > 0)
        scores[my_move_rank] = score;
}

// apply moves with a positive gain
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
__global__
void fm_refinement_apply_kernel(
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t* moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx (DO NOT SORT)
    const uint32_t* move_ranks, // move_ranks[node_idx] -> i (ranking by score) of the move proposed by the idx node
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t num_good_moves, // idx + 1 of the maximum in the updated scores
    uint32_t* partitions // partitions[idx] is the partition node idx is part of
    //uint32_t* pins_per_partitions // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
) {
    // STYLE: one node (move) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    // stop at the last gain-increasing move
    if (move_ranks[tid] >= num_good_moves) return;

    //const uint32_t my_partition = partitions[tid];
    const uint32_t my_move_part = moves[tid];

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