#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "utils.cuh"

namespace cg = cooperative_groups;


// REMEMBER: "const" means the data pointed to is not modified, not the pointer itself!

// find the best neighbor for each node to stay with (edge-coarsening)
__global__
void candidates_kernel(
    const uint32_t* hedges,
    const uint32_t* hedge_offsets,
    const uint32_t* neighbors,
    const uint32_t* neighbors_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t bits_key_neg, // do the bitwise-and only when using node ids as indices!
    uint32_t* pairs, // do not put the (multi)function bits here!
    float* scores
) {
    // STYLE: one node per warp!
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // global across blocks - coincides with the node to handle
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;

    /*
    * Idea:
    * - one node per warp
    * - same part of the histogram in each thread's registers
    * - scan hyperedges (with caching in shared memory - either passive or automatic) once per histogram part
    * - warp primitives to both reduce each bin and find the maximum bin
    */

    const uint32_t* my_neighbors = neighbors + neighbors_offsets[warp_id];
    const uint32_t* not_my_neighbors = neighbors + neighbors_offsets[warp_id + 1];
    uint32_t neighbors_count = neighbors_offsets[warp_id + 1] - neighbors_offsets[warp_id];
    bin histogram[HIST_SIZE]; // make sure this fits in registers (no spill) !!

    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];

    // all threads in the warp should agree on those...
    float best_score = 0.0f;
    uint32_t best_neighbor = UINT32_MAX;

    // handle HIST_SIZE neighbors at a time
    for (; my_neighbors < not_my_neighbors; my_neighbors += HIST_SIZE) {
        // load the first HIST_SIZE neighbors and setup per-thread local histograms
        for (uint32_t nb = 0; nb < HIST_SIZE; nb++) {
            if (nb < neighbors_count) {
                uint32_t my_neighbor = my_neighbors[nb];
                // NOTE: neighbors and hedges use different (multi)function bits, so we must remove them before comparing!!
                histogram[nb].node = my_neighbor & bits_key_neg;
                if (my_neighbor == UINT32_MAX) // reached blank neighbors area left by coarsening
                    neighbors_count = 0;
            } else
                histogram[nb].node = UINT32_MAX;
            histogram[nb].score = 0.0f;
        }

        // TODO: shared memory caching of hyperedges!!

        // scan touching hyperedges
        // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            uint32_t actual_hedge_idx = *hedge_idx; // NOTE: no (multi)function bits used in "touching"
            const uint32_t* my_hedge = hedges + hedge_offsets[actual_hedge_idx];
            my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
            const uint32_t* not_my_hedge = hedges + hedge_offsets[actual_hedge_idx + 1];
            float my_hedge_weight = hedge_weights[actual_hedge_idx];
            for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
                uint32_t pin = UINT32_MAX - 1; 
                if (my_hedge < not_my_hedge)
                    pin = *my_hedge & bits_key_neg;
                // update local histogram
                for (uint32_t nb = 0; nb < HIST_SIZE; nb++) {
                    if (pin == histogram[nb].node)
                        histogram[nb].score += my_hedge_weight;
                }
            }
        }

        // reduce local histograms between threads (each thread will see the full histogram)
        for (uint32_t nb = 0; nb < HIST_SIZE; nb++)
            histogram[nb].score = warpReduceSum(histogram[nb].score);

        // reduce max in histogram between threads (each thread grabs a different bin)
        for (uint32_t nb = lane_id; nb < HIST_SIZE; nb += WARP_SIZE) {
            bin max = warpReduceMax(histogram[nb].score, histogram[nb].node);
            if (max.score > best_score) {
                best_score = max.score;
                best_neighbor = max.node;
            }
        }

        neighbors_count -= HIST_SIZE;
    }

    if (lane_id == 0) {
        pairs[warp_id] = best_neighbor; // (multi)function bits already removed while building the histogram
        scores[warp_id] = best_score;
    }
}

// create groups of at most MAX_GROUP_SIZE nodes, highest score first
// TODO: currently MAX_GROUP_SIZE is ignored, and groups are of at most 2!
__global__
void grouping_kernel(
    const uint32_t* pairs, // pairs[idx] is the partner idx wants to be grouped with (no (multi)function bits here)
    const float* scores, // pairs[idx] is the partner idx wants to be grouped with (no (multi)function bits here)
    const uint32_t num_nodes,
    //const uint32_t bits_key_neg, // no need, "pairs" don't have (multi)function bits!
    slot* group_slots // initialized with -1 on the id
) {
    // SETUP FOR GLOBAL SYNC
    cg::grid_group grid = cg::this_grid();

    // STYLE: one node per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    int32_t path_length = 0;
    uint32_t path[PATH_SIZE];

    uint32_t current = tid; // current node in the tree of pairs
    uint32_t target = pairs[tid]; // target node of "current"
    uint32_t target_target = pairs[target]; // target node of "target"
    uint32_t score = (uint32_t)(scores[tid]*FIXED_POINT_SCALE); // score with which "current" points to "target"

    /*
    * logic:
    * - MAYBE: this could be done with a (target-id,score)-key-based sort with thrust, but would be n*logn...
    * - each node has M free slots
    * - one thread per pair (node)
    * - [NAH one round per slot]
    * - slots made of float+uint32 tuples being score+pair
    * - big HP: by construction, a node will always propose a pair with score equal or higher than that of pairs
    *   formulated by others towards him. That is because each node already went over its entire neighborhood and found
    *   the best candidate, others will all be second choices. So either the node succeeds with its target, or it accepts
    *   the best of the second choices -> don't protect writes to yourself (no deadlocks - just lock the slot of the target
    *   node where you write), simply overwrite them if you succeed. After every thread has written, go see if what you
    *   wrote is still there, if so, write to yourself the same pairing. Issue: cascade effect where if this check happens
    *   in parallel, since each thread can only decide (check the "still there") after his target decided what to do.
    *   - upgrade: the M slots of each node are sorted by score, when writing there, lock the node, and insert yourself,
    *     pushing out the lowest score entry (even if that is yourself). If, after the global synch, a node sees that its
    *     write was pushed out of the target's slots, then do nothing, otherwise, if it is still there, insert a the top
    *     of your slots your target, pushing out one of the others...cascade effect...
    *   - cascade effect solution: (1) global synch, (2) everyone checks and updates if needed, (3) if an updated happens,
    *     write a 1 (unprotected - initialized at 0) to a global variable, (4) repeat so long as the variable gets to 1
    * - practically, every thread:
    *   - if its score is higher than what is in its target's slot, atomically updates it with its own
    *   - otherwise do nothing (just accept incoming writes)
    *   - repeat after a global synch:
    *     - if you wrote something, go check if it's still there, if so, write to yourself the pair's counterpart
    * Note: the "big HP" leaves no room for cycles (longer than two - pairs pointing one to the other)! The "pairs" build a tree!
    */

    /*
    * NEW LOGIC: a walk up (and down) the tree for grouping!
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
    */

    // go up the tree
    while (target != UINT32_MAX && current != target_target) {
        // TODO: optimization, if the atomic fails (you are already not the maximum), you can stop here and not even continue!
        atomic_max_on_slot(group_slots, target, current, score);
        path[path_length++] = target;
        current = target;
        target = target_target;
        target_target = pairs[target_target]; // if this goes to -1, stop after the next iteration, as to still handle the current "target" that will be the "root"
        score = (uint32_t)(scores[current]*FIXED_POINT_SCALE);
    }
    // handle "root(s)" as a pair of nodes pointing to each other
    // NOTE: concurrent writes are fine, anyone seing those two nodes will write the same thing: "this is a group" or "would you get married already!?"
    if (target != UINT32_MAX) {
        uint32_t lowest_id = min(current, target);
        group_slots[current].id = lowest_id;
        group_slots[current].score = score;
        group_slots[target].id = lowest_id;
        group_slots[target].score = score;
    }

    // global synch
    grid.sync();

    // go down the tree
    // it the root had no target, path[path_length - 1] is the root, if the root was a mutual-poiting pair, path[path_length - 1] is the first encountered node of the pair
    current = path[path_length - 2]; // NOTE: possible opt. since now "current" is already "== path[path_length - 1]"...
    target = path[path_length - 1]; // this is "who current would have liked to be with", the node one step back up the ladder towards root
    for (path_length = path_length - 3; path_length >= 0; path_length--) {
        if (group_slots[target].id == current) {
            // TODO: maybe writing only after reading and checking if someone else already wrote can spare cache coherence chaos - concurrent writes are still fine
            group_slots[current].id = current;
            // NOTE: do we even care about the score now?
            //group_slots[current].score = group_slots[target].id;
            // NOTE: after a successful link the next node down the path has already lost its target, so we might as well skip it
            current == path[path_length--];
            if (path_length < 0) break;
        }
        target = current;
        current = path[path_length];
    }
    // as of now, current is the last element of the path (that did not include the "tid" node)
    target = current;
    current = tid;
    if (group_slots[target].id == current) {
        group_slots[current].id = current;
        //group_slots[current].score = group_slots[target].id;
    }
}

__global__
void apply_coarsening_hedges(
    const uint32_t num_hedges,
    const uint32_t* hedge_offsets,
    const uint32_t* groups, // node -> new node id (with (multi)function bits set of one-hot encoding per node in the group)
    const uint32_t bits_key,
    const uint32_t bits_key_neg, // do the bitwise-and only when using node ids as indices!
    uint32_t* hedges // MUST BITWISE-OR the (multi)function bits while deduplicating, and then write them
) {
    // STYLE: one hedge per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /*
    * Idea:
    * - never resize "hedges" and never alter "hedge_offsets"
    * - read hyperedges, deduplicate (source included - start form it tho, as to never remove it) in local memory
    *   (runtime array allocation of the hedge's size), update node ids, then overwrite the hyperedge, pad with -1s
    * - TODO: if an hedge gets to zero destinations, leave it be for now...eventually find a way to skip it
    * - TODO: accesses are currently not coalesced per warp...
    */

    uint32_t hedge_start_idx = hedge_offsets[tid], hedge_end_idx = hedge_offsets[tid + 1];
    uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    uint32_t size = hedge_end_idx - hedge_start_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t hedge[MAX_DEDUPE_BUFFER_SIZE];

    for (uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        uint32_t pin = groups[*curr]; // read and map pin to its new id
        bool not_seen = true;
        // NOTE: slightly slow sequential scan, fine for cardinalities ~200
        for (uint32_t i = 0; i < distinct; i++) {
            uint32_t entry = hedge[i];
            if (entry & bits_key_neg == pin & bits_key_neg) {
                not_seen = false;
                hedge[i] = (entry & bits_key_neg) | (entry & bits_key | pin & bits_key); // OR of the (multi)function bits
                break;
            }
        }
        if (not_seen)
            hedge[distinct++] = pin; // initialize the (multi)function bits too
    }

    // NOTE: slightly inefficient, as we write "size" instead of "distinct" elements...
    for (uint32_t i = 0; i < size; i++) {
        if (i < distinct)
            hedges[hedge_start_idx + i] = hedge[i];
        else
            hedges[hedge_start_idx + i] = UINT32_MAX;
    }
}

__global__
void apply_coarsening_neighbors(
    const uint32_t num_nodes,
    const uint32_t* neighbor_offsets,
    const uint32_t* groups, // node -> new node id (with (multi)function bits already inserted)
    const uint32_t bits_key,
    const uint32_t bits_key_neg, // do the bitwise-and only when using node ids as indices!
    uint32_t* neighbors // MUST BITWISE-OR the (multi)function bits while deduplicating, and then write them
) {
    // STYLE: one node (neighbor-set) per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    /*
    * Idea:
    * - never resize "neighbors" and never alter "neighbor_offsets"
    * - read neighbors, deduplicate in local memory (runtime array allocation of the neighbor-set's size),
    *   update node ids, then overwrite the set, pad with -1s
    * - TODO: if an neighbor gets to zero destinations, leave it be for now...eventually find a way to skip it
    * - TODO: accesses are currently not coalesced per warp...
    */

    uint32_t neighbor_start_idx = neighbor_offsets[tid], neighbor_end_idx = neighbor_offsets[tid + 1];
    uint32_t *neighbor_start = neighbors + neighbor_start_idx, *neighbor_end = neighbors + neighbor_end_idx;
    uint32_t size = neighbor_end_idx - neighbor_start_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t neighbor[MAX_DEDUPE_BUFFER_SIZE];

    for (uint32_t* curr = neighbor_start; curr < neighbor_end; curr++) {
        uint32_t pin = groups[*curr]; // read and map pin to its new id
        bool not_seen = true;
        // NOTE: slightly slow sequential scan, fine for cardinalities ~200
        for (uint32_t i = 0; i < distinct; i++) {
            uint32_t entry = neighbor[i];
            if (entry & bits_key_neg == pin & bits_key_neg) {
                not_seen = false;
                neighbor[i] = (entry & bits_key_neg) | (entry & bits_key | pin & bits_key); // OR of the (multi)function bits
                break;
            }
        }
        if (not_seen)
            neighbor[distinct++] = pin; // initialize the (multi)function bits too
    }

    // NOTE: slightly inefficient, as we write "size" instead of "distinct" elements...
    for (uint32_t i = 0; i < size; i++) {
        if (i < distinct)
            neighbors[neighbor_start_idx + i] = neighbor[i];
        else
            neighbors[neighbor_start_idx + i] = UINT32_MAX;
    }
}

__global__
void apply_coarsening_touching(
    const uint32_t num_nodes,
    const uint32_t* groups,
    uint32_t* touching,
    uint32_t* touching_offsets
) {
    // STYLE: one node per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    /*
    * Idea:
    * - big HP: if two nodes are grouped, always at least one of their touching hyperedges in commmon!
    * - when deduplicating, that hyperedge will be removed, and we can repurpose its location to insert a pointer
    *   from the lower-idx touching set to the starting idx of the next set of the next node in the group (ordered
    *   by increasing node id), creating a fragmented linked list!
    *
    * ISSUE: how can this be inverted once you uncoarsen? We don't have (multi)function bits here, nor we have
    *        a map on hyperedges where to build them...
    * => we must resort to duplicating the "touching" data at every level...
    */
}