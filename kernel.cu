#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "utils.cuh"

namespace cg = cooperative_groups;

// DEVICE CONSTANTS:
__constant__ uint32_t max_nodes_per_part;
__constant__ uint32_t max_inbound_per_part;

// REMEMBER: "const" means the data pointed to is not modified, not the pointer itself!

// TODO: upgrade "uint_32_t"s to "uint64_t"s to handle >4M nodes, >4M touching per node, >4M pins per hedge, >4M neighbors per node!

// compute the distinct neighbors count of each node, aided by a global hash-set
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
__global__
void neighborhoods_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    const dim_t max_neighbors,
    uint32_t* __restrict__ neighbors,
    dim_t* __restrict__ neighbors_offsets // here filled as counters of "how many neighbors per node" -> then do a prefix sum for the offsets
) {
    // STYLE: one node per block, one touching hedge per warp, distinct neighbors in shared memory!
    const uint32_t node_id = blockIdx.x;
    // NOTE: the whole block returns, no need to sync
    if (node_id >= num_nodes) return;

    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // local per block - coincides with the touching hedge to handle
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const uint32_t* my_touching = touching + touching_offsets[node_id];
    //const uint32_t* not_my_touching = touching + touching_offsets[node_id + 1];
    const uint32_t my_touching_count = (uint32_t)(touching_offsets[node_id + 1] - touching_offsets[node_id]);

    // no touching hedges, return immediately
    // NOTE: the whole block returns, no need to sync
    if (my_touching_count == 0) {
        if (threadIdx.x == 0)
            neighbors_offsets[node_id] = 0;
        return;
    }

    uint32_t* my_neighbors = neighbors + node_id * max_neighbors;

    // hash-set for deduplication (allows false-negatives, back it up with true deduplication in global memory)
    __shared__ uint32_t dedupe[SM_MAX_DEDUPE_BUFFER_SIZE];
    // HP: each node has less than UINT32_MAX neighbors
    __shared__ uint32_t seen_distinct_total;
    uint32_t seen_distinct = 0;
    
    // initialize shared memory
    blk_init<uint32_t>(dedupe, SM_MAX_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    blk_init<uint32_t>(my_neighbors, max_neighbors, HASH_EMPTY);
    if (threadIdx.x == 0)
        seen_distinct_total = 0;
    __syncthreads();

    // can return only after helping initializing shared memory
    if (warp_id >= my_touching_count) return;

    // TODO: could optimize by iterating directly on pointers
    for (uint32_t touching_hedge_idx = warp_id; touching_hedge_idx < my_touching_count; touching_hedge_idx += warps_per_block) { // the block loops over touching hedges
        if (touching_hedge_idx < my_touching_count) {
            const uint32_t my_hedge_idx = my_touching[touching_hedge_idx];
            const uint32_t* my_hedge = hedges + hedges_offsets[my_hedge_idx];
            //const uint32_t* not_my_hedge = hedges + hedges_offsets[my_hedge_idx + 1];
            const uint32_t my_hedge_size = (uint32_t)(hedges_offsets[my_hedge_idx + 1] - hedges_offsets[my_hedge_idx]);
            for (uint32_t node_idx = lane_id; node_idx < my_hedge_size; node_idx += WARP_SIZE) { // the warp loops over hedge pins
                if (node_idx < my_hedge_size) {
                    uint32_t neighbor = my_hedge[node_idx];
                    uint8_t inserted = sm_hashset_try_insert(dedupe, SM_MAX_DEDUPE_BUFFER_SIZE, neighbor);
                    if (neighbor != node_id && inserted) {
                        if (inserted == 2) { // no need to put into GM what already is in the SM hash-set
                            if(gm_hashset_insert(my_neighbors, max_neighbors, neighbor)) // triggers an assert if 'max_neighbors' is exceeded
                                seen_distinct++;
                        } else
                            seen_distinct++;
                    }
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
        neighbors_offsets[node_id] = (dim_t)seen_distinct_total;
}

// compute the distinct neighbors of each node (again) and write them at the pre-computed offsets in global memory (handled as a hash-set)
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
__global__
void neighborhoods_scatter_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    const dim_t* __restrict__ neighbors_offsets,
    uint32_t* __restrict__ neighbors
) {
    // STYLE: one node per block, one touching hedge per warp, distinct neighbors in shared memory!
    const uint32_t node_id = blockIdx.x;
    if (node_id >= num_nodes) return;

    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // local per block - coincides with the touching hedge to handle
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const uint32_t* my_touching = touching + touching_offsets[node_id];
    //const uint32_t* not_my_touching = touching + touching_offsets[node_id + 1];
    const uint32_t my_touching_count = (uint32_t)(touching_offsets[node_id + 1] - touching_offsets[node_id]);

    uint32_t* my_neighbors = neighbors + neighbors_offsets[node_id];
    const uint32_t my_neighbors_count = (uint32_t)(neighbors_offsets[node_id + 1] - neighbors_offsets[node_id]);

    // hash-set for deduplication
    __shared__ uint32_t dedupe[SM_MAX_DEDUPE_BUFFER_SIZE];
    
    // initialize shared memory
    blk_init<uint32_t>(dedupe, SM_MAX_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    blk_init<uint32_t>(my_neighbors, my_neighbors_count, HASH_EMPTY);
    __syncthreads();

    // can return only after helping initializing shared memory
    if (warp_id >= my_touching_count) return;

    // TODO: could optimize by iterating directly on pointers
    for (uint32_t touching_hedge_idx = warp_id; touching_hedge_idx < my_touching_count; touching_hedge_idx += warps_per_block) { // the block loops over touching hedges
        if (touching_hedge_idx < my_touching_count) {
            const uint32_t my_hedge_idx = my_touching[touching_hedge_idx];
            const uint32_t* my_hedge = hedges + hedges_offsets[my_hedge_idx];
            //const uint32_t* not_my_hedge = hedges + hedges_offsets[my_hedge_idx + 1];
            const uint32_t my_hedge_size = (uint32_t)(hedges_offsets[my_hedge_idx + 1] - hedges_offsets[my_hedge_idx]);
            for (uint32_t node_idx = lane_id; node_idx < my_hedge_size; node_idx += WARP_SIZE) { // the warp loops over hedge pins
                if (node_idx < my_hedge_size) {
                    uint32_t neighbor = my_hedge[node_idx];
                    if (neighbor != node_id && sm_hashset_try_insert(dedupe, SM_MAX_DEDUPE_BUFFER_SIZE, neighbor))
                        gm_hashset_insert(my_neighbors, my_neighbors_count, neighbor); // should never happen, but could trigger an assert if 'my_neighbors_count' is exceeded
                }
            }
        }
    }
}

// find the best neighbor for each node to stay with (edge-coarsening)
// SEQUENTIAL COMPLEXITY: n*h*d + n*(# neighbors)*h
// PARALLEL OVER: n
// SHUFFLES OVER: h*d (neighbors)
__global__
void candidates_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
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
    uint32_t* histogram_node = histogram + 2 * HIST_SIZE * (threadIdx.x / WARP_SIZE);
    uint32_t* histogram_score = histogram + 2 * HIST_SIZE * (threadIdx.x / WARP_SIZE) + HIST_SIZE;

    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
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
        for (uint32_t nb = lane_id; nb < HIST_SIZE; nb += WARP_SIZE) {
            uint32_t curr_neighbor = UINT32_MAX;
            if (nb < neighbors_count) {
                curr_neighbor = my_neighbors[nb];
                 // skip incompatible neighbors due to size constraints
                if (my_size + nodes_sizes[curr_neighbor] <= max_nodes_per_part)
                    histogram_node[nb] = curr_neighbor;
                else
                    histogram_node[nb] = UINT32_MAX;
            } else
                histogram_node[nb] = UINT32_MAX;
        }
        // warp sync after filling histograms
        __syncwarp();

        // sort the histogram by node-id to then rely on binary search
        wrp_bitonic_sort<uint32_t, HIST_SIZE>(histogram_node);
        __syncwarp();

        // add a little bit of symmetric deterministic noise
        for (uint32_t nb = lane_id; nb < HIST_SIZE; nb += WARP_SIZE) {
            uint32_t curr_neighbor = histogram_node[nb];
            if (curr_neighbor != UINT32_MAX)
                histogram_score[nb] = deterministic_noise(curr_neighbor, warp_id);
            else
                histogram_score[nb] = 0u;
        }

        // TODO: shared memory caching of hyperedges!!

        // TODO: add this back when and if you start filling back up invalid nodes before going to the next histogram...
        // if no neighbor passed constraints checks (histogram is all UINT32_MAXs), exit the loop early
        //if (neighbors_count == 0)
        //    break;

        // iterate over touching hyperedges
        // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            dim_t my_hedge_offset, not_my_hedge_offset;
            uint32_t my_hedge_weight;
            if (lane_id == 0) {
                const uint32_t actual_hedge_idx = *hedge_idx;
                my_hedge_offset = hedges_offsets[actual_hedge_idx];
                not_my_hedge_offset = hedges_offsets[actual_hedge_idx + 1];
                my_hedge_weight = (uint32_t)(hedge_weights[actual_hedge_idx]*FIXED_POINT_SCALE);
            }
            const uint32_t* my_hedge = lane_id + hedges + __shfl_sync(0xFFFFFFFF, my_hedge_offset, 0); // each thread in the warp reads one every WARP_SIZE pins
            const uint32_t* not_my_hedge = hedges + __shfl_sync(0xFFFFFFFF, not_my_hedge_offset, 0);
            my_hedge_weight = __shfl_sync(0xFFFFFFFF, my_hedge_weight, 0);
            for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
                uint32_t pin = UINT32_MAX - 1;
                if (my_hedge < not_my_hedge) pin = *my_hedge;
                // update local histogram
                const uint32_t hist_idx = binary_search<uint32_t>(histogram_node, HIST_SIZE, pin);
                // NOTE: no atomic needed => all threads in the warp advance in sync and there are not duplicates in hedges, no two threads will ever write the same bin!
                if (hist_idx != UINT32_MAX) histogram_score[hist_idx] += my_hedge_weight;
            }
        }
        // warp sync after computing scores
        __syncwarp();

        // sort the histogram by lowest score first (empty bins have score = 0), highest id first as a tiebreaker
        // TODO: maybe replace with a warp-parallel max per candidate iteration below (dunno tho, the max requires a warp-parallel read of the whole histogram)
        wrp_bitonic_sort_by_key<uint32_t, uint32_t, HIST_SIZE>(histogram_score, histogram_node);
        __syncwarp();

        // updated global maximum(s), checking inbound constraints before doing it
        // => this postpones such an expensive constrain check as much as possible, and does it for as few neighbors as possible
        for (int32_t candidate = HIST_SIZE - 1; candidate >= 0; candidate--) {
            uint32_t curr_neighbor;
            if (lane_id == 0)
                curr_neighbor = histogram_node[candidate];
            curr_neighbor = __shfl_sync(0xFFFFFFFF, curr_neighbor, 0);
            if (curr_neighbor == UINT32_MAX) break;
            uint32_t curr_score, neighbor_inbound_count;
            dim_t neighbor_touching_offsets;
            if (lane_id == 0) {
                curr_score = histogram_score[candidate];
                neighbor_touching_offsets = touching_offsets[curr_neighbor];
                neighbor_inbound_count = inbound_count[curr_neighbor];
            }
            curr_score = __shfl_sync(0xFFFFFFFF, curr_score, 0);
            if (curr_score < best_score[MAX_CANDIDATES - 1]) break;

            // skip incompatible neighbors due to inbound constraints
            const uint32_t* neighbor_inbound = touching + __shfl_sync(0xFFFFFFFF, neighbor_touching_offsets, 0);
            const uint32_t* not_neighbor_inbound = neighbor_inbound + __shfl_sync(0xFFFFFFFF, neighbor_inbound_count, 0);
            uint32_t new_inbound_count = 0u;
            for (neighbor_inbound += lane_id; neighbor_inbound < not_neighbor_inbound; neighbor_inbound += WARP_SIZE)
                new_inbound_count += binary_search<uint32_t>(my_touching, my_inbound_count, *neighbor_inbound) == UINT32_MAX ? 1 : 0; // binary search only among the inbounds part of my_touching
            new_inbound_count = warpReduceSum<uint32_t>(new_inbound_count) + my_inbound_count;
            if (new_inbound_count > max_inbound_per_part) continue;

            // get the best MAX_CANDIDATES candidates out of the histogram
            for (uint32_t i = 0; i < MAX_CANDIDATES; i++) {
                // tie-breaker: lower id node wins; invariant: partial neighbors order
                if (curr_score > best_score[i] || curr_score == best_score[i] && curr_neighbor < best_neighbor[i]) {
                    for (uint32_t j = MAX_CANDIDATES - 1; j > i; j--) {
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
        for (uint32_t i = 0; i < MAX_CANDIDATES; i++) {
            pairs[warp_id * MAX_CANDIDATES + i] = best_neighbor[i];
            scores[warp_id * MAX_CANDIDATES + i] = best_score[i]; // stay fixed point for now!
        }
    }
}

// create groups of at most MAX_GROUP_SIZE nodes, highest score first
// TODO: currently MAX_GROUP_SIZE is ignored, and groups are of at most 2!
// SEQUENTIAL COMPLEXITY: n*log n
// PARALLEL OVER: n
__global__
void grouping_kernel(
    const uint32_t* __restrict__ pairs, // pairs[idx] is the partner idx wants to be grouped with
    const uint32_t* __restrict__ scores, // scores[idx] is the strenght with which idx wants to be grouped with pairs[idx]
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_nodes,
    const uint32_t num_repeats,
    slot* __restrict__ group_slots, // initialized with -1 on the id
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
    */

    int32_t path_length[MAX_REPEATS];
    uint32_t path[PATH_SIZE];
    const uint32_t actual_path_size = PATH_SIZE / num_repeats;
    uint32_t completed_repeats = 0u;

    for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
        const uint32_t curr_tid = tid + repeat * tcount;
        // initialize yourself to a one-node group, unless someone already claimed you
        if (curr_tid < num_nodes) atomicCAS(&reinterpret_cast<unsigned long long*>(group_slots)[curr_tid*MAX_GROUP_SIZE], pack_slot(0u, UINT32_MAX), pack_slot(0u, curr_tid));
    }

    for (uint32_t i = 0; i < MAX_CANDIDATES; i++) {
        // repeat for every node that you need to handle
        for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
            const uint32_t curr_tid = tid + repeat * tcount;

            // if you are not a valid node
            if (curr_tid >= num_nodes) break;
            // if you formed a pair, stop
            if (completed_repeats & (1u << repeat)) continue;
            if (group_slots[curr_tid*MAX_GROUP_SIZE].score == UINT32_MAX) {
                completed_repeats |= (1u << repeat);
                continue;
            }

            uint32_t* curr_path = path + actual_path_size * repeat;
            int32_t curr_path_length = 0;

            uint32_t current = curr_tid; // current node in the tree of pairs
            uint32_t target = pairs[curr_tid * MAX_CANDIDATES + i]; // target node of "current"
            uint32_t score = scores[curr_tid * MAX_CANDIDATES + i]; // score with which "current" points to "target"

            // go up the tree
            bool outcome = false;
            if (target != UINT32_MAX) {
                outcome = true;
                uint32_t target_target = pairs[target * MAX_CANDIDATES + i]; // target node of "target"
                while (current != target_target) {
                    // NOTE: if, by bad luck, the fixed-point score is the same, the id is used as a tie-breaker
                    outcome = atomic_max_on_slot(group_slots, target*MAX_GROUP_SIZE, current, score);
                    // NOTE: if the atomic fails you are surely not the maximum, thus you can stop here and not even bother continuing (we then start going down without even looking again at this target)
                    // NOTE: if the atomic already sees the value you were about to write (strict), you CANNOT stop, you must know what happens to your target before taking the final decision on your path
                    // => with the second note being a <cannot>, we can't guarantee that exactly one thread will go up any piece of path, but we still don't need atomics in the downward walk, as every thread takes the same decisions regardless
                    if (!outcome) break;
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
                    target_target = pairs[target_target * MAX_CANDIDATES + i]; // if this goes to -1, stop after the next iteration, as to still handle the current "target" that will be the "root"
                    score = scores[current * MAX_CANDIDATES + i];
                }
            }
            // handle "root(s)" as a pair of nodes pointing to each other
            // NOTE: forceful writes are fine, anyone seing those two nodes will write the same thing: "this is a group" or "would you get married already!?"
            // NOTE: prevent locked groups from forming a 2-cycle after the next pairing candidates are considered
            if (outcome && target != UINT32_MAX && group_slots[target*MAX_GROUP_SIZE].score != UINT32_MAX) {
                const uint32_t lowest_id = min(current, target);
                set_slot(group_slots, current*MAX_GROUP_SIZE, lowest_id, UINT32_MAX);
                set_slot(group_slots, target*MAX_GROUP_SIZE, lowest_id, UINT32_MAX);
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
                if (group_slots[target*MAX_GROUP_SIZE].id == current) {
                    // TODO: maybe writing only after reading and checking if someone else already wrote can spare cache coherence chaos - concurrent writes are still fine
                    group_slots[current*MAX_GROUP_SIZE].id = current;
                    // NOTE: lock groups by setting scores to the maximum
                    group_slots[current*MAX_GROUP_SIZE].score = UINT32_MAX;
                    group_slots[target*MAX_GROUP_SIZE].score = UINT32_MAX;
                    // NOTE: after a successful link the next node down the path has already lost its target, so we might as well skip it
                    curr_path_length--;
                }
            }
            // the path did not include the "curr_tid" node, handle it here iff you did not happen to group path[0] with path[1] during the last iteration above (leading to path_length == -2)
            if (curr_path_length == -1) {
                target = curr_path[0];
                current = curr_tid;
                if (group_slots[target*MAX_GROUP_SIZE].id == current) {
                    group_slots[current*MAX_GROUP_SIZE].id = current;
                    // lock groups by setting scores to the maximum
                    group_slots[current*MAX_GROUP_SIZE].score = UINT32_MAX;
                    group_slots[target*MAX_GROUP_SIZE].score = UINT32_MAX;
                }
            }
        }
        
        // global synch
        grid.sync();

        // TODO: anyone not yet selected (score != UINT32_MAX), set your score to 0 to enable the next pairing round, then sync again and continue!
    }

    // write inside "groups" the minimum id among each node's slots, used to identify its group, eventually, zero-base those ids
    for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
        const uint32_t curr_tid = tid + repeat * tcount;
        if (curr_tid >= num_nodes) break;
        uint32_t min_id = UINT32_MAX;
        for (uint32_t i = 0; i < MAX_GROUP_SIZE; i++)
            min_id = min(group_slots[curr_tid*MAX_GROUP_SIZE + i].id, min_id);
        groups[curr_tid] = min_id;
    }
}

// count how many distinct new pins are in each hedge
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void apply_coarsening_hedges_count(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    const uint32_t* __restrict__ groups, // groups[node idx] -> new group/node id
    dim_t* __restrict__ coarse_hedges_offsets // coarse_hedges_offsets[hedge idx] -> count of distinct groups among its pins
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /*
    * Idea:
    * - first deduplicate inside hedges to count their new, distinct, nodes
    * - scan the counts per hedge to get the new offsets
    * - repeat the deduplication and write (scatter) each new hedge to its offset
    *
    * TODO, upgrade options:
    * - instead of three kernels to count distinct nodes, scan offsets, and scatter distinct nodes, can't we do all in one?
    * - one hedge per warp or block, use the shared memory hash-map to keep a "first-come-first-served" set of counters in shared memory,
    *   and reditect additional entries to atomically incremented global counters (issue: musre ensure the src is preserved!!)
    *
    * Must ensure that:
    * - there are no self-cycles
    * - the same node never appears twice in the same hedge
    *   => exploited by the coarsening of touching sets
    */

    const dim_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t pins[MAX_DEDUPE_BUFFER_SIZE];
    // TODO: maybe "memset(pins, 0xFF, sizeof(pins))" is faster?
    thr_init<uint32_t>(pins, MAX_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t pin = groups[*curr]; // read and map pin to its new id
        bool not_seen = lm_hashset_insert(pins, MAX_DEDUPE_BUFFER_SIZE, pin);
        if (not_seen) {
            distinct++;
        }
    }

    // leave the first entry to be 0 (offset of the first hedge)
    coarse_hedges_offsets[tid + 1] = (dim_t)distinct;
}

// write the new distinct pins of hedges
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void apply_coarsening_hedges_scatter(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    const uint32_t* __restrict__ groups, // groups[node idx] -> new group/node id
    const dim_t* __restrict__ coarse_hedges_offsets, // coarse_hedges_offsets[hedge idx] -> count of distinct groups among its pins
    uint32_t* __restrict__ coarse_hedges
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /* Idea: second part of "apply_coarsening_hedges_count" -> scatter
    * NOTE: must ensure the source remains the first node in each coarse hedge
    *
    * Important: this currently deletes self-cycles -> between src and dst, it keeps the SRC when deduplicating!
    * => therefore, this is the opposite of coarsening 'touching', where 'touching' preserves the inbound set and
    *    deduplicates the outbound, hedges preserve the notion of outbound! The two are complementary!
    */

    const dim_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    const dim_t coarse_hedge_start_idx = coarse_hedges_offsets[tid];
    const uint32_t coarse_hedge_size = (uint32_t)(coarse_hedges_offsets[tid + 1] - coarse_hedges_offsets[tid]);
    uint32_t *coarse_hedge_start = coarse_hedges + coarse_hedge_start_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t pins[MAX_DEDUPE_BUFFER_SIZE];
    thr_init<uint32_t>(pins, MAX_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) { // first pin is the src, always preserved
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
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ groups, // groups[node id] -> node's group id
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    dim_t* __restrict__ coarse_neighbors_offsets // group id -> count of distinct neighbors
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

    const dim_t ungroups_start_idx = ungroups_offsets[tid], ungroups_end_idx = ungroups_offsets[tid + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t new_neighbors[MAX_LARGE_DEDUPE_BUFFER_SIZE];
    thr_init<uint32_t>(new_neighbors, MAX_LARGE_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

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
    coarse_neighbors_offsets[tid + 1] = (dim_t)distinct;
}

// write distinct neighbors
// SEQUENTIAL COMPLEXITY: n*d*h (in reality there are <<d*h neighbors per node)
// PARALLEL OVER: n
__global__
void apply_coarsening_neighbors_scatter(
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ groups, // groups[node id] -> node's group id
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const dim_t* __restrict__ coarse_neighbors_offsets, // group id -> count of distinct neighbors
    uint32_t* __restrict__ coarse_neighbors
) {
    // STYLE: one group (new node) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_groups) return;

    // Idea: second part of "apply_coarsening_neighbors_count" -> scatter

    const dim_t ungroups_start_idx = ungroups_offsets[tid], ungroups_end_idx = ungroups_offsets[tid + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    const dim_t coarse_neighbors_start_idx = coarse_neighbors_offsets[tid];
    const uint32_t coarse_neighbors_size = (uint32_t)(coarse_neighbors_offsets[tid + 1] - coarse_neighbors_offsets[tid]);
    uint32_t *coarse_neighbors_start = coarse_neighbors + coarse_neighbors_start_idx;
    uint32_t distinct = 0;
    // TODO: this is just a very large buffer for now (a part in registers, a part in cache) -> replace with small buffer + large spill buffer
    uint32_t new_neighbors[MAX_LARGE_DEDUPE_BUFFER_SIZE];
    thr_init<uint32_t>(new_neighbors, MAX_LARGE_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

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
    const uint32_t* __restrict__ hedges, // already coarsened as of here, thus contain group ids!
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    dim_t* __restrict__ coarse_touching_offsets // here filled as counters of "how many hedges per node" -> then do a prefix sum for the offsets
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /*
    * Idea:
    * - first coarsen hedge, then going over coarse hedge and counting the occurrencies of each group (new node) should be
    *   faster than just deduplicating touching hedges between nodes in the same group
    * - scan the touching counts per group (new node) to get the new offsets
    * - the scatter operates in a different way, assigning one thread per group, the thread goes over the nodes
    *   in the group via the "ungroups" and "ungroups_offsets" structures, deduplicates hedges and writes them in the touching set
    *
    * NOTE: this exploits the fact that the same pin never appears twice in the same hedge!
    *
    * TODO, upgrade options:
    * - for now we just do atomics towards global memory, because nodes are too many for shared memory, eventually:
    *   => use the shared memory hash-map to keep a "first-come-first-served" set of counters in shared memory,
    *     and reditect additional entries to global counters, then atomically increment global memory
    *
    * TODO: could partially skip this counting step, because we already have the new distinct inbound count evaluated during the candidates kernel!
    */

    const dim_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;

    // TODO: initialize shared memory
    // in utils.cuh : #define SM_MAX_HASHMAP_SIZE 4096u ?
    //__shared__ hashmap_entry hashmap[SM_MAX_HASHMAP_SIZE];
    //blk_init<uint32_t>(hashmap, SM_MAX_HASHMAP_SIZE*2, HASH_EMPTY);

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t pin = *curr; // already a group id
        atomicAdd(&coarse_touching_offsets[pin + 1], 1); // leave the first entry to be 0 (offset of the first set)
    }
}

// write distinct inbound touching hedges
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
// SHUFFLES OVER: h (touching)
__global__
void apply_coarsening_touching_scatter_inbound(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const dim_t* __restrict__ coarse_touching_offsets, // group id -> count of distinct touching hedges
    uint32_t* __restrict__ coarse_touching,
    uint32_t* __restrict__ coarse_inbound_count
) {
    // STYLE: one group (new node) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the group to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_groups) return;

    /*
    * Idea: second part of "apply_coarsening_touching_scatter" -> scatter inbound
    * Alternative version: one thread per hedge and dedupe inside each touching set (by linearly going over it, with a CAS at the end)
    * 
    * Must ensure that:
    * - inbound hedges are at the start of the set
    * - inbound hedges are sorted by id
    *
    * TODO upgrade option:
    * - one warp per node is fine with many nodes, but when nodes are few switch to one node per block!
    */

    /*
    * Important: here hedges that are both inbound and outbound are lost on one side, the outbound one specifically.
    *            In other words, and hedge both entering and leaving a group will only be remembered as an inbound one,
    *            this is because the deduplication is done for all touching at once, while the inbound count is incremented
    *            only for the first occurrence, that is, the inbound one!
    */

    const dim_t ungroups_start_idx = ungroups_offsets[warp_id], ungroups_end_idx = ungroups_offsets[warp_id + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    const dim_t coarse_touching_start_idx = coarse_touching_offsets[warp_id];
    const uint32_t coarse_touching_size = (uint32_t)(coarse_touching_offsets[warp_id + 1] - coarse_touching_offsets[warp_id]);
    uint32_t *coarse_touching_start = coarse_touching + coarse_touching_start_idx;
    uint32_t new_inbound_count = 0u;
    wrp_init<uint32_t>(coarse_touching_start, coarse_touching_size, HASH_EMPTY); // HP: HASH_EMPTY is > than any value that could be inserted

    // SM deduplication hash-set
    extern __shared__ uint32_t block_new_touching[];
    uint32_t *new_touching = block_new_touching + MAX_SM_DEDUPE_BUFFER_SIZE * (threadIdx.x / WARP_SIZE);
    wrp_init<uint32_t>(new_touching, MAX_SM_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    __syncwarp();

    // for every original node in the group, go over its inbound set
    for (const uint32_t* ungroup = ungroups_start; ungroup < ungroups_end; ungroup++) {
        const uint32_t node = *ungroup;
        const uint32_t* my_inbound = touching + touching_offsets[node];
        const uint32_t my_inbound_count = inbound_count[node];
        for (uint32_t i = lane_id; i < my_inbound_count; i += WARP_SIZE) {
            if (i < my_inbound_count) {
                const uint32_t hedge_idx = my_inbound[i];
                if (sm_hashset_try_insert(new_touching, MAX_SM_DEDUPE_BUFFER_SIZE, hedge_idx)) { // check in the SM cache
                    if (gm_hashset_insert(coarse_touching_start, coarse_touching_size, hedge_idx)) // dedupe among inbound hedges
                        new_inbound_count++;
                }
            }
        }
    }

    new_inbound_count = warpReduceSumLN0<uint32_t>(new_inbound_count);

    if (lane_id == 0)
        coarse_inbound_count[warp_id] = new_inbound_count;
}

// write distinct outbound touching hedges
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
// SHUFFLES OVER: h (touching)
__global__
void apply_coarsening_touching_scatter_outbound(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const dim_t* __restrict__ coarse_touching_offsets, // group id -> count of distinct touching hedges
    const uint32_t* __restrict__ coarse_inbound_count,
    uint32_t* __restrict__ coarse_touching
) {
    // STYLE: one group (new node) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the group to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_groups) return;

    /* 
    * Idea: third part of "apply_coarsening_touching_scatter" -> scatter outbound
    *  => assumes each inbound set has been now sorted, and can be used for binary search
    */

    const dim_t ungroups_start_idx = ungroups_offsets[warp_id], ungroups_end_idx = ungroups_offsets[warp_id + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    const dim_t coarse_touching_start_idx = coarse_touching_offsets[warp_id];
    const uint32_t coarse_touching_size = (uint32_t)(coarse_touching_offsets[warp_id + 1] - coarse_touching_offsets[warp_id]);
    const uint32_t new_inbound_count = coarse_inbound_count[warp_id];
    const uint32_t coarse_outbound_size = coarse_touching_size - new_inbound_count;
    uint32_t *coarse_touching_start = coarse_touching + coarse_touching_start_idx;
    uint32_t *coarse_outbound_start = coarse_touching_start + new_inbound_count;

    // SM deduplication hash-set
    extern __shared__ uint32_t block_new_touching[];
    uint32_t *new_touching = block_new_touching + MAX_SM_DEDUPE_BUFFER_SIZE * (threadIdx.x / WARP_SIZE);
    wrp_init<uint32_t>(new_touching, MAX_SM_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    __syncwarp();

    // for every original node in the group, go over its outbound set
    for (const uint32_t* ungroup = ungroups_start; ungroup < ungroups_end; ungroup++) {
        const uint32_t node = *ungroup;
        const uint32_t* my_outbound = touching + touching_offsets[node] + inbound_count[node];
        const uint32_t my_outbound_count = (uint32_t)(touching_offsets[node + 1] - touching_offsets[node]) - inbound_count[node];
        for (uint32_t i = lane_id; i < my_outbound_count; i += WARP_SIZE) {
            if (i < my_outbound_count) {
                const uint32_t hedge_idx = my_outbound[i];
                if (sm_hashset_try_insert(new_touching, MAX_SM_DEDUPE_BUFFER_SIZE, hedge_idx)) { // check in the SM cache
                    if(binary_search<uint32_t>(coarse_touching_start, new_inbound_count, hedge_idx) == UINT32_MAX) // check among inbound hedges
                        gm_hashset_insert(coarse_outbound_start, coarse_outbound_size, hedge_idx); // dedupe among outbound hedges
                }
            }
        }
    }
}

// write to each node the partition of its group
__global__
void apply_uncoarsening_partitions(
    const uint32_t* __restrict__ groups, // groups[node id] -> node's group
    const uint32_t* __restrict__ coarse_partitions, // coarse_partitions[group id] -> group's partition
    const uint32_t num_nodes,
    uint32_t* __restrict__ partitions // partitions[node id] -> group's partition
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
    uint32_t *my_pins_per_partitions = pins_per_partitions + tid * num_partitions;

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t part = partitions[*curr];
        const uint32_t prev = atomicAdd(&my_pins_per_partitions[part], 1);
        if (prev == 0) atomicAdd(&partitions_inbound_sizes[part], 1);
    }
}

// for each hyperedge, remove it source from the pins per partition counts
// SEQUENTIAL COMPLEXITY: e
// PARALLEL OVER: e
__global__
void inbound_pins_per_partition_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
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
    const uint32_t hedge_src = hedges[hedge_start_idx];
    const uint32_t src_part = partitions[hedge_src];
    uint32_t *my_inbound_pins_per_partitions = inbound_pins_per_partitions + tid * num_partitions;

    const uint32_t prev = atomicSub(&my_inbound_pins_per_partitions[src_part], 1);
    if (prev == 1) atomicSub(&partitions_inbound_sizes[src_part], 1);
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
    float histogram[PART_HIST_SIZE]; // make sure this fits in registers (no spill) !! Store here only the score, the partition id can be inferred!
    // each thread handles, at once, min(PART_HIST_SIZE, partitions_per_thread) partitions, each partition is handled by exactly one thread per warp
    uint32_t partitions_per_thread = min((num_partitions + WARP_SIZE - 1) / WARP_SIZE, PART_HIST_SIZE); // ceiled

    // all threads in the warp should agree on those...
    float best_score = 0.0f;
    uint32_t best_move = UINT32_MAX;

    // handle PART_HIST_SIZE*WARP_SIZE partitions at a time, that is partitions_per_thread per thread in the warp
    for (uint32_t curr_base_part = 0; curr_base_part < num_partitions; curr_base_part += PART_HIST_SIZE*WARP_SIZE) {
        // TODO: could make threads that have 0 partitions left (my_initial_part >= num_partitions) skip the iterations
        partitions_per_thread = min((num_partitions - curr_base_part + WARP_SIZE - 1) / WARP_SIZE, PART_HIST_SIZE);
        const uint32_t my_initial_part = curr_base_part + lane_id * partitions_per_thread;
        const uint32_t my_final_part = min(curr_base_part + (lane_id + 1) * partitions_per_thread, num_partitions);

        // clear per-thread local histograms
        for (uint32_t p = 0; p < partitions_per_thread; p++)
            histogram[p] = 0.0f;

        // TODO: shared memory caching of pins_per_partitions!! (maybe not worth it if PART_HIST_SIZE is large enough...)

        // scan touching hyperedges
        // NOTE: interpret this as "for each hedge, see if you moving to a certain partition is something that they like or not"
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            const uint32_t actual_hedge_idx = *hedge_idx;
            const uint32_t* my_pin_per_partition = pins_per_partitions + actual_hedge_idx * num_partitions;
            my_pin_per_partition += my_initial_part;
            const float my_hedge_weight = hedge_weights[actual_hedge_idx];
            // each thread in the warp reads partitions_per_thread counters
            for (uint32_t p = 0; p < partitions_per_thread; p++) {
                if (my_initial_part + p < my_final_part) {
                    // Option 1: the gain is the weighted connections count with another partition <minus> the weighted connections count with the current one
                    /*uint32_t counter = my_pin_per_partition[my_initial_part + p];
                    if (my_initial_part + p == my_partition)
                        counter--; // exclude your presence in the hyperedge from the count for your partition
                    // update local histogram
                    histogram[p] += counter*my_hedge_weight;*/
                    // Option 2: the gain is the sum the weights of hedges such that moving to a certain partition the hedge (same as HyperG)
                    uint32_t counter = my_pin_per_partition[my_initial_part + p];
                    float gain = 0.0f;
                    // hedge connected to my partition: gain the hedge's weight iff moving would disconnect it from my partition (I am its last pin left there)
                    if (counter == 1 && my_initial_part + p == my_partition)
                        gain = my_hedge_weight;
                    // hedge not yet connected to the partition: pay the hedge's weight iff moving there connects it to the new partition (I would become its first pin there)
                    else if (counter == 0 && my_initial_part + p != my_partition)
                        gain = -my_hedge_weight;
                    // update local histogram
                    histogram[p] += gain;
                }
            }
        }

        // reduce max inside each threads
        for (uint32_t p = 0; p < partitions_per_thread; p++) {
            float score = histogram[p];
            uint32_t part = my_initial_part + p;
            // TODO: could anticipate the constraint check! E.g. at the beginning of the iteration, entirely removing some partitions from the histogram,
            //       but this will require keeping a list of active partitions, since the initial-final indices won't suffice anymore...
            if (my_initial_part + p < my_final_part && partitions_sizes[part] + my_size < max_nodes_per_part && (score > best_score || score == best_score && part < best_move)) {
                best_score = score;
                best_move = part;
            }
        }

        // reduce max between threads
        bin max = warpReduceMax(best_score, best_move);
        best_score = max.score;
        best_move = max.node; // yeah, "node" should be called "partition" here, but this way we repurpose the struct...
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
        const uint32_t my_curr_part_counter = pins_per_partitions[actual_hedge_idx * num_partitions + my_partition];
        const uint32_t my_move_part_counter = pins_per_partitions[actual_hedge_idx * num_partitions + my_move_part];
        int32_t my_curr_part_counter_delta = 0;
        int32_t my_move_part_counter_delta = 0;
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            if (my_hedge < not_my_hedge) {
                uint32_t pin = *my_hedge;
                // Option 1: sorting scores to build and pass the ranking of moves
                if (move_ranks[pin] < my_move_rank) { // speculation: better-ranked move -> applied
                // Option 2: compare scores on the fly and tie-break with the node ids
                //if (scores[pin] > my_move_score || scores[pin] == my_move_score && pin < warp_id) { // speculation: better-ranked move -> applied
                // NOTE: MUST use option (1) because we need to keep the original sorting of the scores even after updating them!
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
        }
        // Option 1: the gain is the weighted connections count with another partition <minus> the weighted connections count with the current one
        //score -= my_hedge_weight*(my_curr_part_counter + warpReduceSumLN0<uint32_t>(my_curr_part_counter_delta));
        //score += my_hedge_weight*(my_move_part_counter + warpReduceSumLN0<uint32_t>(my_move_part_counter_delta));
        // Option 2: the gain is the sum the weights of hedges such that moving to a certain partition the hedge (same as HyperG)
        // gain the hedge's weight iff moving would disconnect the hedge from my partition (I am its last pin left there)
        if (my_curr_part_counter + warpReduceSumLN0<int32_t>(my_curr_part_counter_delta) == 1)
            score += my_hedge_weight;
        // pay the hedge's weight iff moving there connects the hedge to the new partition (I would become its first pin there)
        if (my_move_part_counter + warpReduceSumLN0<int32_t>(my_move_part_counter_delta) == 0)
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

    const uint32_t size = nodes_sizes[tid];

    // TODO: create no events when moves[tid] == UINT32_MAX, or set for both the ev_partition to UINT32_MAX

    // first event: node leaves its current partition
    const uint32_t e0 = 2 * tid;
    ev_partition[e0] = partitions[tid];
    ev_index[e0] = ranks[tid];
    ev_delta[e0] = -static_cast<int32_t>(size);

    // second event: node enters its destination partition
    const uint32_t e1 = e0 + 1;
    ev_partition[e1] = moves[tid];
    ev_index[e1] = ranks[tid];
    ev_delta[e1] = static_cast<int32_t>(size);
}

// mark moves that are valid points in the sequence w.r.t. size constraints
__global__
void flag_size_events_kernel(
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const int32_t* __restrict__ ev_delta,
    const uint32_t* __restrict__ partitions_sizes,
    const uint32_t num_events,
    uint32_t* __restrict__ valid_moves // initialized with 0s
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
    * - add 1 to the flag when you are the event making it invalid: the sum before you was valid, with you it is not
    * - add -1 to the flag when you are the event making the sum valid again: you are valid, then one before you was not
    * - do a scan of the flags and search for when the count is 0, when it is zero they are valid states
    * - essentially, the counter in the flags after the scan is the count of invalid partitions as of that move
    * - pick the zero-flag highest-gain move
    */

    const uint32_t part = ev_partition[tid];
    const uint32_t rank = ev_index[tid];

    // dispose of invalid moves
    if (part == UINT32_MAX) {
        valid_moves[rank] = 1;
        return;
    }

    const int32_t base_size = static_cast<int32_t>(partitions_sizes[part]);
    const int32_t curr_size = base_size + static_cast<int32_t>(ev_delta[tid]);
    const bool is_valid = curr_size <= static_cast<int32_t>(max_nodes_per_part); // true iff after this event the partition's size is valid

    const uint32_t pred_part = tid > 0 ? ev_partition[tid - 1] : UINT32_MAX; // partition acted upon by the event before this one
    const bool was_valid = pred_part != part || base_size + static_cast<int32_t>(ev_delta[tid - 1]) <= static_cast<int32_t>(max_nodes_per_part); // true iff before this event the partition's size is valid

    if (was_valid && !is_valid) // this event made the partition invalid -> track a +1 in invalid partitions as of this event
        atomicAdd(&valid_moves[rank], 1);
    if (!was_valid && is_valid) // this event made the partition invalid -> track a -1 in invalid partitions as of this event
        atomicSub(&valid_moves[rank], 1);
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
    * - let each move write to an offset given by "touching_offsets", this will leave some blank events at the end of each
    *   move's, but we can easily filter those out later by identifying them from the UINT32_MAX ev_partition...
    */

    uint32_t *my_ev_partition = ev_partition + 2 * touching_offsets[warp_id] + 2 * lane_id;
    uint32_t *my_ev_index = ev_index + 2 * touching_offsets[warp_id] + 2 * lane_id;
    uint32_t *my_ev_hedge = ev_hedge + 2 * touching_offsets[warp_id] + 2 * lane_id;
    int32_t *my_ev_delta = ev_delta + 2 * touching_offsets[warp_id] + 2 * lane_id;

    const uint32_t src_part = partitions[warp_id];
    const uint32_t dst_part = moves[warp_id];
    const uint32_t my_rank = ranks[warp_id];

    const uint32_t* inbound = touching + touching_offsets[warp_id];
    const uint32_t my_inbound_count = inbound_count[warp_id];
    for (uint32_t i = lane_id; i < my_inbound_count; i += WARP_SIZE) {
        if (i < my_inbound_count) {
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
        }
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
    uint32_t num_events,
    uint32_t num_partitions,
    uint32_t* inbound_size_events_offsets // init. to zero
) {
    // STYLE: one event per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    const uint32_t part = ev_partition[tid];

    // dispose of invalid events
    if (part == UINT32_MAX) return;

    const uint32_t hedge = ev_hedge[tid];
    const uint32_t init_hedge_inbound_count = partitions_inbound_counts[hedge*num_partitions + part];

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
    const uint32_t* inbound_size_events_offsets,
    uint32_t num_events,
    uint32_t num_partitions,
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
    const uint32_t init_hedge_inbound_count = partitions_inbound_counts[hedge*num_partitions + part];

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
    const uint32_t num_events,
    uint32_t* __restrict__ valid_moves // initialized with 0s
) {
    // STYLE: one event per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    /*
    * Idea:
    * - for each move, compute how many partitions it brings to be invalid or it brings back to a valid state
    * - then compute the number of invalid partitions at each point in time as the prefix sum of the number going from ok to not-ok (+1) and not-ok to ok (-1)
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
    const bool is_valid = curr_size <= static_cast<int32_t>(max_inbound_per_part); // true iff after this event the partition's inbound set size is valid

    const uint32_t pred_part = tid > 0 ? ev_partition[tid - 1] : UINT32_MAX; // partition acted upon by the event before this one
    const bool was_valid = pred_part != part || base_size + ev_delta[tid - 1] <= static_cast<int32_t>(max_inbound_per_part); // true iff before this event the partition's inbound set size is valid

    if (was_valid && !is_valid) // this event made the partition invalid -> track a +1 in invalid partitions as of this event
        atomicAdd(&valid_moves[rank], 1);
    if (!was_valid && is_valid) // this event made the partition invalid -> track a -1 in invalid partitions as of this event
        atomicSub(&valid_moves[rank], 1);
}