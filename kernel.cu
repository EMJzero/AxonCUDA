#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32u
#define HIST_SIZE 64u // must be a multiple of WARP_SIZE (for the histogram max reduction)

#define MAX_GROUP_SIZE 4 // => MAX_GROUP_SIZE - 1 slots per node

typedef struct {
    uint32_t node;
    float score;
} bin;

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ bin warpReduceMax(float val, uint32_t payload) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        uint32_t other_payload = __shfl_down_sync(0xffffffff, payload, offset);
        if (other_val > val) {
            val = other_val;
            payload = other_payload;
        }
    }
    return {.node = payload, .score = val};
}

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
    float* scores,
    uint32_t* groups // piggyback for initialization...
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
            if (nb < neighbors_count)
                histogram[nb].node = my_neighbors[nb];
            else
                histogram[nb].node = UINT32_MAX;
            histogram[nb].score = 0.0f;
        }

        // TODO: shared memory caching of hyperedges!!

        // scan touching hyperedges
        // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            uint32_t actual_hedge_idx = *hedge_idx & bits_key_neg;
            const uint32_t* my_hedge = hedges + hedge_offsets[actual_hedge_idx];
            my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
            const uint32_t* not_my_hedge = hedges + hedge_offsets[actual_hedge_idx + 1];
            float my_hedge_weight = hedge_weights[actual_hedge_idx];
            for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
                uint32_t pin = UINT32_MAX - 1; 
                if (my_hedge < not_my_hedge)
                    pin = *my_hedge;
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
        pairs[warp_id] = best_neighbor & bits_key_neg;
        scores[warp_id] = best_score;
        // TODO: cheeky hack to initialize groups in parallel for the next kernel ("grouping_kernel")
        groups[warp_id] = warp_id;
    }
}

// create groups of at most M nodes, highest score first
__global__
void grouping_kernel(
    const uint32_t* pairs, // pairs[idx] is the partner idx wants to be grouped with (no (multi)function bits here)
    const uint32_t num_nodes,
    const uint32_t bits_key_neg, // do the bitwise-and only when using node ids as indices!
    // TODO: change this thing's type to a struct float+uint32_t, or reuse "bin" directly!
    bin* group_slots // node -> MAX_GROUP_SIZE contigous slots per node, each containing the score of the pair and the other half of the pair
) {
    // STYLE: one node per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t target = pairs[tid];

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
    */
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
    uint32_t* hedge_start = hedges + hedge_start, hedge_end = hedges + hedge_end;
    uint32_t size = hedge_end_idx - hedge_start_idx;
    uint32_t distinct = 0;
    uint32_t hedge[size];

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
            hedges[hedge_start + i] = hedge[i];
        else
            hedges[hedge_start + i] = UINT32_MAX;
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
    uint32_t* neighbor_start = neighbors + neighbor_start, neighbor_end = neighbors + neighbor_end;
    uint32_t size = neighbor_end_idx - neighbor_start_idx;
    uint32_t distinct = 0;
    uint32_t neighbor[size];

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
            neighbors[neighbor_start + i] = neighbor[i];
        else
            neighbors[neighbor_start + i] = UINT32_MAX;
    }
}

__global__
void apply_coarsening_touching(
    const uint32_t num_nodes,
    const uint32_t* groups,
    uint32_t* touching,
    uint32_t* touching_offsets,
) {
    // STYLE: one node per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t target = pairs[tid];

    // TODO: ...
}