#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32u
#define HIST_SIZE 64u // must be a multiple of WARP_SIZE (for the histogram max reduction)

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
    const uint32_t* hedge_offsets,
    const uint32_t* hedges,
    const uint32_t* neighbors,
    const uint32_t* neighbors_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    uint32_t* pairs,
    uint32_t* groups
) {
    // STYLE: one node per warp!
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // global across blocks - coincides with the node to handle
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;

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
            uint32_t actual_hedge_idx = *hedge_idx;
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
        pairs[warp_id] = best_neighbor;
        // TODO: cheeky hack to initialize groups in parallel for the next kernel ("grouping_kernel")
        //groups[warp_id] = ((uint64_t) warp_id << 32) | 0u;
        groups[warp_id] = warp_id;
    }
}

// find disjoint set root for a node while running path compression
__device__ __forceinline__
void find_root(
    const uint32_t start, // node for which to find root and distance
    uint32_t* groups,
    uint32_t &root_out
) {
    uint32_t curr = start, prev = start;
    while (true) {
        uint32_t parent = groups[curr];
        if (parent == curr) break; // root
        atomicCAS(&groups[prev], curr, parent);
        prev = curr;
        curr = parent;
    }
    root_out = curr;
}

//__device__ uint32_t synch = 0u;

// build the disjoint set
// TODO: could benefit from coarsening on larger hgraphs!
__global__
void grouping_minfirst_kernel(
    const uint32_t* pairs, // pairs[idx] is the partner idx wants to be grouped with
    const uint32_t num_nodes,
    // TODO: could be made more efficient like HyperG, by keeping groups and distances in one uint64 value (better locality)...
    uint32_t* groups // initialize for each node equal to itself (root of its own group)
) {
    // STYLE: one node per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t target = pairs[tid];

    // trivial self-pair
    if (target == tid) return;

    // repeatedly attempt to connect the current node and its target until they share the same root
    // => limit iterations just in case, to avoid pathological infinite loops.
    const int MAX_ITERS = 1024;
    for (int i = 0; i < MAX_ITERS; i++) {
        uint32_t r1, r2;
        find_root(tid, groups, r1);
        find_root(target, groups, r2);

        if (r1 == r2) break;

        // Deterministic order (the root with lower id becomes the parent)
        uint32_t parent = (r1 < r2) ? r1 : r2;
        uint32_t child = (r1 < r2) ? r2 : r1;

        // try to link child --to-> parent, only modify the child's root group to keep the tree acyclic
        // => path compression will eventually propagate the update root's group downward
        atomicMin(&groups[child], parent);
        // optional: immediate (forceful) update of the distances and path compression
        find_root(tid, groups, r1);
        find_root(target, groups, r2);

        if (r1 == r2) break;
    }
}

__global__
void groups_sort_kernel(
    const uint32_t* pairs, // pairs[idx] is the partner idx wants to be grouped with
    const uint32_t* groups,
    const uint32_t num_nodes,
    uint32_t* nodes_map // assign to each node (identified by its idx) a new id (in practice, sorting them)
) {
    // STYLE: one node per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t target = pairs[tid];
}

__global__
void groups_breakdown_kernel(
    const uint32_t* pairs, // pairs[idx] is the partner idx wants to be grouped with
    const uint32_t num_nodes,
    uint32_t* groups,
    uint32_t* nodes_map // assign to each node (identified by its idx) a new id (in practice, sorting them)
) {
    // STYLE: one node per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t target = pairs[tid];
}


// FUUUUCK: hyperedges and neighbors, to be stored contigously AND be deduplicated can only be handled sequentially!
// => the most parallelism I can exploit here is along "d" or "h"...

__global__
void apply_coarsening_hedges(
    const uint32_t num_nodes,
    const uint32_t* groups,
    uint32_t* hedge_offsets,
    uint32_t* hedges,
    uint32_t* hedge_offsets,
    uint32_t* hedges,
    uint32_t* num_hedges,
) {
    // STYLE: one node per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t target = pairs[tid];
}

__global__
void apply_coarsening_neighbors(
    const uint32_t num_nodes,
    const uint32_t* groups,
    uint32_t* neighbors,
    uint32_t* neighbors_offsets,
) {
    // STYLE: one node per thread!
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t target = pairs[tid];
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
}