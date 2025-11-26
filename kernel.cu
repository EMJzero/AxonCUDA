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

__global__
void hyperedge_candidate_kernel(
    const uint32_t* hedge_offsets,
    const uint32_t* hedges,
    const uint32_t* neighbors,
    const uint32_t* neighbors_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    uint32_t* out_pairs,
    float* out_scores
) {
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
        out_pairs[warp_id] = best_neighbor;
        out_scores[warp_id] = best_score;
    }
}
