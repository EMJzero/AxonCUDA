#include "prep.cuh"
#include "utils.cuh"

// count how many hedges touch each node
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// SHUFFLES OVER: d
__global__
void touching_count_kernel(
    const uint32_t* __restrict__ hedges, // stores srcs first, then dsts
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    dim_t* __restrict__ touching_offsets // initialized at 0s
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the hedge to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    /*
    * Idea:
    * - every warp visits an hedge
    * - for every pin, it atomically increments the pin's touching set size
    */

    const uint32_t* hedge = hedges + hedges_offsets[warp_id];
    const uint32_t hedge_size = (uint32_t)(hedges_offsets[warp_id + 1] - hedges_offsets[warp_id]);

    for (uint32_t pin_idx = lane_id; pin_idx < hedge_size; pin_idx += WARP_SIZE) {
        const uint32_t pin = hedge[pin_idx]; // already a group id
        atomicAdd(&touching_offsets[pin + 1], 1); // leave the first entry to be 0 (offset of the first set)
    }
}

// write incidence sets
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
// SHUFFLES OVER: h (touching)
__global__
void touching_build_kernel(
    const uint32_t* __restrict__ hedges, // already coarsened as of here, thus contain group ids!
    const dim_t* __restrict__ hedges_offsets,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_hedges,
    uint32_t* __restrict__ touching,
    uint32_t* __restrict__ inserted_count // initialized at 0s
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the hedge to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    /*
    * Idea:
    * - every warp visits an hedge
    * - for every pin, it claims a slot by atomically incrementing the pin's count of seen srcs, then inserts the hedge in the pin's incidence set
    */

    const uint32_t* hedge = hedges + hedges_offsets[warp_id];
    const uint32_t hedge_size = (uint32_t)(hedges_offsets[warp_id + 1] - hedges_offsets[warp_id]);

    for (uint32_t pin_idx = lane_id; pin_idx < hedge_size; pin_idx += WARP_SIZE) {
        const uint32_t pin = hedge[pin_idx]; // already a group id
        uint32_t *pin_touching = touching + touching_offsets[pin];
        uint32_t insert_idx = atomicAdd(&inserted_count[pin], 1);
        pin_touching[insert_idx] = warp_id;
    }
}