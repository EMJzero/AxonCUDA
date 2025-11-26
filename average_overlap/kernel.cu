#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32

__inline__ __device__ uint32_t warpReduceSum(uint32_t val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__
void hyperedge_overlap_kernel(
    const uint32_t* hedge_offsets,
    const uint32_t* hedges,
    float* averages,
    uint32_t num_hedges
) {
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    uint32_t lane = threadIdx.x % WARP_SIZE;

    // hyperedge assigned to this warp
    uint32_t he_start = hedge_offsets[warp_id];
    uint32_t he_end = hedge_offsets[warp_id + 1];
    uint32_t he_len = he_end - he_start;

    // load assigned hyperedge nodes into registers/local memory
    // extern __shared__ uint32_t shared_nodes[]; // dynamically sized if needed
    // TODO: for now, let's assume each warp can hold its hyperedge locally,
    // then, if > 128*WARP_SIZE, we can iterate in batches
    uint32_t he_nodes[128]; // adjust if max hyperedge size is > 128*WARP_SIZE
    uint32_t he_block_len = 0;

    for (; lane + he_block_len*WARP_SIZE < he_len; he_block_len += 1)
        he_nodes[he_block_len] = hedges[he_start + he_block_len*WARP_SIZE + lane];

    __syncwarp(); // make sure all threads have loaded

    uint32_t total_common = 0;
    // iterate over all hyperedges
    for (uint32_t other = 0; other < num_hedges; ++other) {
        if (other == warp_id) continue; // skip itself

        uint32_t o_start = hedge_offsets[other];
        uint32_t o_end = hedge_offsets[other + 1];
        uint32_t o_len = o_end - o_start;

        // scan other hyperedge in batches of WARP_SIZE
        for (uint32_t batch = 0; batch < o_len; batch += WARP_SIZE) {
            uint32_t node_idx = batch + lane;
            uint32_t node = (node_idx < o_end) ? hedges[o_start + node_idx] : UINT32_MAX;

            // shuffle (full-circle) so all threads see all nodes in batch
            #pragma unroll
            for (int i = 0; i < WARP_SIZE; i++) {
                uint32_t n = __shfl_sync(0xffffffff, node, i); // i % width is done automatically

                // check if this node is in (this thread's part of) the assigned hyperedge
                for (uint32_t j = 0; j < he_block_len; j++)
                    if (he_nodes[j] == n)
                        total_common++;
            }
        }
    }

    // Warp-level reduction to lane 0
    uint32_t warp_sum = warpReduceSum(total_common);

    if (lane == 0)
        averages[warp_id] = static_cast<float>(warp_sum) / (num_hedges - 1); // avg w.r.t. other hyperedges
        //averages[warp_id] = static_cast<float>(warp_sum) / he_len; // avg w.r.t. my pins
}
