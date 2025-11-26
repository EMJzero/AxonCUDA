#include </home/mronzani/cuda/include/cuda_runtime.h>
#include <stdint.h>

// returns the average index of nodes (pins) for each hyperedge (arguably useless)
extern __global__
void hyperedge_avg_kernel(
    const uint32_t* hedge_offsets, // array of indices where each hyperedge in "hedge" starts
    const uint32_t* hedges, // concatenated nodes of all hyperedges
    float* averages,
    uint32_t num_hedges
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_hedges) return;

    // NOTE: for now performance will be rubbish, accesses are not coalesced!

    uint32_t start = hedge_offsets[i];
    uint32_t end = hedge_offsets[i + 1];

    float sum = 0.0f;
    uint32_t count = end - start;

    for (uint32_t j = start; j < end; j++) {
        sum += hedges[j];
    }

    averages[i] = sum / count;
}