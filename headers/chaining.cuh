#pragma once
#include <cuda_runtime.h>

#include "defines.cuh"
#include "data_types.cuh"


// USED BY: chaining

#define ITERS 4 // multi-iteration greedy chaining
#define WINDOW 256 // candidates scanned per node per iteration
#define ALPHA 1e-6f // node size penalty scale (adjust based on size magnitude)
//#define BETA 1e-7f // inbound set size penalty scale (adjust based on inbound set size magnitude)


void chaining(
    const uint32_t *srcs,
    const uint32_t *dsts,
    const uint32_t *size,
    const float *weight,
    const uint32_t num,
    uint32_t *sequence_idx,
    cudaStream_t stream = 0
);

void build_orphan_pairs(
    const uint32_t* d_nodes_sizes,
    const uint32_t* d_inbound_count,
    const uint32_t* d_pairs,
    const uint32_t curr_num_nodes,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    const uint32_t candidates_count,
    uint32_t* d_groups
);