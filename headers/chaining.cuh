#pragma once
#include <stdint.h>
#include <algorithm>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include "utils.cuh"

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