#pragma once
#include <string>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "defines.cuh"
#include "data_types.cuh"

namespace config {
    struct runconfig;
}

using namespace config;


// USED BY: initial part

#define VERBOSE_INIT_LENGTH 20

// controls
#define RANDOM_INIT_TRIES 32 // number of different random initializations to try
#define MAX_CONSECUTIVE_INIT_FAILURES 4 // number of attempts at random initialization before giving up (-> exception)
#define JACOBI_TRIES 64 // number of jacobi improvement rounds per initial partition to perform
#define FM_TRIES 64 // number of FM refinement rounds per initial partition to perform
//#define REPAIR_SPEED 1 // by how much to recover size constraints violations per repair repetition

#define MAX_OMP_THREADS 16


// STEPS

std::tuple<uint32_t*, uint32_t*> initial_partitioning(
    const runconfig &cfg,
    const uint32_t num_nodes,
    const uint32_t num_hedges,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const float* d_hedge_weights,
    const dim_t hedges_size,
    const uint32_t* d_touching,
    const dim_t* d_touching_offsets,
    const uint32_t* d_nodes_sizes,
    const uint32_t max_parts,
    const uint32_t h_max_nodes_per_part
);

std::tuple<uint32_t*, uint32_t*> initial_partitioning_kahypar(
    const runconfig &cfg,
    const uint32_t num_nodes,
    const uint32_t num_hedges,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const float* d_hedge_weights,
    const dim_t* d_touching_offsets,
    const dim_t hedges_size,
    const uint32_t* d_nodes_sizes,
    const uint32_t k,
    const float epsilon,
    const uint32_t h_max_nodes_per_part
);

float compute_connectivity(
    const runconfig &cfg,
    const uint32_t max_parts,
    const uint32_t num_hedges,
    const uint32_t* d_pins_per_partitions,
    const float* d_hedge_weights,
    const cudaStream_t stream
);

void logInitPartitions(
    const uint32_t num_nodes,
    const uint32_t max_parts,
    const uint32_t h_max_nodes_per_part,
    const uint32_t* d_partitions,
    const uint32_t* d_partitions_sizes,
    std::string text,
    const cudaStream_t stream
);


// KERNELS

__global__
void init_partitions_random(
    const uint32_t num_nodes,
    const uint32_t seed,
    const uint32_t num_partitions,
    const uint32_t* __restrict__ nodes_sizes,
    uint32_t* __restrict__ partitions,
    uint32_t* __restrict__ partitions_sizes,
    bool* __restrict__ fail
);

__global__
void pins_per_partition_only_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ pins_per_partitions
);

__global__
void jacobi_apply_moves_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ partitions_sizes_aux,
    const uint32_t* __restrict__ move_srcs,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    uint32_t* __restrict__ partitions,
    uint32_t* __restrict__ partitions_sizes,
    uint32_t* __restrict__ pins_per_partitions,
    bool* __restrict__ continue_flag,
    uint64_t* __restrict__ partitions_hash
);

__global__
void fm_refinement_apply_update_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ move_ranks,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t num_good_moves,
    uint32_t* __restrict__ partitions,
    uint32_t* __restrict__ partitions_sizes,
    uint32_t* __restrict__ pins_per_partitions,
    uint64_t* __restrict__ partitions_hash
);

__global__
void armonic_degree_score_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    float* __restrict__ hedge_ratio
);

__global__
void prune_hedges_kernel(
    const float* __restrict__ hedge_weights,
    const float* __restrict__ hedge_ratio,
    const uint32_t num_hedges,
    const float threshold,
    const uint32_t seed,
    float* __restrict__ hedge_scaled_weights,
    uint8_t* __restrict__ keep
);

__global__
void compute_connectivity_kernel(
    const uint32_t* __restrict__ pins_per_partitions,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    float* __restrict__ connectivity
);