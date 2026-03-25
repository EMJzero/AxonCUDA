#pragma once
#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <format>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <optional>
#include <algorithm>
#include <filesystem>
#include <unordered_map>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include <cub/cub.cuh>

#include <omp.h>

#include "utils.cuh"
#include "constants.cuh"
#include "refinement.cuh"
#include "chaining.cuh"

#define VERBOSE_INIT false
#define VERBOSE_INIT_LOG false
#define VERBOSE_INIT_CONN false
#define VERBOSE_INIT_LENGTH 20

// controls
#define RANDOM_INIT_TRIES 32 // number of different random initializations to try
#define MAX_CONSECUTIVE_INIT_FAILURES 4 // number of attempts at random initialization before giving up (-> exception)
#define JACOBI_TRIES 64 // number of jacobi improvement rounds per initial partition to perform
#define FM_TRIES 64 // number of FM refinement rounds per initial partition to perform
#define REPAIR_SPEED 1 // by how much to recover size constraints violations per repair repetition

#define MAX_THREADS 16

std::tuple<uint32_t*, uint32_t*> initial_partitioning(
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