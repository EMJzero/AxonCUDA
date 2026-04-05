#include "init_part.cuh"
#include "constants.cuh"
#include "utils.cuh"

// for each node, randomly assign it to the first partition that still has space
// SEQUENTIAL COMPLEXITY: n*p (worst case linear scan)
// PARALLEL OVER: n
__global__
void init_partitions_random(
    const uint32_t num_nodes,
    const uint32_t seed,
    const uint32_t num_partitions,
    const uint32_t* __restrict__ nodes_sizes, // node_sizes[idx] is how many original nodes the idx coarse node contains
    uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    uint32_t* __restrict__ partitions_sizes, // partitions_remaining_space[idx] is how many nodes (by size) are in partition idx
    bool* __restrict__ fail // set when one node cannot be assigned
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t x = seed ^ tid;
    auto rng = [&]() {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    };
    
    const uint32_t my_size = nodes_sizes[tid];
    
    // try random partitions
    for (int t = 0; t < 8; t++) {
        const uint32_t part = rng() % num_partitions;
        const uint32_t old = atomicAdd(&partitions_sizes[part], my_size);
        if (old + my_size <= max_nodes_per_part) {
            partitions[tid] = part;
            return;
        }
        atomicSub(&partitions_sizes[part], my_size);
    }

    // fallback: linear scan
    rng();
    for (uint32_t p = 0; p < num_partitions; p++) {
        const uint32_t part = (p + x) % num_partitions;
        const uint32_t old = atomicAdd(&partitions_sizes[part], my_size);
        if (old + my_size <= max_nodes_per_part) {
            partitions[tid] = part;
            return;
        }
        atomicSub(&partitions_sizes[part], my_size);
    }

    //assert(false && "Could not fit every node in the available partitions!");
    *fail = true;
}

// for each hyperedge, count how many of its pins are in each partition
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void pins_per_partition_only_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ pins_per_partitions // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    const dim_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    uint32_t *my_pins_per_partitions = pins_per_partitions + tid * num_partitions;

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t part = partitions[*curr];
        atomicAdd(&my_pins_per_partitions[part], 1);
    }
}

// apply moves within the size constraint
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
__global__
void jacobi_apply_moves_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ partitions_sizes_aux, // -> this is now the node sizes scan + initial partition size = final partition size if a move is applied
    const uint32_t* __restrict__ move_srcs,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    uint32_t* __restrict__ partitions_sizes,
    uint32_t* __restrict__ pins_per_partitions, // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
    bool* __restrict__ continue_flag,
    uint64_t* __restrict__ partitions_hash
) {
    // STYLE: one node (move) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    // block-level hash
    __shared__ uint64_t blk_partitions_hash;
    if (threadIdx.x == 0) blk_partitions_hash = 0;
    __syncthreads();

    const uint32_t dst_part = moves[tid];
    const uint32_t final_part_size = partitions_sizes_aux[tid];
    const uint32_t node = move_srcs[tid];

    // filter out null moves AND stop at the first move that would push partition size beyond constraints
    if (dst_part == UINT32_MAX || final_part_size > max_nodes_per_part) {
        uint64_t hash = hash_uint64((uint64_t)partitions[node] ^ (uint64_t(node) << 32));
        atomicXor((unsigned long long*)&blk_partitions_hash, (unsigned long long)hash);
    } else {
        // if this is a valid move, set the flag
        *continue_flag = true;
        
        const uint32_t my_size = nodes_sizes[node];
        const uint32_t src_part = partitions[node];

        // update partition sizes
        atomicAdd(&partitions_sizes[dst_part], my_size);
        atomicSub(&partitions_sizes[src_part], my_size);

        // update my partition
        partitions[node] = dst_part;
        uint64_t hash = hash_uint64((uint64_t)dst_part ^ (uint64_t(node) << 32));
        atomicXor((unsigned long long*)&blk_partitions_hash, (unsigned long long)hash);

        const uint32_t* my_touching = touching + touching_offsets[node];
        const uint32_t* not_my_touching = touching + touching_offsets[node + 1];

        // scan touching hyperedges and update pins_per_partitions counts
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            const uint32_t actual_hedge_idx = *hedge_idx;
            atomicAdd(&pins_per_partitions[actual_hedge_idx * num_partitions + dst_part], 1);
            atomicSub(&pins_per_partitions[actual_hedge_idx * num_partitions + src_part], 1);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) atomicXor((unsigned long long*)partitions_hash, (unsigned long long)blk_partitions_hash);
}

// apply moves with a positive gain
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
__global__
void fm_refinement_apply_update_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ moves, // moves[idx] -> positive-gain move (target partition idx) proposed by node idx (DO NOT SORT)
    const uint32_t* __restrict__ move_ranks, // move_ranks[node_idx] -> i (ranking by score) of the move proposed by the idx node
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t num_good_moves, // idx + 1 of the maximum in the updated scores
    uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    uint32_t* __restrict__ partitions_sizes,
    uint32_t* __restrict__ pins_per_partitions, // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
    uint64_t* __restrict__ partitions_hash
) {
    // STYLE: one node (move) per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    // block-level hash
    __shared__ uint64_t blk_partitions_hash;
    if (threadIdx.x == 0) blk_partitions_hash = 0;
    __syncthreads();
    
    const uint32_t my_partition = partitions[tid];
    const uint32_t my_move_part = moves[tid];
    const uint32_t my_size = nodes_sizes[tid];

    // stop at the last gain-increasing move
    // TODO: remove "moves[tid] == UINT32_MAX", it's redundant and here "just in case", invalid moves should always be outside of num_good_moves
    if (move_ranks[tid] >= num_good_moves || my_move_part == UINT32_MAX) {
        uint64_t hash = hash_uint64((uint64_t)my_partition ^ (uint64_t(tid) << 32));
        atomicXor((unsigned long long*)&blk_partitions_hash, (unsigned long long)hash);
    } else {
        // update partition sizes
        atomicSub(&partitions_sizes[my_partition], my_size);
        atomicAdd(&partitions_sizes[my_move_part], my_size);
        
        // update my partition
        partitions[tid] = my_move_part;
        uint64_t hash = hash_uint64((uint64_t)my_move_part ^ (uint64_t(tid) << 32));
        atomicXor((unsigned long long*)&blk_partitions_hash, (unsigned long long)hash);

        const uint32_t* my_touching = touching + touching_offsets[tid];
        const uint32_t* not_my_touching = touching + touching_offsets[tid + 1];
        
        // scan touching hyperedges and update pins_per_partitions counts
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            const uint32_t actual_hedge_idx = *hedge_idx;
            atomicSub(&pins_per_partitions[actual_hedge_idx * num_partitions + my_partition], 1);
            atomicAdd(&pins_per_partitions[actual_hedge_idx * num_partitions + my_move_part], 1);
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) atomicXor((unsigned long long*)partitions_hash, (unsigned long long)blk_partitions_hash);
}

// compute for each hedge a score proportional to how "rare" are its nodes:
// score(e) = w(e) / (\sum_{n \in e} 1/deg(n))
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// WARPS OVER: d
__global__
void armonic_degree_score_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    float* __restrict__ hedge_ratio
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the hedge to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    const dim_t hedge_start_idx = hedges_offsets[warp_id], hedge_end_idx = hedges_offsets[warp_id + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;

    float score = 0.0f;

    for (const uint32_t* curr = hedge_start + lane_id; curr < hedge_end; curr += WARP_SIZE) {
        const uint32_t pin = *curr;
        const uint32_t deg = touching_offsets[pin + 1] - touching_offsets[pin];
        score += 1/(float)deg;
    }

    score = warpReduceSumLN0<float>(score);
    
    if (lane_id == 0)
        hedge_ratio[warp_id] = hedge_weights[warp_id] / score;
        //hedge_score[warp_id] = 1/score;
}

// flag each hedge as 'keep' or 'prune' with a probability conditioned on its weight and score
// SEQUENTIAL COMPLEXITY: e
// PARALLEL OVER: e
__global__
void prune_hedges_kernel(
    const float* __restrict__ hedge_weights,
    const float* __restrict__ hedge_ratio,
    const uint32_t num_hedges,
    const float threshold,
    const uint32_t seed,
    float* __restrict__ hedge_scaled_weights,
    uint8_t* __restrict__ keep
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    // probability of keeping the hedge: p(e) = min(1, threshold * w(e) / score(e))

    const float keep_prob = min(1.0f, threshold * hedge_ratio[tid]);

    // sample random float in (0, 1]
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);
    const float rand = curand_uniform(&state);

    keep[tid] = keep_prob >= rand;
    hedge_scaled_weights[tid] = hedge_weights[tid] / keep_prob; // rescale weights (lower probability -> higher weight)
}

// quickly compute the connectivity resulting from a given partitioning
// SEQUENTIAL COMPLEXITY: e*p
// PARALLEL OVER: e
__global__
void compute_connectivity_kernel(
    const uint32_t* __restrict__ pins_per_partitions,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    float* __restrict__ connectivity // connectivity[idx] -> total cut cost paid by the idx-th group of block-size hedges
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    __shared__ float blk_conn;
    if (threadIdx.x == 0) blk_conn = 0.0f;
    __syncthreads();

    uint32_t lambda = 0;
    const float hedge_weight = hedge_weights[tid];

    for (uint32_t p = 0u; p < num_partitions; p++) {
        if (pins_per_partitions[tid * num_partitions + p] > 0)
            lambda++;
    }

    atomicAdd(&blk_conn, lambda > 0 ? hedge_weight*(lambda - 1) : 0.0f);

    __syncthreads();
    if (threadIdx.x == 0) connectivity[blockIdx.x] = blk_conn;
}