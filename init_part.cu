#include <tuple>
#include <string>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <filesystem>

#include </home/mronzani/cuda/include/cuda_runtime.h>

#include </home/mronzani/cuda/include/thrust/sort.h>
#include </home/mronzani/cuda/include/thrust/scan.h>
#include </home/mronzani/cuda/include/thrust/gather.h>
#include </home/mronzani/cuda/include/thrust/scatter.h>
#include </home/mronzani/cuda/include/thrust/sequence.h>
#include </home/mronzani/cuda/include/thrust/transform.h>
#include </home/mronzani/cuda/include/thrust/device_ptr.h>
#include </home/mronzani/cuda/include/thrust/device_vector.h>
#include </home/mronzani/cuda/include/thrust/iterator/discard_iterator.h>
#include </home/mronzani/cuda/include/thrust/iterator/permutation_iterator.h>

#include </home/mronzani/cuda/include/cub/cub.cuh>

#include <omp.h>

#include "utils.cuh"

#define VERBOSE_INIT false
#define VERBOSE_INIT_LOG false
#define VERBOSE_INIT_CONN false
#define VERBOSE_INIT_LENGTH 20

// controls
#define RANDOM_INIT_TRIES 512 // number of different random initializations to try
#define MAX_CONSECUTIVE_INIT_FAILURES 64 // number of attempts at random initialization before giving up (-> exception)
#define JACOBI_TRIES 64 // number of jacobi improvement rounds per initial partition to perform
#define FM_TRIES 64 // number of FM refinement rounds per initial partition to perform
#define REPAIR_SPEED 1 // by how much to recover size constraints violations per repair repetition

#define MAX_THREADS 16

extern __global__ void fm_refinement_gains_kernel(
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t* partitions,
    const uint32_t* pins_per_partitions,
    const uint32_t* nodes_sizes,
    const uint32_t* partitions_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t randomizer,
    const uint32_t discount,
    uint32_t* moves,
    float* scores
);

extern __global__ void fm_refinement_cascade_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t* move_ranks,
    const uint32_t* moves,
    const uint32_t* partitions,
    const uint32_t* pins_per_partitions,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    float* scores
);

extern __global__ void build_size_events_kernel(
    const uint32_t* moves,
    const uint32_t* ranks,
    const uint32_t* partitions,
    const uint32_t* nodes_sizes,
    const uint32_t num_nodes,
    uint32_t* ev_partition,
    uint32_t* ev_index,
    int32_t* ev_delta
);

extern __global__ void flag_size_events_kernel(
    const uint32_t* ev_partition,
    const uint32_t* ev_index,
    const int32_t* ev_delta,
    const uint32_t* partitions_sizes,
    const uint32_t num_events,
    int32_t* valid_moves
);

extern void chaining(
    const uint32_t *srcs,
    const uint32_t *dsts,
    const uint32_t *size,
    const float *weight,
    const uint32_t num,
    uint32_t *sequence_idx,
    cudaStream_t stream
);

extern __constant__ uint32_t max_nodes_per_part;


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
    const uint32_t* __restrict__ partition_sizes_aux, // -> this is now the node sizes scan + initial partition size = final partition size if a move is applied
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
    const uint32_t final_part_size = partition_sizes_aux[tid];
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

// find the best move of each node from its partition to another
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
    // STYLE: one node (move) per thread!
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

void log_partitions(
    const uint32_t num_nodes,
    const uint32_t max_parts,
    const uint32_t h_max_nodes_per_part,
    const uint32_t* d_partitions,
    const uint32_t* d_partitions_sizes,
    std::string text,
    const cudaStream_t stream
) {
    std::vector<uint32_t> partitions_tmp(num_nodes);
    CUDA_CHECK(cudaMemcpyAsync(partitions_tmp.data(), d_partitions, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    std::vector<uint32_t> partitions_sizes_tmp(max_parts);
    CUDA_CHECK(cudaMemcpyAsync(partitions_sizes_tmp.data(), d_partitions_sizes, max_parts * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::unordered_map<uint32_t, int> part_count;
    std::cout << text << ":\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        uint32_t part = partitions_tmp[i];
        part_count[part]++;
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_INIT_LENGTH)) {
            std::cout << "  node " << i << " -> " << part;
            std::cout << ((i + 1) % 4 == 0 ? "\n" : "\t");
        }
    }
    for (uint32_t i = 0; i < max_parts; ++i) {
        uint32_t part_size = partitions_sizes_tmp[i];
        if (part_size > h_max_nodes_per_part)
            std::cerr << "  WARNING, max partition size constraint (" << h_max_nodes_per_part << ") violated by part=" << i << " with part_size=" << part_size << " !!\n";
    }
    int max_ps = part_count.empty() ? 0 : std::max_element(part_count.begin(), part_count.end(), [](auto &a, auto &b){ return a.second < b.second; })->second;
    std::cout << "Non-empty partitions count: " << part_count.size() << ", Max partition size: " << max_ps << "\n";
    std::vector<uint32_t>().swap(partitions_tmp);
    std::vector<uint32_t>().swap(partitions_sizes_tmp);
    std::unordered_map<uint32_t, int>().swap(part_count);
}

float compute_connectivity(
    const uint32_t max_parts,
    const uint32_t num_hedges,
    const uint32_t* d_pins_per_partitions,
    const float* d_hedge_weights,
    const cudaStream_t stream
) {
    float* d_connectivity;
    // launch configuration - fm-ref gains kernel
    int threads_per_block = 1024;
    int num_threads_needed = num_hedges; // 1 thread per hedge
    int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
    CUDA_CHECK(cudaMallocAsync(&d_connectivity, blocks * sizeof(float), stream)); // first reduce inside each block, then across blocks with thrust
    // launch - fm-ref gains kernel
    #if VERBOSE_INIT
    std::cout << "Running compute connectivity kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    #endif
    compute_connectivity_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_pins_per_partitions,
        d_hedge_weights,
        num_hedges,
        max_parts,
        d_connectivity // connectivity[idx] -> total cut cost paid by the idx-th partition
    );
    CUDA_CHECK(cudaGetLastError());
    thrust::device_ptr<float> t_connectivity(d_connectivity);
    float conn = thrust::reduce(thrust::cuda::par.on(stream), t_connectivity, t_connectivity + blocks);
    CUDA_CHECK(cudaFreeAsync(d_connectivity, stream));
    return conn;
}

// initial partitioning for the k-way balanced version (no inbound constraint)
std::tuple<uint32_t*, uint32_t*> initial_partitioning(
    const uint32_t num_nodes,
    const uint32_t num_hedges,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    //const uint32_t* d_srcs_count,
    const float* d_hedge_weights,
    const dim_t hedges_size,
    const uint32_t* d_touching,
    const dim_t* d_touching_offsets,
    //const dim_t touching_size,
    //const uint32_t* d_inbound_count,
    const uint32_t* d_nodes_sizes,
    const uint32_t max_parts,
    const uint32_t h_max_nodes_per_part
) {
    std::cout << "Building initial partitioning, remaining nodes=" << num_nodes << ", remaining pins=" << hedges_size << "\n";

    // best results
    float best_conn = FLT_MAX;
    uint32_t *d_best_partitions = nullptr;
    uint32_t *d_best_partitions_sizes = nullptr;

    // setup streams for OpenMP parallel random initializations
    const int max_threads = std::min(omp_get_max_threads(), MAX_THREADS);
    omp_set_num_threads(max_threads);
    std::cout << "Spawning " << max_threads << " OpenMP threads ...\n";
    std::vector<cudaStream_t> streams(max_threads);
    for (int i = 0; i < max_threads; ++i)
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

    #pragma omp parallel for schedule(dynamic) shared(best_conn, d_best_partitions, d_best_partitions_sizes, streams) 
    for (int repeats = 0; repeats < RANDOM_INIT_TRIES; repeats++) {
        // OpenMP stuff
        int tid = omp_get_thread_num();
        cudaStream_t stream = streams[tid];
        auto thrust_exec = thrust::cuda::par.on(stream);

        // kernel dimensions
        int blocks, threads_per_block, warps_per_block;
        int num_threads_needed, num_warps_needed;
        //size_t bytes_per_thread, bytes_per_warp, shared_bytes;
        //int blocks_per_SM, max_blocks;

        // partitions
        uint32_t *d_partitions = nullptr;
        uint32_t *d_partitions_sizes = nullptr;
        uint32_t *d_pins_per_partitions = nullptr;
        uint64_t *d_partitions_hash = nullptr;
        uint64_t h_partitions_hash = 0;
        bool* d_fail = nullptr;
        bool h_fail = true;
        CUDA_CHECK(cudaMallocAsync(&d_partitions, num_nodes * sizeof(uint32_t), stream)); // partitions[idx] -> partition in which is node idx
        CUDA_CHECK(cudaMallocAsync(&d_partitions_sizes, max_parts * sizeof(uint32_t), stream)); // partitions_sizes[idx] -> how many nodes (by size) are in the partition
        CUDA_CHECK(cudaMallocAsync(&d_pins_per_partitions, num_hedges * max_parts * sizeof(uint32_t), stream)); // hedge * num_partitions + partition -> count of pins of "hedge" in that "partition"
        CUDA_CHECK(cudaMallocAsync(&d_partitions_hash, sizeof(uint64_t), stream)); // hash used to identify repeated states
        CUDA_CHECK(cudaMallocAsync(&d_fail, sizeof(bool), stream));
        thrust::device_ptr<uint32_t> t_partitions_sizes(d_partitions_sizes);
        
        std::set<uint64_t> seen_partitionings;
        
        int consecutive_failures = 0;
        while (h_fail && consecutive_failures < MAX_CONSECUTIVE_INIT_FAILURES) {
            CUDA_CHECK(cudaMemsetAsync(d_partitions_sizes, 0x00, max_parts * sizeof(uint32_t), stream));
            // initial random (valid) partitions
            CUDA_CHECK(cudaMemsetAsync(d_fail, 0x00, sizeof(bool), stream));
            // launch configuration - random init kernel
            threads_per_block = 256;
            num_threads_needed = num_nodes; // 1 thread per node
            blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - random init kernel
            #if VERBOSE_INIT
            std::cout << "Running random init kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") (thread=" << tid << ")...\n";
            #endif
            init_partitions_random<<<blocks, threads_per_block, 0, stream>>>(
                num_nodes,
                INIT_SEED + repeats * MAX_CONSECUTIVE_INIT_FAILURES + consecutive_failures,
                max_parts,
                d_nodes_sizes,
                d_partitions,
                d_partitions_sizes,
                d_fail
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(&h_fail, d_fail, sizeof(bool), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // random initialization failed, try again
            if (h_fail) consecutive_failures++;
        }
        if (consecutive_failures >= MAX_CONSECUTIVE_INIT_FAILURES) continue;

        // =============================
        // print some temporary results
        #if VERBOSE_INIT_LOG
        log_partitions(num_nodes, max_parts, h_max_nodes_per_part, d_partitions, d_partitions_sizes, "Random initial partitioning", stream);
        #endif
        // =============================

        // compute pins per partition
        CUDA_CHECK(cudaMemsetAsync(d_pins_per_partitions, 0x00, num_hedges * max_parts * sizeof(uint32_t), stream));
        // launch configuration - pins per partition kernel
        threads_per_block = 256;
        num_threads_needed = num_hedges; // 1 thread per hedge
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - pins per partition kernel
        #if VERBOSE_INIT
        std::cout << "Running pins per partition kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") (thread=" << tid << ")...\n";
        #endif
        // NOTE: having this available during FM refinement makes its complexity linear in the connectivity, instead of quadratic!
        pins_per_partition_only_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_hedges,
            d_hedges_offsets,
            d_partitions,
            num_hedges,
            max_parts,
            d_pins_per_partitions
        );
        CUDA_CHECK(cudaGetLastError());

        // moves
        uint32_t *d_moves = nullptr;
        float *d_gains = nullptr;
        uint32_t *d_nodes_sizes_aux = nullptr;
        uint32_t *d_move_srcs = nullptr;
        bool *d_continue_flag = nullptr;
        bool h_continue_flag = true;
        CUDA_CHECK(cudaMallocAsync(&d_moves, num_nodes * sizeof(uint32_t), stream)); // moves[idx] -> partition the idx-th node wants to get into
        CUDA_CHECK(cudaMallocAsync(&d_gains, num_nodes * sizeof(float), stream)); // gains[idx] -> connectivity gained by moving the idx-th node
        CUDA_CHECK(cudaMallocAsync(&d_nodes_sizes_aux, num_nodes * sizeof(uint32_t), stream)); // copy of nodes_sizes used to accumulate partition size changes
        CUDA_CHECK(cudaMallocAsync(&d_move_srcs, num_nodes * sizeof(uint32_t), stream)); // move_srcs[idx] -> node proposing the move currently in position idx
        CUDA_CHECK(cudaMallocAsync(&d_continue_flag, sizeof(bool), stream)); // stop when false
        thrust::device_ptr<uint32_t> t_moves(d_moves);
        thrust::device_ptr<float> t_gains(d_gains);
        thrust::device_ptr<uint32_t> t_nodes_sizes_aux(d_nodes_sizes_aux);
        thrust::device_ptr<uint32_t> t_move_srcs(d_move_srcs);

        // repeated application of any in-isolation improving valid move
        for (int jacobi = 0; jacobi < JACOBI_TRIES && h_continue_flag; jacobi++) {
            // every node proposes a valid move (in-isolation only here)
            CUDA_CHECK(cudaMemsetAsync(d_moves, 0xFF, num_nodes * sizeof(uint32_t), stream));
            // launch configuration - jacobi gains / label propagation gains kernel
            // NOTE: choose threads_per_block multiple of WARP_SIZE
            threads_per_block = 128; // 128/32 -> 4 warps per block
            warps_per_block = threads_per_block / WARP_SIZE;
            num_warps_needed = num_nodes ; // 1 warp per node
            blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - jacobi gains kernel
            #if VERBOSE_INIT
            std::cout << "Running jacobi gains kernel (iter=" << jacobi << ") (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") (thread=" << tid << ")...\n";
            #endif
            fm_refinement_gains_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_touching,
                d_touching_offsets,
                d_hedge_weights,
                d_partitions,
                d_pins_per_partitions,
                d_nodes_sizes,
                d_partitions_sizes,
                num_hedges,
                num_nodes,
                max_parts,
                jacobi,
                UINT32_MAX,
                d_moves,
                d_gains
            );
            CUDA_CHECK(cudaGetLastError());

            // sort moves by (dst-part, gain) lexicographically and carry node sizes and proposing nodes along
            CUDA_CHECK(cudaMemcpyAsync(d_nodes_sizes_aux, d_nodes_sizes, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
            thrust::sequence(thrust_exec, t_move_srcs, t_move_srcs + num_nodes);
            auto jacobi_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_moves, t_gains));
            auto jacobi_key_end = jacobi_key_begin + num_nodes;
            auto jacobi_values_begin = thrust::make_zip_iterator(thrust::make_tuple(t_nodes_sizes_aux, t_move_srcs));
            // TODO: maybe also sort w.r.t. the size, putting smaller sizes first, before larger ones, then ordering by gain?
            thrust::sort_by_key(thrust_exec, jacobi_key_begin, jacobi_key_end, jacobi_values_begin, thrust::greater<thrust::tuple<uint32_t, float>>{});
            // inclusive scan inside each key (= partition) on the size of entering nodes
            thrust::inclusive_scan_by_key(thrust_exec, t_moves, t_moves + num_nodes, t_nodes_sizes_aux, t_nodes_sizes_aux);
            // add to each element d_nodes_sizes[i] its d_partitions_sizes[d_moves[i]], to get final partition sizes (iff d_moves is not UINT32_MAX)
            thrust::transform_if(
                thrust_exec,
                thrust::make_zip_iterator(thrust::make_tuple(t_nodes_sizes_aux, t_moves)),
                thrust::make_zip_iterator(thrust::make_tuple(t_nodes_sizes_aux + num_nodes, t_moves + num_nodes)),
                t_moves, t_nodes_sizes_aux,
                [ps = thrust::raw_pointer_cast(t_partitions_sizes)] __device__ (const thrust::tuple<uint32_t, uint32_t>& t) { return thrust::get<0>(t) + ps[thrust::get<1>(t)]; },
                [] __device__ (uint32_t move) { return move != UINT32_MAX; }
            );

            // apply all at once the moves towards every partition until the point that such partition would be full (assuming no node were to leave it)
            CUDA_CHECK(cudaMemsetAsync(d_continue_flag, 0x00, sizeof(bool), stream));
            CUDA_CHECK(cudaMemsetAsync(d_partitions_hash, 0x00, sizeof(uint64_t), stream));
            // launch configuration - jacobi apply kernel
            threads_per_block = 128;
            num_threads_needed = num_nodes; // 1 thread per move to apply
            blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - jacobi apply kernel
            #if VERBOSE_INIT
            std::cout << "Running jacobi apply kernel (iter=" << jacobi << ") (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") (thread=" << tid << ")...\n";
            #endif
            jacobi_apply_moves_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_touching,
                d_touching_offsets,
                d_nodes_sizes,
                d_moves,
                d_nodes_sizes_aux,
                d_move_srcs,
                num_hedges,
                num_nodes,
                max_parts,
                d_partitions,
                d_partitions_sizes,
                d_pins_per_partitions,
                d_continue_flag,
                d_partitions_hash
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(&h_continue_flag, d_continue_flag, sizeof(bool), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&h_partitions_hash, d_partitions_hash, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (!h_continue_flag && jacobi < 4) std::cerr << "WARNING: no valid jacobi moves just after " << jacobi << " iterations (thread=" << tid << ") !!\n";
            if (seen_partitionings.contains(h_partitions_hash)) {
                #if VERBOSE_INIT
                std::cout << "Stopping jacobi in iteration " << jacobi << " for reaching a repeated state (thread=" << tid << ") !\n";
                #endif
                break;
            } else seen_partitionings.insert(h_partitions_hash);
            // =============================
            // print some temporary results
            #if VERBOSE_INIT_CONN
            float log_conn = compute_connectivity(max_parts, num_hedges, d_pins_per_partitions, d_hedge_weights, stream);
            std::cout << "Jacobi iteration " << jacobi << " connectivity: " << std::fixed << std::setprecision(3) << log_conn << " (thread=" << tid << ")\n";
            #endif
            // =============================
        }

        // =============================
        // print some temporary results
        #if VERBOSE_INIT_LOG
        log_partitions(num_nodes, max_parts, h_max_nodes_per_part, d_partitions, d_partitions_sizes, "Post-jacobi partitioning", stream);
        float log_conn = compute_connectivity(max_parts, num_hedges, d_pins_per_partitions, d_hedge_weights, stream);
        std::cout << "Post-jacobi connectivity: " << std::fixed << std::setprecision(3) << log_conn << " (thread=" << tid << ")\n";
        #endif
        // =============================

        h_continue_flag = true;
        // in-sequence move ranks
        uint32_t *d_ranks = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_ranks, num_nodes * sizeof(uint32_t), stream)); // ranks[idx] -> position of the idx-th move in the sorted sequence
        thrust::device_ptr<uint32_t> t_ranks(d_ranks);
        uint32_t *d_fm_indices = nullptr; // support indices for ranks construction
        CUDA_CHECK(cudaMallocAsync(&d_fm_indices, num_nodes * sizeof(uint32_t), stream));
        thrust::device_ptr<uint32_t> t_fm_indices(d_fm_indices);

        // validity size events
        uint32_t *d_size_events_partition = nullptr;
        uint32_t *d_size_events_index = nullptr;
        int32_t *d_size_events_delta = nullptr;
        int32_t *d_valid_moves = nullptr;
        const uint32_t num_size_events = 2 * num_nodes;
        CUDA_CHECK(cudaMallocAsync(&d_size_events_partition, num_size_events * sizeof(uint32_t), stream)); // size_events_partition[ev] -> partition affected by the event
        CUDA_CHECK(cudaMallocAsync(&d_size_events_index, num_size_events * sizeof(uint32_t), stream)); // size_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        CUDA_CHECK(cudaMallocAsync(&d_size_events_delta, num_size_events * sizeof(int32_t), stream)); // size_events_delta[ev] -> size variation brought by the event
        CUDA_CHECK(cudaMallocAsync(&d_valid_moves, num_nodes * sizeof(int32_t), stream)); // valid_move[rank idx] -> 1 if applying all moves up to the idx one in the ordered sequence gives a valid state
        thrust::device_ptr<uint32_t> t_size_events_partition(d_size_events_partition);
        thrust::device_ptr<uint32_t> t_size_events_index(d_size_events_index);
        thrust::device_ptr<int32_t> t_size_events_delta(d_size_events_delta);
        thrust::device_ptr<int32_t> t_valid_moves(d_valid_moves);

        // repeated application of the longest valid improving moves subsequence
        for (int fm = 0; fm < FM_TRIES && h_continue_flag; fm++) {
            // by how much of a node's size to allow an invalid move to be proposed (but filtered later by events - if still invalid)
            uint32_t discount = UINT32_MAX; // fm < FM_TRIES / 3 ? 1u : (fm < 2 * FM_TRIES / 3 ? 2u : UINT32_MAX);

            // every node proposes a valid move in-isolation
            CUDA_CHECK(cudaMemsetAsync(d_moves, 0xFF, num_nodes * sizeof(uint32_t), stream));
            // launch configuration - fm-ref gains kernel
            // NOTE: choose threads_per_block multiple of WARP_SIZE
            threads_per_block = 128; // 128/32 -> 4 warps per block
            warps_per_block = threads_per_block / WARP_SIZE;
            num_warps_needed = num_nodes ; // 1 warp per node
            blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - fm-ref gains kernel
            #if VERBOSE_INIT
            std::cout << "Running fm-ref gains kernel (iter=" << fm << ") (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") (thread=" << tid << ") ...\n";
            #endif
            fm_refinement_gains_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_touching,
                d_touching_offsets,
                d_hedge_weights,
                d_partitions,
                d_pins_per_partitions,
                d_nodes_sizes,
                d_partitions_sizes,
                num_hedges,
                num_nodes,
                max_parts,
                fm,
                discount,
                d_moves,
                d_gains
            );
            CUDA_CHECK(cudaGetLastError());

            // sort gains and build an array of ranks by carrying indices along (node id -> his move's idx in sorted gains) (use node ids as a tie-breaker when sorting)
            thrust::sequence(thrust_exec, t_fm_indices, t_fm_indices + num_nodes);
            thrust::device_ptr<float> t_gains(d_gains);
            auto rank_keys_begin = thrust::make_zip_iterator(thrust::make_tuple(t_gains, t_fm_indices));
            auto rank_keys_end = rank_keys_begin + num_nodes;
            thrust::sort(thrust_exec, rank_keys_begin, rank_keys_end, [] __device__ (const thrust::tuple<float, uint32_t>& a, const thrust::tuple<float, uint32_t>& b) {
                    float sa = thrust::get<0>(a), sb = thrust::get<0>(b);
                    if (sa > sb) return true; // highest score first
                    if (sa < sb) return false;
                    return thrust::get<1>(a) < thrust::get<1>(b); // deterministic tie-break
            });
            thrust::scatter(thrust_exec, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(num_nodes), t_fm_indices, t_ranks); // invert the permutation such that: ranks[original_index] = sorted_position

            /*chaining(
                d_partitions,
                d_moves,
                d_nodes_sizes,
                d_gains,
                num_nodes,
                d_ranks,
                stream
            );*/

            // launch configuration - fm-ref cascade kernel - same as fm-ref gains kernel
            // launch - fm-ref cascade kernel
            #if VERBOSE_INIT
            std::cout << "Running fm-ref cascade kernel (iter=" << fm << ") (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") (thread=" << tid << ") ...\n";
            #endif
            fm_refinement_cascade_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_hedges,
                d_hedges_offsets,
                d_touching,
                d_touching_offsets,
                d_hedge_weights,
                d_ranks,
                d_moves,
                d_partitions,
                d_pins_per_partitions,
                num_hedges,
                num_nodes,
                max_parts,
                d_gains
            );
            CUDA_CHECK(cudaGetLastError());
            // not re-sorting the scores array means you have the array ordered as per the in-isolation gains, but now, this scan updates the scores "as if all previous moves were applied"!
            thrust::inclusive_scan(thrust_exec, t_gains, t_gains + num_nodes, t_gains); // in-place (we don't need scores anymore anyway)

            // compute moves validity by size -> explode each move into two events, one decrementing and incrementing the size of the src and dst partition respectively
            // launch configuration - build size events kernel
            threads_per_block = 128;
            num_threads_needed = num_nodes; // 1 thread per move
            blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - build size events kernel
            #if VERBOSE_INIT
            std::cout << "Running build size events kernel (iter=" << fm << ") (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") (thread=" << tid << ") ...\n";
            #endif
            build_size_events_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_moves,
                d_ranks,
                d_partitions,
                d_nodes_sizes,
                num_nodes,
                d_size_events_partition,
                d_size_events_index,
                d_size_events_delta
            );
            CUDA_CHECK(cudaGetLastError());
            // sort events by (partition, rank) [in lexicographical order for the tuple] and carry size_events_delta along
            auto size_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_size_events_partition, t_size_events_index));
            auto size_events_key_end = size_events_key_begin + num_size_events;
            thrust::sort_by_key(thrust_exec, size_events_key_begin, size_events_key_end, t_size_events_delta);
            // inclusive scan inside each key (= partition) on the event deltas => for each event we get the cumulative size delta for that partition at that point in the sequence
            thrust::inclusive_scan_by_key(thrust_exec, t_size_events_partition, t_size_events_partition + num_size_events, t_size_events_delta, t_size_events_delta);
            // now mark moves that would violate size constraint if the sequence were to end on them
            CUDA_CHECK(cudaMemsetAsync(d_valid_moves, 0u, num_nodes * sizeof(int32_t), stream));
            // launch configuration - flag size events kernel
            threads_per_block = 128;
            num_threads_needed = num_size_events; // 1 thread per event
            blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - flag size events kernel
            #if VERBOSE_INIT
            std::cout << "Running flag size events kernel (iter=" << fm << ") (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") (thread=" << tid << ") ...\n";
            #endif
            flag_size_events_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_size_events_partition,
                d_size_events_index,
                d_size_events_delta,
                d_partitions_sizes,
                num_size_events,
                d_valid_moves
            );
            CUDA_CHECK(cudaGetLastError());
            // compute, as of each event, the cumulative number of partitions that are invalid by summing the count of those made/unmade invalid at each event
            thrust::inclusive_scan(thrust_exec, t_valid_moves, t_valid_moves + num_nodes, t_valid_moves);

            // find the move in the sequence that yields both the highest gain and a valid state (when all moves before it are applied)
            auto idx_begin = thrust::make_counting_iterator<uint32_t>(0);
            // functor masking invalid endpoints in the sequence => invalid moves get a -inf score
            // NOTE: valid_moves => the move is valid when the counter is 0!
            masked_value_functor masked_gains { thrust::raw_pointer_cast(t_gains), thrust::raw_pointer_cast(t_valid_moves) };
            auto masked_begin = thrust::make_transform_iterator(idx_begin, masked_gains);
            auto masked_end = masked_begin + num_nodes;
            // max over valid endpoints only, find the point in the sequence of moves where applying them further never nets a higher gain in a valid state
            auto best_iterator_entry = thrust::max_element(thrust_exec, masked_begin, masked_end);
            const uint32_t best_rank = static_cast<uint32_t>(best_iterator_entry - masked_begin);
            const uint32_t num_good_moves = best_rank + 1; // "+1" to make this the improving moves count, rather than the last improving move's idx
            // validity double-check (if there were no valid moves...)
            int32_t size_validity;
            CUDA_CHECK(cudaMemcpyAsync(&size_validity, d_valid_moves + best_rank, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (size_validity == 0) {
                CUDA_CHECK(cudaMemsetAsync(d_partitions_hash, 0x00, sizeof(uint64_t), stream));
                // launch configuration - fm-ref apply kernel
                threads_per_block = 128;
                num_threads_needed = num_nodes; // 1 thread per move to apply
                blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
                // launch - fm-ref apply kernel
                #if VERBOSE_INIT
                std::cout << "Running fm-ref apply (iter=" << fm << ") (" << num_good_moves << " good moves) kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") (thread=" << tid << ") ...\n";
                #endif
                fm_refinement_apply_update_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_touching,
                    d_touching_offsets,
                    d_moves,
                    d_ranks,
                    d_nodes_sizes,
                    num_hedges,
                    num_nodes,
                    max_parts,
                    num_good_moves,
                    d_partitions,
                    d_partitions_sizes,
                    d_pins_per_partitions,
                    d_partitions_hash
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(&h_partitions_hash, d_partitions_hash, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (seen_partitionings.contains(h_partitions_hash)) {
                    #if VERBOSE_INIT
                    std::cout << "Stopping FM in iteration " << fm << " for reaching a repeated state (thread=" << tid << ") !\n";
                    #endif
                    break;
                } else seen_partitionings.insert(h_partitions_hash);
            } else {
                h_continue_flag = false;
                if (fm < 2) std::cerr << "WARNING: no valid FM moves just after " << fm << " iterations (thread=" << tid << ") !!\n";
            }
            // =============================
            // print some temporary results
            #if VERBOSE_INIT_CONN
            float log_conn = compute_connectivity(max_parts, num_hedges, d_pins_per_partitions, d_hedge_weights, stream);
            std::cout << "FM iteration " << fm << " connectivity: " << std::fixed << std::setprecision(3) << log_conn << " (thread=" << tid << ")\n";
            #endif
            // =============================
        }

        // =============================
        // print some temporary results
        #if VERBOSE_INIT_LOG
        log_partitions(num_nodes, max_parts, h_max_nodes_per_part, d_partitions, d_partitions_sizes, "Post-fm-refinement partitioning", stream);
        #endif
        // =============================

        float conn = compute_connectivity(max_parts, num_hedges, d_pins_per_partitions, d_hedge_weights, stream);

        // update best initial partitioning
        #pragma omp critical
        {
            if (conn < best_conn) {
                best_conn = conn;
                CUDA_CHECK(cudaFreeAsync(d_best_partitions, 0));
                CUDA_CHECK(cudaFreeAsync(d_best_partitions_sizes, 0));
                d_best_partitions = d_partitions;
                d_best_partitions_sizes = d_partitions_sizes;
                #if VERBOSE_INIT
                std::cout << "Updated initial partitioning, connectivity=" << std::fixed << std::setprecision(3) << best_conn << " (thread=" << tid << ")\n";
                #endif
            } else {
                CUDA_CHECK(cudaFreeAsync(d_partitions, stream));
                CUDA_CHECK(cudaFreeAsync(d_partitions_sizes, stream));
                #if VERBOSE_INIT
                std::cout << "Post-fm connectivity (worse than best): " << std::fixed << std::setprecision(3) << conn << " (thread=" << tid << ")\n";
                #endif
            }
        }
        
        CUDA_CHECK(cudaFreeAsync(d_pins_per_partitions, stream));
        CUDA_CHECK(cudaFreeAsync(d_moves, stream));
        CUDA_CHECK(cudaFreeAsync(d_gains, stream));
        CUDA_CHECK(cudaFreeAsync(d_nodes_sizes_aux, stream));
        CUDA_CHECK(cudaFreeAsync(d_move_srcs, stream));
        CUDA_CHECK(cudaFreeAsync(d_continue_flag, stream));
        CUDA_CHECK(cudaFreeAsync(d_ranks, stream));
        CUDA_CHECK(cudaFreeAsync(d_size_events_partition, stream));
        CUDA_CHECK(cudaFreeAsync(d_size_events_index, stream));
        CUDA_CHECK(cudaFreeAsync(d_size_events_delta, stream));
        CUDA_CHECK(cudaFreeAsync(d_valid_moves, stream));
        CUDA_CHECK(cudaFreeAsync(d_fm_indices, stream));
    }

    for (auto s : streams) {
        CUDA_CHECK(cudaStreamDestroy(s));
    }

    if (d_best_partitions == nullptr)
        throw std::runtime_error("Could not randomly construct a valid initial partitioning.");

    std::cout << "Completed initial partitioning, connectivity=" << std::fixed << std::setprecision(3) << best_conn << "\n";
    return std::make_tuple(d_best_partitions, d_best_partitions_sizes);
}