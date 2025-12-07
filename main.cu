#include <iostream>
#include <iomanip>
#include <string>
#include <tuple>

#include </home/mronzani/cuda/include/cuda_runtime.h>
#include </home/mronzani/cuda/include/thrust/device_ptr.h>
#include </home/mronzani/cuda/include/thrust/sequence.h>
#include </home/mronzani/cuda/include/thrust/scatter.h>
#include </home/mronzani/cuda/include/thrust/sort.h>
#include </home/mronzani/cuda/include/thrust/scan.h>
#include </home/mronzani/cuda/include/thrust/device_vector.h>
#include </home/mronzani/cuda/include/thrust/transform.h>

#include "hgraph.hpp"
#include "utils.cuh"

#define DEVICE_ID 0

#define VERBOSE true

extern __global__ void neighborhoods_count_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t num_nodes,
    uint32_t* neighbors_offsets
);

extern __global__ void neighborhoods_scatter_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t num_nodes,
    const uint32_t* neighbors_offsets,
    uint32_t* neighbors
);

extern __global__ void candidates_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* neighbors,
    const uint32_t* neighbor_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    uint32_t* pairs,
    float* scores
);

extern __global__ void grouping_kernel(
    const uint32_t* pairs,
    const float* scores,
    const uint32_t num_nodes,
    slot* group_slots,
    uint32_t* groups
);

extern __global__ void apply_coarsening_hedges_count(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t num_hedges,
    const uint32_t* groups,
    uint32_t* coarse_hedges_offsets
);

extern __global__ void apply_coarsening_hedges_scatter(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t num_hedges,
    const uint32_t* groups,
    const uint32_t* coarse_hedges_offsets,
    uint32_t* coarse_hedges
);

extern __global__ void apply_coarsening_neighbors(
    const uint32_t num_nodes,
    const uint32_t* neighbor_offsets,
    const uint32_t* groups,
    uint32_t* neighbors
);

extern __global__ void apply_coarsening_neighbors_count(
    const uint32_t* neighbors,
    const uint32_t* neighbors_offsets,
    const uint32_t* groups,
    const uint32_t* ungroups,
    const uint32_t* ungroups_offsets,
    const uint32_t num_groups,
    uint32_t* coarse_neighbors_offsets
);

extern __global__ void apply_coarsening_neighbors_scatter(
    const uint32_t* neighbors,
    const uint32_t* neighbors_offsets,
    const uint32_t* groups,
    const uint32_t* ungroups,
    const uint32_t* ungroups_offsets,
    const uint32_t num_groups,
    const uint32_t* coarse_neighbors_offsets,
    uint32_t* coarse_neighbors
);

extern __global__ void apply_coarsening_touching_count(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t num_hedges,
    uint32_t* coarse_touching_offsets
) ;

extern __global__ void apply_coarsening_touching_scatter(
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t* ungroups,
    const uint32_t* ungroups_offsets,
    const uint32_t num_groups,
    const uint32_t* coarse_touching_offsets,
    uint32_t* coarse_touching
);

extern __global__ void apply_uncoarsening_partitions(
    const uint32_t* groups,
    const uint32_t* coarse_partitions,
    const uint32_t num_nodes,
    uint32_t* partitions
);

extern __global__ void pins_per_partition_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* pins_per_partitions
);

extern __global__ void fm_refinement_gains_kernel(
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t* partitions,
    const uint32_t* pins_per_partitions,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    uint32_t* moves,
    float* scores
);

extern __global__ void fm_refinement_cascade_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
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

extern __global__ void fm_refinement_apply_kernel(
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t* moves,
    const uint32_t* move_ranks,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t num_good_moves,
    uint32_t* partitions
    //uint32_t* pins_per_partitions
);

using namespace hgraph;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

void printHelp() {
    std::cout <<
        "Usage:\n"
        "  prog -r <input_file> [-s <output_file>]\n"
        "  prog -h\n\n"
        "Options:\n"
        "  -r <file>   Reload hypergraph from file\n"
        "  -s <file>   Save loaded hypergraph to file\n"
        "  -h          Show this help\n";
}

int main(int argc, char** argv) {
    if (argc == 1) {
        printHelp();
        return 0;
    }

    std::string load_path;
    std::string save_path;

    // CLI handling
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h") { printHelp(); return 0; }
        else if (arg == "-r") {
            if (i + 1 >= argc) { std::cerr << "Error: -r requires a file path\n"; return 1; }
            load_path = argv[++i];
        }
        else if (arg == "-s") {
            if (i + 1 >= argc) { std::cerr << "Error: -s requires a file path\n"; return 1; }
            save_path = argv[++i];
        }
        else { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
    }

    // load hypergraph
    HyperGraph hg(0, {}, {}); // placeholder -> overwritten if "-r" is given
    bool loaded = false;

    if (!load_path.empty()) {
        try {
            hg = HyperGraph::load(load_path);
            loaded = true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading file: " << e.what() << "\n";
            return 1;
        }

        // Print statistics
        float total_freq = 0.0f;
        for (auto& he : hg.hedges())
            total_freq += he.spikeFrequency() * he.connections();

        std::cout << "Loaded hypergraph:\n";
        std::cout << "  Nodes:       " << hg.nodes() << "\n";
        std::cout << "  Hyperedges:  " << hg.hedges().size() << "\n";
        std::cout << "  Total pins:  " << hg.hedgesFlat().size() << "\n";
        std::cout << "  Total Spike Frequency: " << total_freq << "\n";
    }

    // ============================
    // === CUDA STUFF GOES HERE ===

    std::cout << "CUDA device:\n";
    
    // get device properties
    int device_cnt;
    cudaGetDeviceCount(&device_cnt);
    std::cout << "  Found " << device_cnt << " devices: using device " << DEVICE_ID << "\n";
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, DEVICE_ID);
    std::cout << "  Dev. name " << props.name << "\n";
    std::cout << "  Available VRAM: " << std::fixed << std::setprecision(1) << (float)(props.totalGlobalMem) / (1 << 30) << " GB\n";
    std::cout << "  Shared mem. per block: " << std::fixed << std::setprecision(1) << (float)(props.sharedMemPerBlock) / (1 << 10) << " KB\n";
    std::cout << "  Max. grid size: " << props.maxGridSize[0] << " x " << props.maxGridSize[1] << " x " << props.maxGridSize[2] << "\n";
    std::cout << "  Max. block size: " << props.maxThreadsDim[0] << " x " << props.maxThreadsDim[1] << " x " << props.maxThreadsDim[2] << "\n";
    
    std::cout << "Setting up GPU memory...\n";

    uint32_t num_hedges = static_cast<uint32_t>(hg.hedges().size());
    std::vector<uint32_t> hedges_offsets; // hedge idx -> hedge start index in the contiguous hedges array
    hedges_offsets.reserve(num_hedges + 1);

    // prepare hedge offsets
    for (uint32_t i = 0; i < num_hedges; ++i)
        hedges_offsets.push_back(hg.hedges()[i].offset());
    hedges_offsets.push_back(static_cast<uint32_t>(hg.hedgesFlat().size()));

    std::vector<uint32_t> touching_hedges;
    std::vector<uint32_t> touching_hedges_offsets;
    touching_hedges.reserve(hg.hedgesFlat().size()); // with one outbound hedge per node, the total number of pins (e*d) is the total number of connections (n*h)
    touching_hedges_offsets.reserve(hg.nodes() + 1);

    // prepare touching sets
    for (uint32_t n = 0; n < hg.nodes(); ++n) {
        touching_hedges_offsets.push_back(touching_hedges.size());
        for (uint32_t h : hg.outboundIds(n))
            touching_hedges.push_back(h);
        for (uint32_t h : hg.inboundIds(n))
            touching_hedges.push_back(h);
    }
    touching_hedges_offsets.push_back(touching_hedges.size());

    // prepare hyperedge weights
    std::vector<float> hedge_weights(num_hedges);
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedge_weights[i] = hg.hedges()[i].weight();
    }

    // total number of distinct nodes (for output indexing)
    uint32_t num_nodes = hg.nodes(); // nodes count used when allocating outputs
    
    // device pointers
    uint32_t *d_hedges_offsets = nullptr, *d_hedges = nullptr;
    uint32_t *d_neighbors = nullptr, *d_neighbors_offsets = nullptr;
    uint32_t *d_touching = nullptr, *d_touching_offsets = nullptr;
    float *d_hedge_weights = nullptr;
    uint32_t *d_pairs = nullptr;
    slot *d_slots = nullptr;
    float *d_scores = nullptr;

    // kernel dimensions
    int blocks, threads_per_block, warps_per_block;
    int num_threads_needed, num_warps_needed;
    size_t bytes_per_thread, bytes_per_warp, shared_bytes;
    int blocks_per_SM, max_blocks;

    // allocate device memory
    CUDA_CHECK(cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t))); // contigous hedges array
    CUDA_CHECK(cudaMalloc(&d_hedges_offsets, hedges_offsets.size() * sizeof(uint32_t))); // hedges_offsets[hedge idx] -> hedge start idx in d_hedges
    CUDA_CHECK(cudaMalloc(&d_touching, touching_hedges.size() * sizeof(uint32_t))); // contigous inbound+outbout sets array
    CUDA_CHECK(cudaMalloc(&d_touching_offsets, touching_hedges_offsets.size() * sizeof(uint32_t))); // touching_offsets[node idx] -> touching set start idx in d_touching
    CUDA_CHECK(cudaMalloc(&d_hedge_weights, num_hedges * sizeof(float))); // hedge_weights[hedge idx] -> weight
    CUDA_CHECK(cudaMalloc(&d_pairs, num_nodes * sizeof(uint32_t) * MAX_CANDIDATES)); // partitions[node idx] -> best neighbor
    CUDA_CHECK(cudaMalloc(&d_scores, num_nodes * sizeof(float) * MAX_CANDIDATES)); // connection streght for each pair
    CUDA_CHECK(cudaMalloc(&d_slots, num_nodes * sizeof(slot) * MAX_GROUP_SIZE)); // slot to finalize node pairs during grouping (true dtype: "slot")

    // copy to device
    // NOTE: initially with no (multi)function bits!
    CUDA_CHECK(cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedges_offsets, hedges_offsets.data(), hedges_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching, touching_hedges.data(), touching_hedges.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching_offsets, touching_hedges_offsets.data(), touching_hedges_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedge_weights, hedge_weights.data(), num_hedges * sizeof(float), cudaMemcpyHostToDevice));
    
    // prepare neighborhoods
    // NOTE: the host-side version was "hg.buildNeighborhoods()", this is the CUDA version!
    // TODO: this computes (deduplicates) the neighborhoods twice, first to have their size, then to actually write them. We could do better by caching in global memory the already-deduped per-block arrays...
    CUDA_CHECK(cudaMalloc(&d_neighbors_offsets, (num_nodes + 1) * sizeof(uint32_t))); // node -> neighbors set start idx in d_neighbors
    CUDA_CHECK(cudaMemset(d_neighbors_offsets, 0x00, (num_nodes + 1) * sizeof(uint32_t))); // gotta initialize it because some nodes might have zero touching hedges
    blocks = num_nodes;
    threads_per_block = 256; // 256/32 -> 8 warps per block
    std::cout << "Running neighborhoods kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    // count each node's neighbors
    neighborhoods_count_kernel<<<blocks, threads_per_block>>>(
        d_hedges,
        d_hedges_offsets,
        d_touching,
        d_touching_offsets,
        num_nodes,
        d_neighbors_offsets
    );
    // compute final offsets
    thrust::device_ptr<uint32_t> t_neigh_offsets(d_neighbors_offsets);
    thrust::exclusive_scan(t_neigh_offsets, t_neigh_offsets + (num_nodes + 1), t_neigh_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
    uint32_t total_neighbors;
    CUDA_CHECK(cudaMemcpy(&total_neighbors, d_neighbors_offsets + num_nodes, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&d_neighbors, total_neighbors * sizeof(uint32_t))); // contigous neighborhood sets array
    // write neighbors in at their correct offset
    neighborhoods_scatter_kernel<<<blocks, threads_per_block>>>(
        d_hedges,
        d_hedges_offsets,
        d_touching,
        d_touching_offsets,
        num_nodes,
        d_neighbors_offsets,
        d_neighbors
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO: try to apply coarsening to each kernel!

    // returns the number of partitions and the pointer to the final partitions device buffer
    std::function<std::tuple<uint32_t, uint32_t*>(uint32_t, uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*)> coarsen_refine_uncoarsen = [&](
        uint32_t level_idx,
        uint32_t curr_num_nodes,
        uint32_t* d_hedges,
        uint32_t* d_hedges_offsets,
        uint32_t* d_touching,
        uint32_t* d_touching_offsets
    ) { // this is a lambda
        std::cout << "Coarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

        /*
        * Flow:
        * 1) coarsen
        *   - propose candidate node pairs
        *   - group nodes w.r.t. strongest pairs
        *   => if groups are less than the threshold -> return them as the initial partitions
        *   - coarsen all data structures
        * 2) recursive call to the next coarsening level
        *   - returns the coarse partitions
        * 3) uncoarsen
        *   - uncoarsen partitions
        *   - revert to using pre-coarsening data structures (free coarse ones)
        * 4) refinement
        *   - compute pins per partition
        *   - propose refinement moves in isolation
        *   - compute per-move gain as if applied in sequence
        *   - apply the highest-gain subsequence of moves
        *   => return final partitioning to the outer level
        */

        /*
        * Buffers allocated on (and local to) each level:
        * - d_groups
        * - d_ungroups, d_ungroups_offsets
        * - d_pins_per_partitions
        * Buffers constructed anew before (and passed as args to) each level:
        * - d_hedges, d_hedges_offsets
        * - d_touching, d_touching_offsets
        * Buffers (constructed by and) returned from each level:
        * - d_partitions
        * Buffers updated (globally) in-place after each level:
        * - d_pairs
        * - d_scores
        * - d_slots
        * - d_ranks
        * - d_neighbors, d_neighbors_offsets
        * Untouched buffers:
        * - d_hedge_weights
        *
        * TODO:
        * - use (multi)function bits to coarsen/uncoarsen hedges in-place
        * - use (multi)function bits to coarsen/uncoarsen hedges in-place
        * - DO NOT DO THE ABOVE for neighbors (not used during refinement)
        */

        // zero-out candidates kernel's outputs
        // TODO: could just init. up to curr_num_nodes
        CUDA_CHECK(cudaMemset(d_pairs, 0xFF, num_nodes * sizeof(uint32_t) * MAX_CANDIDATES)); // 0xFF -> UINT32_MAX
        // NOTE: no need to init. "d_scores" if we use "d_pairs" to see which locations are valid
        
        // launch configuration - candidates kernel
        // NOTE: choose threads_per_block multiple of WARP_SIZE
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = curr_num_nodes ; // 1 warp per node
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        bytes_per_warp = 0; //TODO
        shared_bytes = warps_per_block * bytes_per_warp;
        // launch - candidates kernel
        std::cout << "Running candidates kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        candidates_kernel<<<blocks, threads_per_block, shared_bytes>>>(
            d_hedges,
            d_hedges_offsets,
            d_neighbors,
            d_neighbors_offsets,
            d_touching,
            d_touching_offsets,
            d_hedge_weights,
            num_hedges,
            curr_num_nodes,
            d_pairs,
            d_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // =============================
        // print some temporary results
        #if VERBOSE
        std::vector<uint32_t> pairs_tmp(curr_num_nodes * MAX_CANDIDATES);
        std::vector<float> scores_tmp(curr_num_nodes * MAX_CANDIDATES);
        std::vector<std::set<uint32_t>> candidates_count(MAX_CANDIDATES);
        CUDA_CHECK(cudaMemcpy(pairs_tmp.data(), d_pairs, curr_num_nodes * sizeof(uint32_t) * MAX_CANDIDATES, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scores_tmp.data(), d_scores, curr_num_nodes * sizeof(float) * MAX_CANDIDATES, cudaMemcpyDeviceToHost));
        std::cout << "Pairing results:";
        for (uint32_t i = 0; i < curr_num_nodes; ++i) {
            if (i < std::min<uint32_t>(curr_num_nodes, 20))
            std::cout << "\n  node " << i << " ->";
            for (uint32_t j = 0; j < MAX_CANDIDATES; ++j) {
                float score = scores_tmp[i * MAX_CANDIDATES + j];
                uint32_t target = pairs_tmp[i * MAX_CANDIDATES + j];
                candidates_count[j].insert(target);
                if (i < std::min<uint32_t>(curr_num_nodes, 20)) {
                    if (target == UINT32_MAX) std::cout << " (target=none score=none) ";
                    else if (target == i) std::cout << " !!SELF TARGETED!! ";
                    else std::cout << " (" << j << " target=" << target << " score=" << std::fixed << std::setprecision(3) << score << ")";
                }
                if (target == UINT32_MAX) continue;
                // check the symmetry invariant: mutual pairs or the other has found a higher score pair (or one with lower id - tiebreaker) [easy for j = 0, for j > 0 check first that the target wasn't already used at a lower j]
                if (pairs_tmp[target * MAX_CANDIDATES + j] != i && pairs_tmp[target * MAX_CANDIDATES + j] != UINT32_MAX && std::find(pairs_tmp.begin() + target * MAX_CANDIDATES, pairs_tmp.begin() + target * MAX_CANDIDATES + j, i) == pairs_tmp.begin() + target * MAX_CANDIDATES + j && !(scores_tmp[target * MAX_CANDIDATES + j] > score || scores_tmp[target * MAX_CANDIDATES + j] == score && pairs_tmp[target * MAX_CANDIDATES + j] < i))
                    std::cout << "\n  WARNING, symmetry violated: node " << i << " (" << j << " target=" << target << " score=" << std::fixed << std::setprecision(3) << score << ") AND node " << target << " (" << j << " target=" << pairs_tmp[target * MAX_CANDIDATES + j] << " score=" << std::fixed << std::setprecision(3) << scores_tmp[target * MAX_CANDIDATES + j] << ") !!";
            }
        }
        std::cout << "\n";
        for (uint32_t j = 0; j < MAX_CANDIDATES; ++j)
        std::cout << "Candidates count (" << j << "): " << candidates_count[j].size() << "\n";
        scores_tmp.clear();
        #endif
        // =============================

        // zero-out grouping kernel's outputs
        slot init_slot; init_slot.id = 0xFFFFFFFFu; init_slot.score = 0u;
        thrust::device_ptr<slot> d_slots_ptr(d_slots);
        // TODO: could lower to just curr_num_nodes
        thrust::fill(d_slots_ptr, d_slots_ptr + num_nodes * MAX_GROUP_SIZE, init_slot); // upper 32 bits to 0x00, lower 32 to 0xFF
        
        // prepare this level's coarsening groups
        uint32_t *d_groups = nullptr;
        CUDA_CHECK(cudaMalloc(&d_groups, curr_num_nodes * sizeof(uint32_t))); // groups[node idx] -> node's group id (zero-based)

        // launch configuration - grouping kernel
        threads_per_block = 256;
        num_threads_needed = curr_num_nodes; // 1 thread per node
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        bytes_per_thread = 0; //TODO
        shared_bytes = threads_per_block * bytes_per_thread;
        // additional checks for the cooperative kernel mode
        blocks_per_SM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_SM, grouping_kernel, threads_per_block, shared_bytes);
        max_blocks = blocks_per_SM * props.multiProcessorCount;
        if (blocks > max_blocks) {
            std::cout << "ABORTING: grouping kernel required blocks=" << blocks << ", but max-blocks=" << max_blocks << " !\n";
            abort();
        }
        // launch - grouping kernel
        std::cout << "Running grouping kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        void *kernel_args[] = {
            (void*)&d_pairs,
            (void*)&d_scores,
            (void*)&curr_num_nodes,
            (void*)&d_slots,
            (void*)&d_groups
        };
        cudaLaunchCooperativeKernel((void*)grouping_kernel, blocks, threads_per_block, kernel_args, shared_bytes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // order groups kernel (parallel label compression)
        // TODO: custom kernel for this?
        // as of now "d_groups" contains the new non-zero-based group id for every node
        thrust::device_vector<uint32_t> t_indices(curr_num_nodes);
        thrust::sequence(t_indices.begin(), t_indices.end());
        // sort by groups, carrying node indices (represented by the sequence) along; after d_groups is sorted, t_indices tells where each sorted element came from
        thrust::device_ptr<uint32_t> t_groups(d_groups);
        thrust::sort_by_key(t_groups, t_groups + curr_num_nodes, t_indices.begin()); // sort groups and carry indices along for a ride
        // build "head of group flags": 1 at first occurrence of each group in the sorted array, 0 otherwise ( flags[i] = 1 if i == 0 or d_groups[i] != d_groups[i-1] )
        thrust::device_vector<uint32_t> t_headflags(curr_num_nodes);
        // the first element is part of groups zero (the initial default)
        t_headflags[0] = 0;
        thrust::transform(t_groups + 1, t_groups + curr_num_nodes, t_groups, t_headflags.begin() + 1, [] __device__ (uint32_t curr, uint32_t prev) { return curr != prev ? 1u : 0u; });
        // the prefix sum of head flags gives the new group id per element (w.r.t. the sorted order) ( new_id[i] = number of heads before position i )
        thrust::inclusive_scan(t_headflags.begin(), t_headflags.end(), t_headflags.begin()); // in-place
        // the last flag, after the scan, gives you the total number of distinct groups
        uint32_t new_num_nodes = t_headflags.back() + 1;
        // scatter the new ids back to original positions using the sequence; for sorted position i, original index is t_indices[i]; we want: d_groups[t_indices[i]] = t_headflags[i]
        thrust::scatter(t_headflags.begin(), t_headflags.end(), t_indices.begin(), t_groups);
        // if the number of groups has reached the required threshold, they become the partitions
        // => now "d_groups[idx]" contains the new zero-based group ID for every node

        // ======================================
        // base case, return inital partitioning
        // TODO: set the threshold
        // TODO: could increase the threshold and instead of "become the partitions" run a host-side robust partitioning algorithm
        //       => what this does now is equivalent to using the coarsening algorithm also as the algorithm to perform the initial partitioning
        if (new_num_nodes <= 4096) {
            // HERE we repurpose the coarsening routine as the routine for initial partitions:
            // - num_partitions = new_num_nodes
            // - partitions = groups

            // NOTE: d_partitions eventually will coincide with the innermost group each node was part of + refinement moves
            //       => the innermost nodes (groups) count is also the number of partitions

            // NOTE: just like groups, partitions need to ordered, as they be used as indices; however, partitions are few, and if one becomes
            //       empty we can just discard its index and leave a few empty spots in the data structures, it's cheaper to compress at the end

            // TODO: call here "apply_coarsening_touching_count" using partitions as groups to compute the initial distinct inbound counts per partition
            // TODO: no need to distinguish inbound count from touching count, just subtract 1 for every node in the partition!
            //       => or even, if we want to go willy nilly about it, just subtract the maximum capacity per partition!

            std::cout << "Initial partitioning built at level " << level_idx << ", remaining nodes=" << curr_num_nodes << ", number of partitions=" << new_num_nodes << "\n";
            
            return std::make_tuple(new_num_nodes, d_groups);
        }
        // base case, failure to coarsen further
        if (new_num_nodes == curr_num_nodes) {
            std::cout << "FAILED TO COARSEN FURTHER at level " << level_idx << ", remaining nodes=" << curr_num_nodes << "=number of partitions=" << new_num_nodes << "\n";
            std::cout << "WARNING: falling back to returning current groups as individual partitions...\n";
            return std::make_tuple(new_num_nodes, d_groups);
        }
        // ======================================

        // prepare this level's uncoarsening data structures
        uint32_t *d_ungroups = nullptr, *d_ungroups_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ungroups, curr_num_nodes * sizeof(uint32_t))); // ungroups[ungroups_offsets[group id] + i] -> the group's i-th node (its original idx)
        CUDA_CHECK(cudaMalloc(&d_ungroups_offsets, (1 + new_num_nodes) * sizeof(uint32_t))); // ungroups_offsets[node idx] -> node's group id (zero-based)
        
        // build reverse multifunction from groups to their original nodes
        // from above, t_indices is the list of node idxs sorted by their group id, hence, the reverse list is simply t_indices, we just need to compute the offsets to reach, from each group id, its original nodes
        CUDA_CHECK(cudaMemcpy(d_ungroups, thrust::raw_pointer_cast(t_indices.data()), curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        thrust::device_ptr<uint32_t> t_ungroups_offsets(d_ungroups_offsets);
        // predicate to detect group starts: is_group_start(i) = (i == 0) || (headflags[i] != headflags[i-1])
        //auto is_group_start = [groups = t_groups] __device__ (uint32_t i) { return (i == 0) || (groups[i] != groups[i - 1]); }; // WRONG: we should use heads!
        auto is_group_start = [heads = t_headflags.begin()] __device__ (uint32_t i) { return (i == 0) || (heads[i] != heads[i - 1]); };
        // counting iterator over sorted positions
        auto t_iter_begin = thrust::make_counting_iterator<uint32_t>(0);
        auto t_iter_end = thrust::make_counting_iterator<uint32_t>(curr_num_nodes);
        // copy positions of (only) group starts directly into ungroups_offsets
        thrust::copy_if(t_iter_begin, t_iter_end, t_iter_begin, t_ungroups_offsets, is_group_start);
        // append the (curr_num_nodes + 1)-th value
        CUDA_CHECK(cudaMemcpy(d_ungroups_offsets + new_num_nodes, &curr_num_nodes, sizeof(uint32_t), cudaMemcpyHostToDevice));
        // free up thrust vectors
        //thrust::device_vector<uint32_t>().swap(t_indices); // DO NOT FREE THIS UP! We need it later for REFINEMENT!
        thrust::device_vector<uint32_t>().swap(t_headflags);

        // =============================
        // print some temporary results
        #if VERBOSE
        std::vector<uint32_t> groups_tmp(curr_num_nodes);
        CUDA_CHECK(cudaMemcpy(pairs_tmp.data(), d_pairs, curr_num_nodes * sizeof(uint32_t) * MAX_CANDIDATES, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(groups_tmp.data(), d_groups, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::unordered_map<uint32_t, int> groups_count;
        std::cout << "Grouping results:\n";
        for (uint32_t i = 0; i < curr_num_nodes; ++i) {
            uint32_t group = groups_tmp[i];
            groups_count[group]++;
            if (i < std::min<uint32_t>(curr_num_nodes, 20)) {
                std::cout << "  node " << i << " ->";
                for (uint32_t j = 0; j < MAX_CANDIDATES; ++j) {
                    uint32_t target = pairs_tmp[i * MAX_CANDIDATES + j];
                    if (target == UINT32_MAX) std::cout << " (" << j << " target=none)";
                    else std::cout << " (" << j << " target=" << target << ")";
                }
                std::cout << " group=" << group << "\n";
            }
        }
        int max_gs = groups_count.empty() ? 0 : std::max_element(groups_count.begin(), groups_count.end(), [](auto &a, auto &b){ return a.second < b.second; })->second;
        std::cout << "Groups count: " << groups_count.size() << ", Max group size: " << max_gs << "\n";
        pairs_tmp.clear();
        groups_tmp.clear();
        #endif
        // =============================

        // prepare coarse hedges buffers
        uint32_t *d_coarse_hedges = nullptr, *d_coarse_hedges_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_coarse_hedges_offsets, (1 + num_hedges) * sizeof(uint32_t))); // NOTE: the number of hedges never decreases (for now), unlike that of nodes!
        CUDA_CHECK(cudaMemset(d_coarse_hedges_offsets, 0x00, sizeof(uint32_t))); // init. the first offset at 0
        // launch configuration - coarsening kernel (hedges - both)
        threads_per_block = 128;
        num_threads_needed = num_hedges; // 1 thread per hedge
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - coarsening kernel (hedges - count)
        std::cout << "Running coarsening kernel (hedges - count) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_hedges_count<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            num_hedges,
            d_groups,
            d_coarse_hedges_offsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        thrust::device_ptr<uint32_t> t_coarse_hedges_offsets(d_coarse_hedges_offsets);
        // NOTE: the scan wants the last index EXCLUDED, while the memcopy wants the last index exactly! That's why we use here the +1, and not later!
        thrust::inclusive_scan(t_coarse_hedges_offsets, t_coarse_hedges_offsets + (num_hedges + 1), t_coarse_hedges_offsets); // in-place exclusive scan (the last element collects the full reduce)
        uint32_t new_hedges_size = 0; // last value in the inclusive scan = full reduce = total number of pins among all hedges
        CUDA_CHECK(cudaMemcpy(&new_hedges_size, d_coarse_hedges_offsets + num_hedges, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMalloc(&d_coarse_hedges, new_hedges_size * sizeof(uint32_t)));
        // launch - coarsening kernel (hedges - scatter)
        std::cout << "Running coarsening kernel (hedges - scatter) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_hedges_scatter<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            num_hedges,
            d_groups,
            d_coarse_hedges_offsets,
            d_coarse_hedges
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t *d_coarse_neighbors = nullptr, *d_coarse_neighbors_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_coarse_neighbors_offsets, (1 + new_num_nodes) * sizeof(uint32_t))); // NOTE: the number nodes decreases!
        CUDA_CHECK(cudaMemset(d_coarse_neighbors_offsets, 0x00, sizeof(uint32_t))); // init. the first offset at 0
        // launch configuration - coarsening kernel (neighbors - count)
        threads_per_block = 128;
        num_threads_needed = new_num_nodes; // 1 thread per group
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - coarsening kernel (neighbors - count)
        std::cout << "Running coarsening kernel (neighbors - count) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_neighbors_count<<<blocks, threads_per_block>>>(
            d_neighbors,
            d_neighbors_offsets,
            d_groups,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            d_coarse_neighbors_offsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        thrust::device_ptr<uint32_t> t_coarse_neighbors_offsets(d_coarse_neighbors_offsets);
        thrust::inclusive_scan(t_coarse_neighbors_offsets, t_coarse_neighbors_offsets + (new_num_nodes + 1), t_coarse_neighbors_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
        uint32_t new_neighbors_size = 0; // last value in the inclusive scan = full reduce = total number of neighbors among all sets
        CUDA_CHECK(cudaMemcpy(&new_neighbors_size, d_coarse_neighbors_offsets + new_num_nodes, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMalloc(&d_coarse_neighbors, new_neighbors_size * sizeof(uint32_t)));
        // launch configuration - coarsening kernel (neighbors - scatter)
        threads_per_block = 128;
        num_threads_needed = new_num_nodes; // 1 thread per group
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - coarsening kernel (neighbors - scatter)
        std::cout << "Running coarsening kernel (neighbors - scatter) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_neighbors_scatter<<<blocks, threads_per_block>>>(
            d_neighbors,
            d_neighbors_offsets,
            d_groups,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            d_coarse_neighbors_offsets,
            d_coarse_neighbors
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // de-allocate old neighbors and replace them with coarse ones
        CUDA_CHECK(cudaFree(d_neighbors));
        CUDA_CHECK(cudaFree(d_neighbors_offsets));
        d_neighbors = d_coarse_neighbors;
        d_neighbors_offsets = d_coarse_neighbors_offsets;

        uint32_t *d_coarse_touching = nullptr, *d_coarse_touching_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_coarse_touching_offsets, (1 + new_num_nodes) * sizeof(uint32_t))); // NOTE: the number nodes decreases!
        CUDA_CHECK(cudaMemset(d_coarse_touching_offsets, 0x00, (1 + new_num_nodes) * sizeof(uint32_t))); // remember to leave the first offset at 0
        // launch configuration - coarsening kernel (touching - count)
        threads_per_block = 128;
        num_threads_needed = num_hedges; // 1 thread per hedge
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - coarsening kernel (touching - count)
        std::cout << "Running coarsening kernel (touching - count) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_touching_count<<<blocks, threads_per_block>>>(
            d_coarse_hedges,
            d_coarse_hedges_offsets,
            num_hedges,
            d_coarse_touching_offsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        thrust::device_ptr<uint32_t> t_coarse_touching_offsets(d_coarse_touching_offsets);
        thrust::inclusive_scan(t_coarse_touching_offsets, t_coarse_touching_offsets + (new_num_nodes + 1), t_coarse_touching_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
        uint32_t new_touching_size = 0; // last value in the inclusive scan = full reduce = total number of touching hedges among all sets
        CUDA_CHECK(cudaMemcpy(&new_touching_size, d_coarse_touching_offsets + new_num_nodes, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMalloc(&d_coarse_touching, new_touching_size * sizeof(uint32_t)));
        // launch configuration - coarsening kernel (touching - scatter)
        threads_per_block = 128;
        num_threads_needed = new_num_nodes; // 1 thread per group
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - coarsening kernel (touching - scatter)
        std::cout << "Running coarsening kernel (touching - scatter) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_touching_scatter<<<blocks, threads_per_block>>>(
            d_touching,
            d_touching_offsets,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            d_coarse_touching_offsets,
            d_coarse_touching
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ALTERANTIVE PLAN, RE-BUILD NEIGHBORS FROM SCRATCH:
        // NOTE: already tested to be logially equivalent with the above coarsening routine for neighbors!
        /*CUDA_CHECK(cudaMemset(d_neighbors_offsets, 0x00, (new_num_nodes + 1) * sizeof(uint32_t))); // gotta initialize it because some nodes might have zero touching hedges
        blocks = new_num_nodes;
        threads_per_block = 256; // 256/32 -> 8 warps per block
        std::cout << "Running neighborhoods kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // count each node's neighbors
        neighborhoods_count_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            d_touching,
            d_touching_offsets,
            new_num_nodes,
            d_neighbors_offsets
        );
        // compute final offsets
        thrust::device_ptr<uint32_t> t_neigh_offsets(d_neighbors_offsets);
        thrust::exclusive_scan(t_neigh_offsets, t_neigh_offsets + (new_num_nodes + 1), t_neigh_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
        // write neighbors in at their correct offset
        neighborhoods_scatter_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            d_touching,
            d_touching_offsets,
            new_num_nodes,
            d_neighbors_offsets,
            d_neighbors
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());*/

        // ======================================
        // recursive call, go down one more level
        auto [num_partitions, d_coarse_partitions] = coarsen_refine_uncoarsen(
            level_idx + 1,
            new_num_nodes,
            d_coarse_hedges,
            d_coarse_hedges_offsets,
            d_coarse_touching,
            d_coarse_touching_offsets
        );
        // ======================================

        std::cout << "Uncoarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

        // prepare this level's uncoarsened partitions
        uint32_t *d_partitions = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partitions, curr_num_nodes * sizeof(uint32_t)));

        // launch configuration - uncoarsening kernel (partitions)
        // uncoarsen d_coarse_partitions into d_partitions
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = curr_num_nodes ; // 1 warp per node
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - uncoarsening kernel (partitions)
        std::cout << "Running uncoarsening kernel (partitions) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_uncoarsening_partitions<<<blocks, threads_per_block>>>(
            d_groups,
            d_coarse_partitions,
            curr_num_nodes,
            d_partitions
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // cleanup groups
        CUDA_CHECK(cudaFree(d_groups));
        CUDA_CHECK(cudaFree(d_ungroups));
        CUDA_CHECK(cudaFree(d_ungroups_offsets));
        CUDA_CHECK(cudaFree(d_coarse_hedges));
        CUDA_CHECK(cudaFree(d_coarse_hedges_offsets));
        CUDA_CHECK(cudaFree(d_coarse_touching));
        CUDA_CHECK(cudaFree(d_coarse_touching_offsets));
        CUDA_CHECK(cudaFree(d_coarse_partitions)); // allocated at the next inner level, freed here!

        std::cout << "Refining level " << level_idx << ", remaining nodes=" << curr_num_nodes << "number of partitions=" << num_partitions << "\n";

        // prepare this level's pins per partition
        uint32_t *d_pins_per_partitions = nullptr;
        CUDA_CHECK(cudaMalloc(&d_pins_per_partitions, num_hedges * num_partitions * sizeof(uint32_t))); // hedge * num_partitions + partition -> count of pins of "hedge" in that "partition"
        CUDA_CHECK(cudaMemset(d_pins_per_partitions, 0x00, num_hedges * num_partitions * sizeof(uint32_t)));

        // launch configuration - pins per partition kernel
        // TODO: do we really need to recompute pins per partition, or can we update them in-place?
        // PITFALL: we don't know which node of a group was with which hedge...
        // SOLUTION: we could update pins-per-partition as we uncoarsen hedges in-place, if we do that with (multi)function bits!
        // => If we do this, uncomment "pins_per_partitions" in "fm_refinement_apply_kernel"
        threads_per_block = 256;
        num_threads_needed = num_hedges; // 1 thread per hedge
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        bytes_per_thread = 0; //TODO
        shared_bytes = threads_per_block * bytes_per_thread;
        // launch - pins per partition kernel
        std::cout << "Running pins per partition kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // NOTE: having this available during FM refinement makes its complexity linear in the connectivity, instead of quadratic!
        pins_per_partition_kernel<<<blocks, threads_per_block, shared_bytes>>>(
            d_hedges,
            d_hedges_offsets,
            d_partitions,
            num_hedges,
            num_partitions,
            d_pins_per_partitions
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // zero-out fm-ref gains kernel's outputs
        // TODO: could lower to just curr_num_nodes...
        CUDA_CHECK(cudaMemset(d_pairs, 0xFF, num_nodes * sizeof(uint32_t))); // 0xFF -> UINT32_MAX
        // NOTE: no need to init. "d_scores" if we use "d_pairs" to see which locations are valid

        // launch configuration - fm-ref gains kernel
        // NOTE: choose threads_per_block multiple of WARP_SIZE
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = curr_num_nodes ; // 1 warp per node
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        bytes_per_warp = 0; //TODO
        shared_bytes = warps_per_block * bytes_per_warp;
        // launch - fm-ref gains kernel
        std::cout << "Running fm-ref gains kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        fm_refinement_gains_kernel<<<blocks, threads_per_block, shared_bytes>>>(
            d_touching,
            d_touching_offsets,
            d_hedge_weights,
            d_partitions,
            d_pins_per_partitions,
            num_hedges,
            curr_num_nodes,
            num_partitions,
            // NOTE: repurposing those from the candidates kernel!
            d_pairs, // -> moves
            d_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // sort scores and build an array of ranks (node id -> his move's idx in sorted scores)
        //thrust::device_vector<int> t_indices(curr_num_nodes); // temporary sequence sorted alongside scores -> ALREADY DECLARED FOR COARSEING, reuse!
        thrust::sequence(t_indices.begin(), t_indices.end());
        uint32_t *d_ranks = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ranks, curr_num_nodes * sizeof(uint32_t))); // node -> number of touching hedges seen as of now
        thrust::device_ptr<uint32_t> t_ranks(d_ranks);
        thrust::device_ptr<float> t_scores(d_scores);
        thrust::sort_by_key(d_scores, d_scores + curr_num_nodes, t_indices.begin()); // sort scores according to scores themselves and indices in the same way
        thrust::scatter(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(curr_num_nodes), t_indices.begin(), t_ranks); // invert the permutation such that: ranks[original_index] = sorted_position
        // free up thrust vectors
        thrust::device_vector<uint32_t>().swap(t_indices);
        // launch configuration - fm-ref cascade kernel => same as "fm-ref gains kernel"
        // compute shared memory per block (bytes)
        bytes_per_warp = 0; //TODO
        shared_bytes = warps_per_block * bytes_per_warp;
        // launch - fm-ref cascade kernel
        std::cout << "Running fm-ref gains kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        fm_refinement_cascade_kernel<<<blocks, threads_per_block, shared_bytes>>>(
            d_hedges,
            d_hedges_offsets,
            d_touching,
            d_touching_offsets,
            d_hedge_weights,
            d_ranks,
            d_pairs,
            d_partitions,
            d_pins_per_partitions,
            num_hedges,
            curr_num_nodes,
            num_partitions,
            d_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // not re-sorting the scores array means you have the array ordered as per the initial scores,
        // but now, this scan updates the scores "as if all previous moves were applied"!
        thrust::inclusive_scan(t_scores, t_scores + curr_num_nodes, t_scores); // in-place (we don't need scores anymore anyway)
        auto iter_max_scores = thrust::max_element(t_scores, t_scores + curr_num_nodes); // find the point in the sequence of moves where applying them further never nets a higher gain
        uint32_t num_good_moves = iter_max_scores - t_scores + 1; // "+1" to make this the improving moves count, rather than the last improving move's idx
        // launch configuration - fm-ref apply kernel
        threads_per_block = 128;
        num_threads_needed = curr_num_nodes; // 1 thread per move to apply
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - fm-ref apply kernel
        fm_refinement_apply_kernel<<<blocks, threads_per_block, shared_bytes>>>(
            d_touching,
            d_touching_offsets,
            d_pairs,
            d_ranks,
            num_hedges,
            curr_num_nodes,
            num_partitions,
            num_good_moves,
            d_partitions
            //d_pins_per_partitions
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_ranks));
        CUDA_CHECK(cudaFree(d_pins_per_partitions));

        // required kernels:
        // - uncoarsen touching -> set the upper bits of the hedge id in one-hot encoding for the node in the ...
        // - NO NEED TO uncoarsen neighbors -> not used for refinement
        // - uncoarsen hedges
        // - uncoarsen partitions
        // - uncoarsen pins-per-partition (compute or update? complexity is O(e*d) = O(n*h) regardless, but updating requires atomics... => if recompute, remove it from fm_refinement_apply_kernel!)
        // NOTE: no need for (multi)function bits! We never truly invert groups aside from expanding partitions!
        /*
        * To uncoarsen partitions just use "d_groups" like this:
        * - build a two lists: "d_ungroup" of lenth curr_num_nodes and "d_ungroups_offsets" of length new_num_nodes + 1 such that "d_ungroups_offsets"
        *   is indexed by the group id, and tells you the index in "d_ungroup" where you can start finding the node ids for that group;
        *   can be build by sorting and then doing a scan over "d_groups"; or better, can be build by counting the occurrencies of each group;
        *   this can be built as a side-job while constructing "d_groups"!
        * - create a "new_partitions" buffer, and spawn a thread per group id, that goes and writes to the entries in "new_partitions" for each of
        *   the node ids in "d_ungroup" between "d_ungroups_offsets[group_idx]" and "d_ungroups_offsets[group_idx + 1]".
        */

        return std::make_tuple(num_partitions, d_partitions);
    };

    // START: the multi-leve recursive refinement routine, down we go!
    auto [num_partitions, d_partitions] = coarsen_refine_uncoarsen(
        0, // first level
        num_nodes,
        d_hedges,
        d_hedges_offsets,
        d_touching,
        d_touching_offsets
    );

    // TO FINISH UP!

    // TODO: make d_partitions zero-based again, if we emptied some partitions...

    // copy back results
    std::vector<uint32_t> partitions(num_nodes);
    CUDA_CHECK(cudaMemcpy(partitions.data(), d_partitions, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // print some example outputs
    std::set<uint32_t> part_count;
    std::cout << "Results:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        uint32_t part = partitions[i];
        part_count.insert(part);
        if (i < std::min<uint32_t>(num_nodes, 20)) {
            if (part == UINT32_MAX) std::cout << "node " << i << " -> part=none\n";
            else std::cout << "node " << i << " ->" << " part=" << part << "\n";
        }
    }
    std::cout << "Partitions count: " << part_count.size() << "(" << num_partitions - part_count.size() << " empty)" << "\n";

    // cleanup device memory
    CUDA_CHECK(cudaFree(d_hedges));
    CUDA_CHECK(cudaFree(d_hedges_offsets));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_neighbors_offsets));
    CUDA_CHECK(cudaFree(d_touching));
    CUDA_CHECK(cudaFree(d_touching_offsets));
    CUDA_CHECK(cudaFree(d_hedge_weights));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_partitions));

    // TODO: apply the partitioning!
    // TODO: print the initial and final weight: hg.totalWeight(); !!

    // === CUDA STUFF ENDS HERE ===
    // ============================

    // save hypergraph
    if (!save_path.empty()) {
        if (!loaded) {
            std::cerr << "Error: -s used without loading a hypergraph first.\n";
            return 1;
        }
        try {
            // TODO: apply the partitioning before saving!
            hg.save(save_path);
            std::cout << "Saved to " << save_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error saving file: " << e.what() << "\n";
            return 1;
        }
    }

    return 0;
}
