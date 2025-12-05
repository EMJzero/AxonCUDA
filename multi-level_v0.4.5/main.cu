#include <iostream>
#include <iomanip>
#include <string>

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
    const uint32_t* hedge_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t num_nodes,
    uint32_t* neighbors_offsets
);

extern __global__ void neighborhoods_scatter_kernel(
    const uint32_t* hedges,
    const uint32_t* hedge_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const uint32_t num_nodes,
    const uint32_t* neighbors_offsets,
    uint32_t* neighbors
);

extern __global__ void candidates_kernel(
    const uint32_t* hedges,
    const uint32_t* hedge_offsets,
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
    slot* group_slots
);

extern __global__ void apply_coarsening_hedges(
    const uint32_t num_hedges,
    const uint32_t* hedge_offsets,
    const uint32_t* groups,
    uint32_t* hedges
);

extern __global__ void apply_coarsening_neighbors(
    const uint32_t num_nodes,
    const uint32_t* neighbor_offsets,
    const uint32_t* groups,
    uint32_t* neighbors
);

extern __global__ void apply_coarsening_touching_count(
    const uint32_t* hedges,
    const uint32_t* hedge_offsets,
    const uint32_t num_hedges,
    const uint32_t* groups,
    uint32_t* touching_offsets
);

extern __global__ void apply_coarsening_touching_scatter(
    const uint32_t* hedges,
    const uint32_t* hedge_offsets,
    const uint32_t num_hedges,
    const uint32_t* groups,
    const uint32_t* touching_offsets,
    uint32_t* touching,
    uint32_t* touching_counter
);

extern __global__ void pins_per_partition_kernel(
    const uint32_t* hedges,
    const uint32_t* hedge_offsets,
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
    const uint32_t* hedge_offsets,
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
    uint32_t* partitions,
    uint32_t* pins_per_partitions
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
    std::vector<uint32_t> hedge_offsets; // hedge idx -> hedge start index in the contiguous hedges array
    hedge_offsets.reserve(num_hedges + 1);

    // prepare hedge offsets
    for (uint32_t i = 0; i < num_hedges; ++i)
        hedge_offsets.push_back(hg.hedges()[i].offset());
    hedge_offsets.push_back(static_cast<uint32_t>(hg.hedgesFlat().size()));

    std::vector<uint32_t> touching_hedges;
    std::vector<uint32_t> touching_hedge_offsets;
    touching_hedges.reserve(hg.hedgesFlat().size()); // with one outbound hedge per node, the total number of pins (e*d) is the total number of connections (n*h)
    touching_hedge_offsets.reserve(hg.nodes() + 1);

    // prepare touching sets
    for (uint32_t n = 0; n < hg.nodes(); ++n) {
        touching_hedge_offsets.push_back(touching_hedges.size());
        for (uint32_t h : hg.outboundIds(n))
            touching_hedges.push_back(h);
        for (uint32_t h : hg.inboundIds(n))
            touching_hedges.push_back(h);
    }
    touching_hedge_offsets.push_back(touching_hedges.size());

    // optional per-hyperedge weight array (if you want weighted increments)
    std::vector<float> hedge_weight_host(num_hedges);
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedge_weight_host[i] = hg.hedges()[i].weight();
    }

    // total number of distinct nodes (for output indexing)
    uint32_t num_nodes = hg.nodes(); // nodes count used when allocating outputs
    
    // device pointers
    uint32_t *d_offsets = nullptr, *d_hedges = nullptr;
    uint32_t *d_neighbors = nullptr, *d_neighbors_offsets = nullptr;
    uint32_t *d_touching = nullptr, *d_touching_offsets = nullptr;
    float *d_hedge_weights = nullptr;
    uint32_t *d_pairs = nullptr;
    slot *d_slots = nullptr;
    float *d_scores = nullptr;
    // coincide with the innermost group each node was part of + refinement moves
    // => the innermost nodes (groups) count is also the number of partitions
    uint32_t *d_partitions = nullptr, d_pins_per_partitions = nullptr;

    // kernel dimensions
    int blocks, threads_per_block, warps_per_block;
    int num_threads_needed, num_warps_needed;
    size_t bytes_per_thread, bytes_per_warp, shared_bytes;
    int blocks_per_SM, max_blocks;

    // allocate device memory (use sizeof instead of *4)
    CUDA_CHECK(cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t))); // contigous hedges array
    CUDA_CHECK(cudaMalloc(&d_offsets, hedge_offsets.size() * sizeof(uint32_t))); // hedge id -> hedge start idx in d_hedges
    CUDA_CHECK(cudaMalloc(&d_touching, touching_hedges.size() * sizeof(uint32_t))); // contigous inbound+outbout sets array
    CUDA_CHECK(cudaMalloc(&d_touching_offsets, touching_hedge_offsets.size() * sizeof(uint32_t))); // node -> touching set start idx in d_touching
    CUDA_CHECK(cudaMalloc(&d_hedge_weights, num_hedges * sizeof(float))); // hedge -> weight
    CUDA_CHECK(cudaMalloc(&d_pairs, num_nodes * sizeof(uint32_t) * MAX_CANDIDATES)); // node -> best neighbor
    CUDA_CHECK(cudaMalloc(&d_scores, num_nodes * sizeof(float) * MAX_CANDIDATES)); // connection streght for each pair
    CUDA_CHECK(cudaMalloc(&d_slots, num_nodes * sizeof(slot) * MAX_GROUP_SIZE)); // slot to finalize node pairs during grouping (true dtype: "slot")
    CUDA_CHECK(cudaMalloc(&d_partitions, num_nodes * sizeof(uint32_t))); // node -> partition id

    // copy to device
    // NOTE: initially with no (multi)function bits!
    CUDA_CHECK(cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, hedge_offsets.data(), hedge_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching, touching_hedges.data(), touching_hedges.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching_offsets, touching_hedge_offsets.data(), touching_hedge_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedge_weights, hedge_weight_host.data(), num_hedges * sizeof(float), cudaMemcpyHostToDevice));
    
    // zero-out final outputs
    CUDA_CHECK(cudaMemset(d_partitions, 0xFF, num_nodes * sizeof(uint32_t))); // 0xFF... -> UINT32_MAX
    
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
        d_offsets,
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
        d_offsets,
        d_touching,
        d_touching_offsets,
        num_nodes,
        d_neighbors_offsets,
        d_neighbors
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO: try to apply coarsening to each kernel!

    // returns the number of partitions
    std::function<uint32_t(uint32_t,uint32_t)> coarsen_refine_uncoarsen = [&](uint32_t level_idx, uint32_t curr_num_nodes) { // this is a lambda
        std::cout << "Coarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

        /*
        * Note:
        * - buffers updated in place (not needed after coarsening or only needed during uncoarsening):
        * - hedges
        * - neighbors
        * - partitions
        * - a level just needs to allocate / build:
        *   - a new node -> group map
        *   - a new touching array of sets and its offsets
        */

        // prepare this level's groups
        uint32_t *d_groups = nullptr;
        CUDA_CHECK(cudaMalloc(&d_groups, curr_num_nodes * sizeof(uint32_t)));
        // TODO: thrust sequence?
        CUDA_CHECK(cudaMemset(d_groups, 0x00, curr_num_nodes * sizeof(uint32_t)));

        // zero-out this kernel's outputs
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
            d_offsets,
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
        #if !VERBOSE
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
                    else std::cout << " (" << j << " target=" << target << " score=" << std::fixed << std::setprecision(3) << score << ") ";
                }
            }
        }
        std::cout << "\n";
        for (uint32_t j = 0; j < MAX_CANDIDATES; ++j)
            std::cout << "Candidates count (" << j << "): " << candidates_count[j].size() << "\n";
        #endif
        // =============================

        // zero-out this kernel's outputs
        slot init_slot; init_slot.id = 0xFFFFFFFFu; init_slot.score = 0u;
        thrust::device_ptr<slot> d_slots_ptr(d_slots);
        thrust::fill(d_slots_ptr, d_slots_ptr + num_nodes * MAX_GROUP_SIZE, init_slot); // upper 32 bits to 0x00, lower 32 to 0xFF

        // launch configuration - grouping kernel
        threads_per_block = 256;
        num_threads_needed = curr_num_nodes; // 1 thread per node
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        bytes_per_thread = 0; //TODO
        shared_bytes = threads_per_block * bytes_per_thread;
        // additional checks for the cooperative kernel mode
        blocks_per_SM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_SM, grouping_kernel, threads_per_block, DEVICE_ID);
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
            (void*)&d_slots
        };
        cudaLaunchCooperativeKernel((void*)grouping_kernel, blocks, threads_per_block, kernel_args, shared_bytes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // order groups kernel (parallel label compression)
        // TODO: custom kernel for this?
        // NOTE: overwrite "groups" with a map node -> new node id
        thrust::device_vector<uint32_t> t_indices(curr_num_nodes);
        thrust::sequence(t_indices.begin(), t_indices.end());
        // sort by groups, carrying node indices (represented by the sequence) along; after d_groups is sorted, t_indices tells where each sorted element came from
        thrust::device_ptr<uint32_t> t_groups(d_groups);
        thrust::sort_by_key(t_groups, t_groups + curr_num_nodes, t_indices.begin()); // sort groups and carry indices along for a ride
        // build "head of group flags": 1 at first occurrence of each group in the sorted array, 0 otherwise ( flags[i] = 1 if i == 0 or d_groups[i] != d_groups[i-1] )
        thrust::device_vector<uint32_t> t_headflags(curr_num_nodes);
        // the first element is always a the head of the first group
        t_headflags[0] = 1;
        thrust::transform(t_groups + 1, t_groups + curr_num_nodes, t_groups, t_headflags.begin() + 1, [] __device__ (uint32_t curr, uint32_t prev) { return curr != prev ? 1u : 0u; });
        // the prefix sum of head flags gives the new group id per element (w.r.t. the sorted order) ( new_id[i] = number of heads before position i )
        thrust::exclusive_scan(t_headflags.begin(), t_headflags.end(), t_headflags.begin()); // in-place
        // the last flag, after the scan, gives you the total number of distinct groups
        uint32_t new_num_nodes = t_headflags.back() + 1;
        // scatter the new ids back to original positions using the sequence; for sorted position i, original index is t_indices[i]; we want: d_groups[t_indices[i]] = t_headflags[i]
        thrust::scatter(t_headflags.begin(), t_headflags.end(), t_indices.begin(), t_groups);
        // if the number of groups has reached the required threshold, they become the partitions

        // ======================================
        // base case, return inital partitioning
        // TODO: set the threshold
        // TODO: could increase the threshold and instead of "become the partitions" run a host-side robust partitioning algorithm
        //       => what this does now is equivalent to using the coarsening algorithm also as the algorithm to perform the initial partitioning
        if (new_num_nodes <= 1) {
            // here new_num_nodes = num_partitions
            CUDA_CHECK(cudaMemcpy(d_groups, d_partitions, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaFree(d_groups));

            CUDA_CHECK(cudaMalloc(&d_pins_per_partitions, num_hedges * num_partitions * sizeof(uint32_t))); // hedge * num_partitions + partition -> count of pins of "hedge" in that "partition"
            CUDA_CHECK(cudaMemset(d_pins_per_partitions, 0x00, num_hedges * num_partitions * sizeof(uint32_t)));

            // launch configuration - pins per partition kernel
            threads_per_block = 256;
            num_threads_needed = num_hedges; // 1 thread per hedge
            blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            bytes_per_thread = 0; //TODO
            shared_bytes = threads_per_block * bytes_per_thread;
            // launch - pins per partition kernel
            std::cout << "Running pins per partition kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            // NOTE: compute this once, keep it up to date as you refine!
            // NOTE: having this available during FM refinement makes its complexity linear in the connectivity, instead of quadratic!
            pins_per_partition_kernel<<<blocks, threads_per_block, shared_bytes>>>(
                d_hedges,
                d_offsets,
                d_partitions,
                num_hedges,
                curr_num_nodes,
                d_pins_per_partitions
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            return new_num_nodes;
        }
        // ======================================

        // =============================
        // print some temporary results
        #if !VERBOSE
        std::vector<slot> slots_tmp(curr_num_nodes);
        CUDA_CHECK(cudaMemcpy(pairs_tmp.data(), d_pairs, curr_num_nodes * sizeof(uint32_t) * MAX_CANDIDATES, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(slots_tmp.data(), d_slots, curr_num_nodes * sizeof(slot) * MAX_GROUP_SIZE, cudaMemcpyDeviceToHost));
        std::set<uint32_t> groups_count;
        std::cout << "Grouping results:\n";
        for (uint32_t i = 0; i < curr_num_nodes; ++i) {
            uint32_t target = pairs_tmp[i];
            uint32_t group = slots_tmp[i].id;
            groups_count.insert(group);
            if (i < std::min<uint32_t>(curr_num_nodes, 20)) {
                std::cout << "  node " << i << " ->";
                for (uint32_t j = 0; j < MAX_CANDIDATES; ++j) {
                    if (target == UINT32_MAX) std::cout << "target=none\n";
                    else std::cout << " (" << j << " target=" << target << ")";
                }
                std::cout << " group=" << group << "\n";
            }
        }
        std::cout << "Groups count: " << groups_count.size() << "\n";
        #endif
        // =============================

        // launch configuration - coarsening kernel (hedges)
        threads_per_block = 128;
        num_threads_needed = num_hedges; // 1 thread per hedge
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - coarsening kernel (hedges)
        std::cout << "Running coarsening kernel (hedges) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_hedges<<<blocks, threads_per_block>>>(
            num_hedges,
            d_offsets,
            d_groups,
            d_hedges
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // launch configuration - coarsening kernel (neighbors)
        threads_per_block = 128;
        num_threads_needed = curr_num_nodes; // 1 thread per node
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - coarsening kernel (neighbors)
        std::cout << "Running coarsening kernel (neighbors) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_neighbors<<<blocks, threads_per_block>>>(
            curr_num_nodes,
            d_neighbors_offsets,
            d_groups,
            d_neighbors
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // launch configuration - coarsening kernel (touching)
        threads_per_block = 128;
        num_threads_needed = num_hedges; // 1 thread per hedge
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - coarsening kernel (neighbors)
        std::cout << "Running coarsening kernel (touching) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_touching_count<<<blocks, threads_per_block>>>(
            d_hedges,
            d_offsets,
            num_hedges,
            d_groups,
            d_touching_offsets // written by each group, from idx 0 to new_num_nodes - 1 + 1
        );
        thrust::device_ptr<uint32_t> t_touching_offsets(d_touching_offsets);
        thrust::exclusive_scan(t_touching_offsets, t_touching_offsets + (new_num_nodes + 1), t_touching_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
        uint32_t *d_touching_counter = nullptr;
        CUDA_CHECK(cudaMalloc(&d_touching_counter, new_num_nodes * sizeof(uint32_t))); // node -> number of touching hedges seen as of now
        CUDA_CHECK(cudaMemset(d_touching_counter, 0x00, new_num_nodes * sizeof(uint32_t)));
        apply_coarsening_touching_scatter<<<blocks, threads_per_block>>>(
            d_hedges,
            d_offsets,
            num_hedges,
            d_groups,
            d_touching_offsets,
            d_touching,
            d_touching_counter
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_touching_counter));

        // ======================================
        // recursive call, go down one more level
        uint32_t num_partitions = 0;
        //uint32_t num_partitions = coarsen_refine_uncoarsen(level_idx + 1, new_num_nodes);
        // ======================================

        std::cout << "Refining level " << level_idx << ", remaining nodes=" << curr_num_nodes << "number of partitions=" << num_partitions << "\n";

        // zero-out this kernel's outputs
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
        thrust::device_ptr<uint32_t> t_scores(d_scores);
        thrust::sort_by_key(d_scores, d_scores + curr_num_nodes, t_indices.begin()); // sort scores according to scores themselves and indices in the same way
        thrust::scatter(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(curr_num_nodes), t_indices.begin(), t_ranks.begin()); // invert the permutation such that: ranks[original_index] = sorted_position
        // launch configuration - fm-ref cascade kernel => same as "fm-ref gains kernel"
        // compute shared memory per block (bytes)
        bytes_per_warp = 0; //TODO
        shared_bytes = warps_per_block * bytes_per_warp;
        // launch - fm-ref cascade kernel
        std::cout << "Running fm-ref gains kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        fm_refinement_cascade_kernel<<<blocks, threads_per_block, shared_bytes>>>(
            d_hedges,
            d_offsets,
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
            d_partitions,
            d_pins_per_partitions
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_ranks));

        std::cout << "Uncoarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

        // launch configuration - uncoarsen kernel
        // launch - uncoarsen kernel
        // required kernels:
        // - uncoarsen touching -> set the upper bits of the hedge id in one-hot encoding for the node in the .
        // - uncoarsen neighbors -> not used for refinement
        // - NO NEED TO uncoarsen hedges -> not used for refinement
        // - uncoarsen partitions
        // - uncoarsen pins-per-partition (compute or update? complexity is O(e*d) = O(n*h) regardless, but updating requires atomics... => if recompute, remove it from fm_refinement_apply_kernel!)
        // NOTE: no need for (multi)function bits! We never truly invert groups aside from expanding partitions!
        /*
        * To uncoarsen partitions just use "d_groups" like this:
        * - build a two lists: "d_ungroup" of lenth curr_num_nodes and "d_ungroup_offsets" of length new_num_nodes + 1 such that "d_ungroup_offsets"
        *   is indexed by the group id, and tells you the index in "d_ungroup" where you can start finding the node ids for that group;
        *   can be build by sorting and then doing a scan over "d_groups"; or better, can be build by counting the occurrencies of each group;
        *   this can be built as a side-job while constructing "d_groups"!
        * - create a "new_partitions" buffer, and spawn a thread per group id, that goes and writes to the entries in "new_partitions" for each of
        *   the node ids in "d_ungroup" between "d_ungroup_offsets[group_idx]" and "d_ungroup_offsets[group_idx + 1]".
        *
        * NOTE: just like groups, partitions need to ordered, as they be used as indices; however, partitions are few, and if one becomes
        *       empty we can just discard its index and leave a few empty spots in the data structures, it's cheaper to compress at the end
        */

        // cleanup groups
        CUDA_CHECK(cudaFree(d_groups));
    };

    // START: first level, down we go!
    uint32_t num_partitions = coarsen_refine_uncoarsen(0, num_nodes);

    // TO FINISH UP!

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
    CUDA_CHECK(cudaFree(d_offsets));
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
