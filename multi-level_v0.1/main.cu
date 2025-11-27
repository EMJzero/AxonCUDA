#include </home/mronzani/cuda/include/cuda_runtime.h>
#include <iostream>
#include <string>

#include "hgraph.hpp"

#define WARP_SIZE 32u

extern __global__ void candidates_kernel(
    const uint32_t* hedge_offsets,
    const uint32_t* hedges,
    const uint32_t* neighbors,
    const uint32_t* neighbors_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    uint32_t* pairs,
    uint32_t* groups
);

extern __global__ void grouping_minfirst_kernel(
    const uint32_t* pairs,
    const uint32_t num_nodes,
    uint32_t* groups,
    uint32_t* distances
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

    // prepare neighborhoods
    // TODO: this is SLOOOOW, run it on GPU too!
    hg.buildNeighborhoods();
    
    // device pointers
    uint32_t *d_offsets = nullptr, *d_hedges = nullptr;
    uint32_t *d_neighbors = nullptr, *d_neighbors_offsets = nullptr;
    uint32_t *d_touching = nullptr, *d_touching_offsets = nullptr;
    float *d_hedge_weights = nullptr;
    uint32_t *d_pairs = nullptr;
    uint32_t *d_groups = nullptr, *d_distances = nullptr;

    // allocate device memory (use sizeof instead of *4)
    CUDA_CHECK(cudaMalloc(&d_offsets, hedge_offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_neighbors, hg.getNeighborhoods().size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_neighbors_offsets, hg.getNeighborhoodOffsets().size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_touching, touching_hedges.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_touching_offsets, touching_hedge_offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hedge_weights, num_hedges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pairs, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_groups, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_distances, num_nodes * sizeof(uint32_t)));

    // copy to device
    CUDA_CHECK(cudaMemcpy(d_offsets, hedge_offsets.data(), hedge_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, hg.getNeighborhoods().data(), hg.getNeighborhoods().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors_offsets, hg.getNeighborhoodOffsets().data(), hg.getNeighborhoodOffsets().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching, touching_hedges.data(), touching_hedges.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching_offsets, touching_hedge_offsets.data(), touching_hedge_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedge_weights, hedge_weight_host.data(), num_hedges * sizeof(float), cudaMemcpyHostToDevice));

    // zero-out outputs
    CUDA_CHECK(cudaMemset(d_pairs, 0xFFFFFFFF, num_nodes * sizeof(uint32_t))); // 0xFF... -> UINT32_MAX
    CUDA_CHECK(cudaMemset(d_groups, 0, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_distances, 0, num_nodes * sizeof(uint32_t)));

    // launch configuration
    // choose threads_per_block multiple of WARP_SIZE
    int threads_per_block = 128; // 128/32 -> 4 warps per block
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_warps_needed = hg.nodes() ; // 1 warp per node
    int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
    // compute shared memory per block (bytes)
    size_t bytes_per_warp = 0; //TODO
    size_t shared_bytes = warps_per_block * bytes_per_warp;

    // launch
    std::cout << "Running candidates kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    candidates_kernel<<<blocks, threads_per_block, shared_bytes>>>(
        d_offsets,
        d_hedges,
        d_neighbors,
        d_neighbors_offsets,
        d_touching,
        d_touching_offsets,
        d_hedge_weights,
        num_hedges,
        num_nodes,
        d_pairs,
        d_groups // TODO: this is a cheeky hack to initialize "d_groups" in parallel (each node to itself)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // launch configuration
    threads_per_block = 128;
    int num_threads_needed = hg.nodes() ; // 1 thread per node
    blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
    size_t bytes_per_thread = 0; //TODO
    shared_bytes = threads_per_block * bytes_per_thread;

    std::cout << "Running grouping kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    grouping_minfirst_kernel<<<blocks, threads_per_block, shared_bytes>>>(
        d_pairs,
        num_nodes,
        d_groups,
        d_distances
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back results
    std::vector<uint32_t> pairs(num_nodes);
    std::vector<uint32_t> groups(num_nodes);
    std::vector<uint32_t> distances(num_nodes);
    CUDA_CHECK(cudaMemcpy(pairs.data(), d_pairs, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(groups.data(), d_groups, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(distances.data(), d_distances, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // print some example outputs
    std::set<uint32_t> groups_count;
    std::cout << "Results:\n";
    for (uint32_t i = 0; i < std::min<uint32_t>(num_nodes, 20); ++i) {
        //uint32_t group = (uint32_t)(groups[i] >> 32);
        //uint32_t rank = (uint32_t)(groups[i]);
        uint32_t target = pairs[i];
        uint32_t group = groups[i];
        groups_count.insert(group);
        uint32_t rank = distances[i];
        if (target == UINT32_MAX) std::cout << "node " << i << " -> target=none group=none rank=none\n";
        else std::cout << "node " << i << " ->" << " target=" << target << " group=" << group << " rank=" << rank << "\n";
    }
    std::cout << "Groups count: " << groups_count.size() << "\n";

    // cleanup device memory
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_hedges));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_neighbors_offsets));
    CUDA_CHECK(cudaFree(d_touching));
    CUDA_CHECK(cudaFree(d_touching_offsets));
    CUDA_CHECK(cudaFree(d_hedge_weights));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_groups));
    CUDA_CHECK(cudaFree(d_distances));

    // === CUDA STUFF ENDS HERE ===
    // ============================

    // save hypergraph
    if (!save_path.empty()) {
        if (!loaded) {
            std::cerr << "Error: -s used without loading a hypergraph first.\n";
            return 1;
        }
        try {
            hg.save(save_path);
            std::cout << "Saved to " << save_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error saving file: " << e.what() << "\n";
            return 1;
        }
    }

    return 0;
}
