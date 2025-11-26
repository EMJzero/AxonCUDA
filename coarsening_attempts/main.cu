#include </home/mronzani/cuda/include/cuda_runtime.h>
#include <iostream>
#include <string>

#include "hgraph.hpp"

#define WARP_SIZE 32
#define HASH_SIZE 1024u

extern __global__ void hyperedge_candidate_kernel(const uint32_t* hedge_offsets, const uint32_t* hedges_flat, const float* hedge_weight, uint32_t num_hedges, uint32_t num_nodes, uint32_t* out_best, float* out_best_score);

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
    std::vector<uint32_t> hedge_offsets;
    hedge_offsets.reserve(num_hedges + 1);

    // prepare hedge offsets
    for (uint32_t i = 0; i < num_hedges; ++i)
        hedge_offsets.push_back(hg.hedges()[i].offset());
    hedge_offsets.push_back(static_cast<uint32_t>(hg.hedgesFlat().size()));

    // optional per-hyperedge weight array (if you want weighted increments)
    std::vector<float> hedge_weight_host(num_hedges);
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedge_weight_host[i] = hg.hedges()[i].weight();
    }

    // total number of distinct nodes (for output indexing)
    uint32_t num_nodes = hg.nodes(); // nodes count used when allocating outputs

    // Device pointers
    uint32_t *d_offsets = nullptr, *d_hedges = nullptr, *d_out_best = nullptr;
    float *d_hedge_weights = nullptr, *d_out_best_score = nullptr;

    // allocate device memory (use sizeof instead of *4)
    CUDA_CHECK(cudaMalloc(&d_offsets, hedge_offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out_best, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out_best_score, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hedge_weights, num_hedges * sizeof(float)));

    // copy to device
    CUDA_CHECK(cudaMemcpy(d_offsets, hedge_offsets.data(), hedge_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedge_weights, hedge_weight_host.data(), num_hedges * sizeof(float), cudaMemcpyHostToDevice));

    // zero-out outputs
    CUDA_CHECK(cudaMemset(d_out_best, 0xFF, num_nodes * sizeof(uint32_t))); // 0xFF -> UINT32_MAX
    CUDA_CHECK(cudaMemset(d_out_best_score, 0, num_nodes * sizeof(float)));

    // Launch kernel:
    // choose threads_per_block multiple of WARP_SIZE
    int threads_per_block = 128; // e.g. 4 warps per block
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_warps_needed = (num_hedges + 0) ; // 1 warp per hyperedge
    int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;

    // compute shared memory per block (bytes)
    // per-warp bytes = HASH_SIZE*(sizeof(uint32_t)+sizeof(float))
    size_t bytes_per_warp = (size_t)HASH_SIZE * (sizeof(uint32_t) + sizeof(float));
    size_t shared_bytes = warps_per_block * bytes_per_warp;

    // launch
    std::cout << "Running kernel...\n";
    hyperedge_candidate_kernel<<<blocks, threads_per_block, shared_bytes>>>(
        d_offsets,
        d_hedges,
        d_hedge_weights,
        num_hedges,
        num_nodes,
        d_out_best,
        d_out_best_score
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back results (only nodes that exist)
    std::vector<uint32_t> out_best_host(num_nodes);
    std::vector<float> out_best_score_host(num_nodes);
    CUDA_CHECK(cudaMemcpy(out_best_host.data(), d_out_best, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_best_score_host.data(), d_out_best_score, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));

    // print some example outputs
    std::cout << "Results:\n";
    for (uint32_t i = 0; i < std::min<uint32_t>(num_nodes, 20); ++i) {
        uint32_t best = out_best_host[i];
        float score = out_best_score_host[i];
        if (best == UINT32_MAX) std::cout << "node " << i << " -> none\n";
        else std::cout << "node " << i << " -> " << best << " score=" << score << "\n";
    }

    // cleanup device memory
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_hedges));
    CUDA_CHECK(cudaFree(d_hedge_weights));
    CUDA_CHECK(cudaFree(d_out_best));
    CUDA_CHECK(cudaFree(d_out_best_score));

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
