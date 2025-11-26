#include </home/mronzani/cuda/include/cuda_runtime.h>
#include <iostream>
#include <string>

#include "hgraph.hpp"

extern __global__ void hyperedge_avg_kernel( const uint32_t* hedge_offsets, const uint32_t* hedges, float* averages, uint32_t num_hedges);

using namespace hgraph;

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

    // allocate GPU memory
    uint32_t *d_offsets, *d_hedges;
    float *d_avgs;

    cudaMalloc(&d_offsets, hedge_offsets.size() * sizeof(uint32_t));
    cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t));
    cudaMalloc(&d_avgs, num_hedges * sizeof(float));

    cudaMemcpy(d_offsets, hedge_offsets.data(), hedge_offsets.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size()*4, cudaMemcpyHostToDevice);

    // kernel launch
    int block = 128;
    int grid = (num_hedges + block - 1) / block;

    hyperedge_avg_kernel<<<grid, block>>>(d_offsets, d_hedges, d_avgs, num_hedges);
    cudaDeviceSynchronize();

    // move back results
    std::vector<float> avgs(num_hedges);
    cudaMemcpy(avgs.data(), d_avgs, num_hedges*sizeof(float), cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < 10; i++)
        std::cout << "Hedge " << i << " avg = " << avgs[i] << "\n";

    // cleanup
    cudaFree(d_offsets);
    cudaFree(d_hedges);
    cudaFree(d_avgs);

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
