#include <tuple>
#include <string>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <algorithm>
#include <filesystem>

#include </home/mronzani/cuda/include/cuda_runtime.h>

#include </home/mronzani/cuda/include/thrust/sort.h>
#include </home/mronzani/cuda/include/thrust/scan.h>
#include </home/mronzani/cuda/include/thrust/scatter.h>
#include </home/mronzani/cuda/include/thrust/sequence.h>
#include </home/mronzani/cuda/include/thrust/transform.h>
#include </home/mronzani/cuda/include/thrust/device_ptr.h>
#include </home/mronzani/cuda/include/thrust/device_vector.h>
#include </home/mronzani/cuda/include/thrust/iterator/discard_iterator.h>
#include </home/mronzani/cuda/include/thrust/iterator/permutation_iterator.h>

#include "../hgraph.hpp"
#include "../nmhardware.hpp"
#include "../utils.cuh"
#include "utils.cuh"

#define DEVICE_ID 0

#define VERBOSE true
#define VERBOSE_LENGTH 20

extern __global__ void inverse_placement_kernel(
    const coords* placement,
    const uint32_t num_nodes,
    uint32_t* inv_placement
);

extern __global__ void forces_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const coords* placement,
    const uint32_t num_nodes,
    float* forces
);

extern __global__ void tensions_kernel(
    const coords* placement,
    const uint32_t* inv_placement,
    const float* forces,
    const uint32_t num_nodes,
    uint32_t* pairs,
    uint32_t* scores
);

extern __global__ void exclusive_swaps_kernel(
    const uint32_t* pairs,
    const uint32_t* scores,
    const uint32_t num_nodes,
    const uint32_t num_repeats,
    slot* swap_slots,
    uint32_t* swap_flags
);

extern __global__ void swap_events_kernel(
    const slot* swap_slots,
    const uint32_t* swap_flags,
    const uint32_t num_nodes,
    swap* ev_swaps,
    float* ev_scores
);

extern __global__ void scatter_ranks_kernel(
    const swap* ev_swaps,
    const uint32_t num_events,
    uint32_t* nodes_rank
);

extern __global__ void cascade_kernel(
    const uint32_t* hedges,
    const uint32_t* hedges_offsets,
    const uint32_t* touching,
    const uint32_t* touching_offsets,
    const float* hedge_weights,
    const coords* placement,
    const swap* ev_swaps,
    const uint32_t* nodes_rank,
    const uint32_t num_events,
    float* scores
);

extern __global__ void apply_swaps_kernel(
    const swap* ev_swaps,
    const uint32_t num_good_swaps,
    coords* placement,
    uint32_t* inv_placement
);

extern __constant__ uint32_t max_width;
extern __constant__ uint32_t max_height;


using namespace hgraph;
using namespace hwmodel;
using namespace hwgeom;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
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
        "  -s <file>   Save placement data to file\n"
        "  -c <name>   Constraints set to use (valid ones: truenorth, loihi64, loihi84, loihi1024 - default is loihi64)\n"
        "  -h          Show this help\n";
}

std::vector<coords> hilbertPlacement(uint32_t nodes, uint32_t width, uint32_t height) {
    if (nodes > width * height)
        throw std::runtime_error("Grid too small to hold all nodes.");
    auto sgn = [](int x) -> int { return (x > 0) - (x < 0); };
    std::vector<coords> result;
    result.reserve(nodes);
    // recursive generator
    auto generate = [&](auto&& self, int x, int y, int ax, int ay, int bx, int by) -> void {
        if (result.size() >= nodes) return;
        int w = std::abs(ax + ay);
        int h = std::abs(bx + by);

        int dax = sgn(ax);
        int day = sgn(ay);
        int dbx = sgn(bx);
        int dby = sgn(by);

        // trivial row fill
        if (h == 1) {
            for (int i = 0; i < w && result.size() < nodes; ++i) {
                result.push_back((coords){ x, y });
                x += dax;
                y += day;
            }
            return;
        }

        // trivial column fill
        if (w == 1) {
            for (int i = 0; i < h && result.size() < nodes; ++i) {
                result.push_back((coords){ x, y });
                x += dbx;
                y += dby;
            }
            return;
        }

        int ax2 = ax / 2;
        int ay2 = ay / 2;
        int bx2 = bx / 2;
        int by2 = by / 2;

        int w2 = std::abs(ax2 + ay2);
        int h2 = std::abs(bx2 + by2);

        if (2 * w > 3 * h) {
            if ((w2 & 1) && w > 2) {
                ax2 += dax;
                ay2 += day;
            }

            // long case
            self(self, x, y, ax2, ay2, bx, by);
            self(self, x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by);
        } else {
            if ((h2 & 1) && h > 2) {
                bx2 += dbx;
                by2 += dby;
            }

            // standard case
            self(self, x, y, bx2, by2, ax2, ay2);
            self(self, x + bx2, y + by2, ax, ay, bx - bx2, by - by2);
            self(self, x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby), -bx2, -by2, -(ax - ax2), -(ay - ay2));
        }
    };

    // Reduce width/height to smallest even values fitting nodes
    uint32_t cw = width  - ((width  % 2) ? 1 : 2);
    uint32_t ch = height - ((height % 2) ? 1 : 2);

    while (cw * ch >= nodes) {
        width = cw;
        height = ch;
        if (width > height) cw -= 2;
        else ch -= 2;
    }

    if (width >= height)
        generate(generate, 0, 0, (int)width, 0, 0, (int)height);
    else
        generate(generate, 0, 0, 0, (int)height, (int)width, 0);

    return result;
}

void printMatrixHex16(const uint32_t* matrix, dim_t width, dim_t height, dim_t maxRows, dim_t maxCols) {
    const dim_t rowsToPrint = std::min(height, maxRows);
    const dim_t colsToPrint = std::min(width, maxCols);
    for (dim_t y = 0; y < rowsToPrint; y++) {
        for (dim_t x = 0; x < colsToPrint; x++) {
            uint32_t value = matrix[y * width + x];
            uint16_t low16 = static_cast<uint16_t>(value & 0xFFFF);
            std::printf("%04X", low16);
            if (x + 1 < colsToPrint)
                std::printf(" ");
        }
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc == 1) {
        printHelp();
        return 0;
    }

    std::string load_path;
    std::string save_path;
    std::string constraints;

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
        else if (arg == "-c") {
            if (i + 1 >= argc) { std::cerr << "Error: -c requires a config name\n"; return 1; }
            constraints = argv[++i];
        }
        else { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
    }

    // load hypergraph
    HyperGraph hg(0, {}, {}); // placeholder -> overwritten if "-r" is given
    bool loaded = false;

    if (!load_path.empty()) {
        try {
            std::cout << "Loading hypergraph from: " << load_path << " ...\n";
            if (!std::filesystem::is_regular_file(load_path)) throw std::runtime_error("The provided path is not a file.");
            std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(load_path)) / (1 << 20) << " MB\n";
            HyperGraph hg_tmp = HyperGraph::load(load_path);
            std::cout << "Loading complete, ordering nodes ...\n";
            hg = hg_tmp.feedForwardOrder();
            loaded = true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading file: " << e.what() << "\n";
            return 1;
        }

        // print statistics
        std::cout << "Loaded hypergraph:\n";
        std::cout << "  Nodes:       " << hg.nodes() << "\n";
        std::cout << "  Hyperedges:  " << hg.hedges().size() << "\n";
        std::cout << "  Total pins:  " << hg.hedgesFlat().size() << "\n";
        std::cout << "  Total Spike Frequency: " << std::fixed << std::setprecision(3) << hg.totalSpikeFrequency() << "\n";
    } else {
        std::cout << "WARNING, no hypergraph provided (-r), performing a dry-run !!\n";
    }

    // setup the hardware model
    std::optional<HardwareModel> hw_tmp;
    std::unordered_map<std::string, HardwareModel (*)()> configurations {
        { "loihi64", HardwareModel::createLoihiLarge },
        { "loihi84", HardwareModel::createLoihiJin84 },
        { "loihi1024", HardwareModel::createLoihiJin1024 },
        { "truenorth", HardwareModel::createTrueNorth }
    };
    auto hw_it = configurations.find(constraints);
    if (hw_it == configurations.end()) {
        std::cerr << "WARNING, no constraints provided (-c), using loihi64 !!\n";
        hw_tmp = HardwareModel::createLoihiLarge();
    } else {
        hw_tmp = hw_it->second();
    }
    HardwareModel &hw = *hw_tmp;
    
    std::cout << "Using hardware model \"" << hw.name() << "\":\n";
    std::cout << "  Neurons per core:  " << hw.neuronsPerCore() << "\n";
    std::cout << "  Synapses per core: " << hw.synapsesPerCore() << "\n";
    std::cout << "  Cores along x, y:  " << hw.coresPerChipX() << ", " << hw.coresPerChipY() << " (" << hw.coresPerChipX() * hw.coresPerChipY() << " tot.)" << "\n";
    std::cout << "  Chips along x, y:  " << hw.chipsPerSystemX() << ", " << hw.chipsPerSystemY() << " (" << hw.chipsPerSystemX() * hw.chipsPerSystemY() << " tot.)" << "\n";
    std::cout << "  Routing energy, latency: " << std::fixed << std::setprecision(3) << hw.energyPerRouting() << " pJ, " << hw.latencyPerRouting() << " ns\n";
    std::cout << "  Wire energy, latency:    " << std::fixed << std::setprecision(3) << hw.energyPerWire() << " pJ, " << hw.latencyPerWire() << " ns\n";

    if (hg.nodes() > hw.coresAlongX() * hw.coresAlongY()) {
        std::cerr << "ERROR, the hypergraph has more nodes (" << hg.nodes() << ") than the 2D lattice has points (" << hw.coresAlongX() * hw.coresAlongY() << "), placement would fail !!\n";
        return 1;
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
    
    std::cout << "Starting timer...\n";
    auto time_start = std::chrono::high_resolution_clock::now();

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
        // NOTE: must put in inbounds first!
        for (uint32_t h : hg.inboundSortedIds(n))
            touching_hedges.push_back(h);
        for (uint32_t h : hg.outboundSortedIds(n))
            touching_hedges.push_back(h);
    }
    touching_hedges_offsets.push_back(touching_hedges.size());

    // prepare hyperedge weights
    std::vector<float> hedge_weights(num_hedges);
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedge_weights[i] = hg.hedges()[i].weight();
    }

    // total number of distinct nodes (for output indexing)
    const uint32_t num_nodes = hg.nodes(); // nodes count used when allocating outputs
    
    // constraints
    const uint32_t h_max_width = hw.coresAlongX();
    const uint32_t h_max_height = hw.coresAlongY();

    // device pointers
    uint32_t *d_hedges_offsets = nullptr, *d_hedges = nullptr;
    uint32_t *d_touching = nullptr, *d_touching_offsets = nullptr, *d_inbound_count = nullptr;
    float *d_hedge_weights = nullptr;
    coords *d_placement = nullptr;
    uint32_t *d_inv_placement = nullptr;
    // refinement structures
    float *d_forces = nullptr;
    uint32_t *d_pairs = nullptr;
    uint32_t *d_scores = nullptr;
    slot *d_swap_slots = nullptr;
    uint32_t *d_swap_flags = nullptr;
    // events structures
    swap *d_ev_swaps = nullptr;
    float *d_ev_scores = nullptr;
    uint32_t *d_nodes_rank = nullptr;

     // kernel dimensions
    int blocks, threads_per_block, warps_per_block;
    int num_threads_needed, num_warps_needed;
    size_t bytes_per_thread, shared_bytes; // bytes_per_warp
    int blocks_per_SM, max_blocks;

    // allocate device memory
    CUDA_CHECK(cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t))); // contigous hedges array (each hedge must be stored as src+destinations, with the src in the first position)
    CUDA_CHECK(cudaMalloc(&d_hedges_offsets, (num_hedges + 1) * sizeof(uint32_t))); // hedges_offsets[hedge idx] -> hedge start idx in d_hedges
    CUDA_CHECK(cudaMalloc(&d_touching, touching_hedges.size() * sizeof(uint32_t))); // contigous inbound+outbout sets array (first inbound, then outbound)
    CUDA_CHECK(cudaMalloc(&d_touching_offsets, (num_nodes + 1) * sizeof(uint32_t))); // touching_offsets[node idx] -> touching set start idx in d_touching
    CUDA_CHECK(cudaMalloc(&d_inbound_count, num_nodes * sizeof(uint32_t))); // inbound_count[node idx] -> how many hedge of touching[node idx] are inbound (inbound hedges are before inbound_count[node idx], then outbound)
    CUDA_CHECK(cudaMalloc(&d_hedge_weights, num_hedges * sizeof(float))); // hedge_weights[hedge idx] -> weight
    CUDA_CHECK(cudaMalloc(&d_placement, num_nodes * sizeof(coords))); // placement[node idx] -> x and y placement coordinates of node
    CUDA_CHECK(cudaMalloc(&d_inv_placement, h_max_width * h_max_height * sizeof(uint32_t))); // inv_placement[y * h_max_width + x] -> idx of the node occupying such place, or UINT32_MAX
    CUDA_CHECK(cudaMalloc(&d_forces, 4 * num_nodes * sizeof(float))); // forces[4*node idx + 0 for dx, + 1 for sx, +2 for up, +2 for down] -> direction of the node's two proposed moves
    CUDA_CHECK(cudaMalloc(&d_pairs, MAX_CANDIDATE_MOVES * num_nodes * sizeof(uint32_t))); // pairs[4*node idx + 0..] -> nodes the current one wants to swap with, ordered by decreasing score
    CUDA_CHECK(cudaMalloc(&d_scores, MAX_CANDIDATE_MOVES * num_nodes * sizeof(uint32_t))); // scores[4*node idx + 0..] -> score with which node wants to pair with other nodes
    CUDA_CHECK(cudaMalloc(&d_swap_slots, num_nodes * sizeof(slot))); // slot to finalize node pairs while computing exclusive swaps (true dtype: "slot")
    //CUDA_CHECK(cudaMalloc(&d_swaps, num_nodes * sizeof(uint32_t))); // swaps[node idx] -> other node the current one can be swapped with (only contains mutually-pointing pairs)
    CUDA_CHECK(cudaMalloc(&d_swap_flags, (num_nodes + 1) * sizeof(uint32_t))); // swap_flag[node idx] -> set to 1 for the lower-id of each node in a swap-pair, in order to create swap-events
    CUDA_CHECK(cudaMalloc(&d_ev_swaps, num_nodes * sizeof(swap))); // ev_swaps[event idx] -> pair of nodes involved in the event's swap
    CUDA_CHECK(cudaMalloc(&d_ev_scores, num_nodes * sizeof(float))); // ev_scores[event idx] -> score (cost gain) achieved by the event's swap
    CUDA_CHECK(cudaMalloc(&d_nodes_rank, num_nodes * sizeof(uint32_t))); // node_rank[node idx] -> rank (index) in the sorted events by score of the node

    // copy to device
    CUDA_CHECK(cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedges_offsets, hedges_offsets.data(), (num_hedges + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching, touching_hedges.data(), touching_hedges.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching_offsets, touching_hedges_offsets.data(), (num_nodes + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedge_weights, hedge_weights.data(), num_hedges * sizeof(float), cudaMemcpyHostToDevice));

    // thrust pointers
    thrust::device_ptr<uint32_t> t_touching_offsets(d_touching_offsets);
    thrust::device_ptr<uint32_t> t_inbound_count(d_inbound_count);
    thrust::device_ptr<slot> t_swap_slots(d_swap_slots);
    thrust::device_ptr<uint32_t> t_swap_flags(d_swap_flags);
    thrust::device_ptr<swap> t_ev_swaps(d_ev_swaps);
    thrust::device_ptr<float> t_ev_scores(d_ev_scores);

    // initialize
    // each initial node has one outbound hyperedge -> init. inbound counts to the number of touching - 1
    thrust::transform(t_touching_offsets + 1, t_touching_offsets + 1 + num_nodes, t_touching_offsets, t_inbound_count, [] __device__ (int next, int curr) { return next - curr - 1; });

    // copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(max_width, &h_max_width, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(max_height, &h_max_height, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    // wrap up memory duties with a sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // initial placement
    // TODO: TOPOLOGICAL NODES ORDER BEFORE THIS! OR SPECTRAL LAYOUT !!
    std::vector<coords> init_placement = hilbertPlacement(num_nodes, h_max_width, h_max_height);
    CUDA_CHECK(cudaMemcpy(d_placement, init_placement.data(), num_nodes * sizeof(coords), cudaMemcpyHostToDevice));
    
    std::vector<Coord2D> h_init_placement(num_nodes);
    for (uint32_t i = 0; i < num_nodes; i++) {
        h_init_placement[i] = Coord2D(
            init_placement[i].x,
            init_placement[i].y
        );
    }
    
    if (hw.checkPlacementValidity(hg, h_init_placement, true)) {
        auto metrics = hw.getAllMetrics(hg, h_init_placement);
        std::cout << "Initial placement metrics:\n";
        std::cout << "  Energy:        " << std::fixed << std::setprecision(3) << metrics.energy.value() << "\n";
        std::cout << "  Avg. latency:  " << std::fixed << std::setprecision(3) << metrics.avg_latency.value() << "\n";
        std::cout << "  Max. Latency:  " << std::fixed << std::setprecision(3) << metrics.max_latency.value() << "\n";
        std::cout << "  Avg. congestion:  " << std::fixed << std::setprecision(3) << metrics.avg_congestion.value() << "\n";
        std::cout << "  Max. congestion:  " << std::fixed << std::setprecision(3) << metrics.max_congestion.value() << "\n";
        std::cout << "  Connections locality:\n";
        std::cout << "    Flat:     " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean << " ar. mean, " << metrics.connections_locality.value().geo_mean << " geo. mean\n";
        std::cout << "    Weighted: " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean_weighted << " ar. mean, " << metrics.connections_locality.value().geo_mean_weighted << " geo. mean\n";
    } else {
        std::cerr << "ERROR, invalid initial placement !!\n";
        return 1;
    }
    std::vector<Coord2D>().swap(h_init_placement);

    // =============================
    // print some temporary results
    #if VERBOSE
    std::cout << "Initial placement:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            const coords place = init_placement[i];
            std::cout << "  node " << i << " -> x=" << place.x << " y=" << place.y << "\n";
        }
    }
    #endif
    // =============================

    // initialize inverse placement
    CUDA_CHECK(cudaMemset(d_inv_placement, 0xFF, h_max_width * h_max_height * sizeof(uint32_t)));
    // launch configuration - inverse placement kernel
    threads_per_block = 128;
    num_threads_needed = num_nodes; // 1 thread per node
    blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
    // launch - inverse placement kernel
    std::cout << "Running inverse placement kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    inverse_placement_kernel<<<blocks, threads_per_block>>>(
        d_placement,
        num_nodes,
        d_inv_placement
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // =============================
    // print some temporary results
    #if VERBOSE
    std::vector<uint32_t> inv_place_tmp(h_max_width * h_max_height);
    CUDA_CHECK(cudaMemcpy(inv_place_tmp.data(), d_inv_placement, h_max_width * h_max_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Initial inverse placement:\n";
    printMatrixHex16(inv_place_tmp.data(), h_max_width, h_max_height, VERBOSE_LENGTH, VERBOSE_LENGTH);
    std::vector<uint32_t>().swap(inv_place_tmp);
    #endif
    // =============================

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        std::cout << "Force-directed refinement, iteration " << iter << "\n";

        /*
        * Flow:
        * 1) compute forces from each node to the 4 cardinal placements around it
        * 2) compute the tension between each node and the 4 places around it
        * 3) select the highest-tension pairs of nodes to swap/move
        *   - each node can be moved at most once -> upward and downward passes (same as in grouping for coarsening)
        * 4) find and apply the highest subsequence of improving moves
        *   - create one move-event per pair
        *   - rank events
        *   - update each event's gain assuming all higher-ranked ones already applied
        *   - scan all updated gains, find the highest point in the sequence, and apply all moves up to it
        */

        // launch configuration - forces kernel
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = num_nodes ; // 1 warp per node
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - forces kernel
        std::cout << "Running forces kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        forces_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            d_touching,
            d_touching_offsets,
            d_hedge_weights,
            d_placement,
            num_nodes,
            d_forces
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // =============================
        // print some temporary results
        #if VERBOSE
        std::vector<float> forces_tmp(num_nodes * 4);
        CUDA_CHECK(cudaMemcpy(forces_tmp.data(), d_forces, num_nodes * 4 * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Forces:\n";
        for (uint32_t i = 0; i < num_nodes; ++i) {
            if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
                std::cout << "  node " << i << " ->";
                std::cout << " (" << forces_tmp[4 * i + LEFT] << " LEFT)";
                std::cout << " (" << forces_tmp[4 * i + RIGHT] << " RIGHT)";
                std::cout << " (" << forces_tmp[4 * i + UP] << " UP)";
                std::cout << " (" << forces_tmp[4 * i + DOWN] << " DOWN)\n";
            }
        }
        std::vector<float>().swap(forces_tmp);
        #endif
        // =============================

        // launch configuration - tensions kernel
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = num_nodes ; // 1 warp per node
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - tensions kernel
        std::cout << "Running tensions kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        tensions_kernel<<<blocks, threads_per_block>>>(
            d_placement,
            d_inv_placement,
            d_forces,
            num_nodes,
            d_pairs,
            d_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // =============================
        // print some temporary results
        #if VERBOSE
        std::vector<uint32_t> pairs_tmp(num_nodes * MAX_CANDIDATE_MOVES);
        std::vector<uint32_t> scores_tmp(num_nodes * MAX_CANDIDATE_MOVES);
        CUDA_CHECK(cudaMemcpy(pairs_tmp.data(), d_pairs, num_nodes * sizeof(uint32_t) * MAX_CANDIDATE_MOVES, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scores_tmp.data(), d_scores, num_nodes * sizeof(uint32_t) * MAX_CANDIDATE_MOVES, cudaMemcpyDeviceToHost));
        std::unordered_map<uint32_t, int> groups_count;
        std::cout << "Tensions:\n";
        for (uint32_t i = 0; i < num_nodes; ++i) {
            if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
                std::cout << "  node " << i << " ->";
                for (uint32_t j = 0; j < MAX_CANDIDATE_MOVES; ++j) {
                    uint32_t target = pairs_tmp[i * MAX_CANDIDATE_MOVES + j];
                    uint32_t score = scores_tmp[i * MAX_CANDIDATE_MOVES + j];
                    if (target == UINT32_MAX) std::cout << " (" << j << " target=none score=" << score << ")";
                    else if (target == UINT32_MAX - LEFT) std::cout << " (" << j << " target=LEFT score=" << score << ")";
                    else if (target == UINT32_MAX - RIGHT) std::cout << " (" << j << " target=RIGHT score=" << score << ")";
                    else if (target == UINT32_MAX - UP) std::cout << " (" << j << " target=UP score=" << score << ")";
                    else if (target == UINT32_MAX - DOWN) std::cout << " (" << j << " target=DOWN score=" << score << ")";
                    else std::cout << " (" << j << " target=" << target << " score=" << score << ")";
                }
                std::cout << "\n";
            }
        }
        std::vector<uint32_t>().swap(pairs_tmp);
        std::vector<uint32_t>().swap(scores_tmp);
        #endif
        // =============================

        // zero-out swap slots and flags
        slot init_slot; init_slot.id = UINT32_MAX; init_slot.score = 0u;
        thrust::fill(t_swap_slots, t_swap_slots + num_nodes, init_slot); // upper 32 bits to 0x00, lower 32 to 0xFF
        CUDA_CHECK(cudaMemset(d_swap_flags, 0x00, (num_nodes + 1) * sizeof(uint32_t)));

        // launch configuration - exclusive swaps kernel
        threads_per_block = 256;
        num_threads_needed = num_nodes; // 1 thread per node
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        bytes_per_thread = 0; //TODO
        shared_bytes = threads_per_block * bytes_per_thread;
        // additional checks for the cooperative kernel mode
        blocks_per_SM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_SM, exclusive_swaps_kernel, threads_per_block, shared_bytes);
        max_blocks = blocks_per_SM * props.multiProcessorCount;
        uint32_t num_repeats = 1;
        if (blocks > max_blocks) {
            num_repeats = (blocks + max_blocks - 1) / max_blocks;
            std::cout << "NOTE: exclusive swaps kernel required blocks=" << blocks << ", but max-blocks=" << max_blocks << ", setting repeats=" << num_repeats << " ...\n";
            blocks = (blocks + num_repeats - 1) / num_repeats;
            if (num_repeats > MAX_REPEATS) {
                std::cout << "ABORTING: exclusive swaps kernel required repeats=" << num_repeats << ", but max-repeats=" << MAX_REPEATS << " !!\n";
                abort();
            }
        }
        // launch - exclusive swaps kernel
        std::cout << "Running exclusive swaps kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        void *kernel_args[] = {
            (void*)&d_pairs,
            (void*)&d_scores,
            (void*)&num_nodes,
            (void*)&num_repeats,
            (void*)&d_swap_slots,
            (void*)&d_swap_flags
        };
        cudaLaunchCooperativeKernel((void*)exclusive_swaps_kernel, blocks, threads_per_block, kernel_args, shared_bytes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // =============================
        // print some temporary results
        #if VERBOSE
        std::vector<slot> slots_tmp(num_nodes);
        std::vector<uint32_t> flags_tmp(num_nodes); // leave out the last flag, fine
        CUDA_CHECK(cudaMemcpy(slots_tmp.data(), d_swap_slots, num_nodes * sizeof(slot), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(flags_tmp.data(), d_swap_flags, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::cout << "Swap pairs:\n";
        for (uint32_t i = 0; i < num_nodes; ++i) {
            if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
                slot node_slot = slots_tmp[i];
                if (node_slot.id == UINT32_MAX) std::cout << "  node " << i << " -> target=none score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
                else if (node_slot.id == UINT32_MAX - LEFT) std::cout << "  node " << i << " -> target=LEFT score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
                else if (node_slot.id == UINT32_MAX - RIGHT) std::cout << "  node " << i << " -> target=RIGHT score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
                else if (node_slot.id == UINT32_MAX - UP) std::cout << "  node " << i << " -> target=UP score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
                else if (node_slot.id == UINT32_MAX - DOWN) std::cout << "  node " << i << " -> target=DOWN score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
                else std::cout << "  node " << i << " -> target=" << node_slot.id << " score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
            }
        }
        std::vector<slot>().swap(slots_tmp);
        std::vector<uint32_t>().swap(flags_tmp);
        #endif
        // =============================

        // scan flags to give each event its offset
        thrust::exclusive_scan(t_swap_flags, t_swap_flags + (num_nodes + 1), t_swap_flags);
        uint32_t num_events;
        CUDA_CHECK(cudaMemcpy(&num_events, d_swap_flags + num_nodes, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::cout << "Number of events produced: " << num_events << " ...\n";
        if (num_events > (num_nodes + 1) / 2)
            std::cout << "WARNING, there are more events (" << num_events << ") than half the node (" << (num_nodes + 1) / 2 << "), this >may< an undesirable situation ...\n";
        else if (num_events == 0) {
            std::cout << "Stopping with no events (viable swaps), on iteration " << iter << "\n";
            break;
        }
        // TODO: generate events kernel(s)
        // - extract node1 (lowest id), node2, score in events
        // - no need to rank events, just sort them by score and carry nodes along
        // - now allocate ranks, one entry per node, initialized to UINT32_MAX, then from each event write to its two nodes the event's index (rank)
        // - update the gain of each event in-sequence:
        //   - one warp per event, visit each of the node's touching hedges, one pin per thread in the warp
        //   - for each pin, see its rank, and update its position accordingly
        // - find the maximum gain subsequence and apply it

        // launch configuration - events kernel
        threads_per_block = 128;
        num_threads_needed = num_nodes; // 1 thread per node
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - events kernel
        std::cout << "Running events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        swap_events_kernel<<<blocks, threads_per_block>>>(
            d_swap_slots,
            d_swap_flags,
            num_nodes,
            d_ev_swaps,
            d_ev_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // sort (ascending) events by score while carrying swapped nodes along
        thrust::sort_by_key(t_ev_scores, t_ev_scores + num_events, t_ev_swaps, thrust::greater<float>());
        CUDA_CHECK(cudaMemset(d_nodes_rank, 0xFF, num_nodes * sizeof(uint32_t)));

        // =============================
        // print some temporary results
        #if VERBOSE
        std::vector<swap> ev_swaps_tmp(num_nodes);
        std::vector<float> ev_scores_tmp(num_nodes);
        CUDA_CHECK(cudaMemcpy(ev_swaps_tmp.data(), d_ev_swaps, num_nodes * sizeof(swap), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ev_scores_tmp.data(), d_ev_scores, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Events (sorted - in isolation):\n";
        for (uint32_t i = 0; i < num_nodes; ++i) {
            if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
                swap ev_swap = ev_swaps_tmp[i];
                float ev_score = ev_scores_tmp[i];
                std::cout << "  event " << i << " -> lo=" << ev_swap.lo << " hi=" << ev_swap.hi << " score=" << ev_score << "\n";
            }
        }
        #endif
        // =============================

        // launch configuration - scatter ranks kernel
        threads_per_block = 128;
        num_threads_needed = num_events; // 1 thread per event
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - scatter ranks kernel
        std::cout << "Running scatter ranks kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        scatter_ranks_kernel<<<blocks, threads_per_block>>>(
            d_ev_swaps,
            num_events,
            d_nodes_rank
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // launch configuration - cascade kernel
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = num_events ; // 1 warp per event
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - cascade kernel
        std::cout << "Running cascade kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        cascade_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            d_touching,
            d_touching_offsets,
            d_hedge_weights,
            d_placement,
            d_ev_swaps,
            d_nodes_rank,
            num_events,
            d_ev_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // =============================
        // print some temporary results
        #if VERBOSE
        CUDA_CHECK(cudaMemcpy(ev_scores_tmp.data(), d_ev_scores, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Events (cascade - in sequence):\n";
        for (uint32_t i = 0; i < num_nodes; ++i) {
            if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
                swap ev_swap = ev_swaps_tmp[i];
                float ev_score = ev_scores_tmp[i];
                std::cout << "  event " << i << " -> lo=" << ev_swap.lo << " hi=" << ev_swap.hi << " score=" << ev_score << "\n";
            }
        }
        std::vector<swap>().swap(ev_swaps_tmp);
        std::vector<float>().swap(ev_scores_tmp);
        #endif
        // =============================

        // scan the new scores, find the maximum gain subsequence
        thrust::inclusive_scan(t_ev_scores, t_ev_scores + num_events, t_ev_scores);
        auto best_ev_pos = thrust::max_element(t_ev_scores, t_ev_scores + num_events);
        const uint32_t best_ev = static_cast<uint32_t>(best_ev_pos - t_ev_scores);
        const uint32_t num_good_swaps = best_ev + 1;
        float gain;
        CUDA_CHECK(cudaMemcpy(&gain, d_ev_scores + best_ev, sizeof(float), cudaMemcpyDeviceToHost));

        // stop if the sequence has length 0 or the gain is too low
        if (num_good_swaps == 0 || gain < 0.001f) {
            if (num_good_swaps == 0) std::cout << "Stopping with no further improving swaps, on iteration " << iter << "\n";
            else std::cout << "Stopping with gain" << std::fixed << std::setprecision(3) << gain << " ( < 10^-3) on iteration " << iter << "\n";
            break;
        } else {
            std::cout << "Number of good swaps performed: " << num_good_swaps << " ...\n";
        }

        // update placement and inv_placement
        // launch configuration - apply swaps kernel
        threads_per_block = 128;
        num_threads_needed = num_good_swaps; // 1 thread per swap
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - apply swaps kernel
        std::cout << "Running apply swaps kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_swaps_kernel<<<blocks, threads_per_block>>>(
            d_ev_swaps,
            num_good_swaps,
            d_placement,
            d_inv_placement
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // copy back results
    std::vector<coords> placement(num_nodes);
    CUDA_CHECK(cudaMemcpy(placement.data(), d_placement, num_nodes * sizeof(coords), cudaMemcpyDeviceToHost));

    // =============================
    // print some example outputs
    #if VERBOSE
    std::cout << "Final placement:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            coords place = placement[i];
            std::cout << "  node " << i << " -> x=" << place.x << " y=" << place.y << "\n";
        }
    }
    std::vector<uint32_t> final_inv_place_tmp(h_max_width * h_max_height);
    CUDA_CHECK(cudaMemcpy(final_inv_place_tmp.data(), d_inv_placement, h_max_width * h_max_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Final inverse placement:\n";
    printMatrixHex16(final_inv_place_tmp.data(), h_max_width, h_max_height, VERBOSE_LENGTH, VERBOSE_LENGTH);
    std::vector<uint32_t>().swap(final_inv_place_tmp);
    #endif
    // =============================
    
    // cleanup device memory
    CUDA_CHECK(cudaFree(d_hedges));
    CUDA_CHECK(cudaFree(d_hedges_offsets));
    CUDA_CHECK(cudaFree(d_touching));
    CUDA_CHECK(cudaFree(d_touching_offsets));
    CUDA_CHECK(cudaFree(d_inbound_count));
    CUDA_CHECK(cudaFree(d_hedge_weights));
    CUDA_CHECK(cudaFree(d_placement));
    CUDA_CHECK(cudaFree(d_inv_placement));
    CUDA_CHECK(cudaFree(d_forces));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_swap_slots));
    CUDA_CHECK(cudaFree(d_swap_flags));
    CUDA_CHECK(cudaFree(d_ev_swaps));
    CUDA_CHECK(cudaFree(d_ev_scores));
    CUDA_CHECK(cudaFree(d_nodes_rank));

    // final sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << "Stopping timer...\n";

    // === CUDA STUFF ENDS HERE ===
    // ============================

    std::cerr << "CUDA section: complete; proceeding with placement results validation and evalution...\n";

    double total_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
    std::cout << "Total execution time: " << std::fixed << std::setprecision(3) << total_ms << " ms\n";

    std::vector<Coord2D> h_placement(num_nodes);
    for (uint32_t i = 0; i < num_nodes; i++) {
        h_placement[i] = Coord2D(
            placement[i].x,
            placement[i].y
        );
    }

    if (hw.checkPlacementValidity(hg, h_placement, true)) {
        auto metrics = hw.getAllMetrics(hg, h_placement);
        std::cout << "Placement metrics:\n";
        std::cout << "  Energy:        " << std::fixed << std::setprecision(3) << metrics.energy.value() << "\n";
        std::cout << "  Avg. latency:  " << std::fixed << std::setprecision(3) << metrics.avg_latency.value() << "\n";
        std::cout << "  Max. Latency:  " << std::fixed << std::setprecision(3) << metrics.max_latency.value() << "\n";
        std::cout << "  Avg. congestion:  " << std::fixed << std::setprecision(3) << metrics.avg_congestion.value() << "\n";
        std::cout << "  Max. congestion:  " << std::fixed << std::setprecision(3) << metrics.max_congestion.value() << "\n";
        std::cout << "  Connections locality:\n";
        std::cout << "    Flat:     " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean << " ar. mean, " << metrics.connections_locality.value().geo_mean << " geo. mean\n";
        std::cout << "    Weighted: " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean_weighted << " ar. mean, " << metrics.connections_locality.value().geo_mean_weighted << " geo. mean\n";

        // save hypergraph
        if (!save_path.empty()) {
            if (!loaded) {
                std::cerr << "Error: -s used without loading a hypergraph first.\n";
                return 1;
            }
            try {
                // TODO: apply the partitioning before saving!
                coords_to_file(h_placement, save_path);
                std::cout << "Placement data saved to " << save_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error saving file: " << e.what() << "\n";
                return 1;
            }
        }
    } else {
        std::cerr << "WARNING, invalid placement !!\n";
    }

    return 0;
}
