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

#include "hgraph.hpp"
#include "nmhardware.hpp"
#include "utils.cuh"

#define DEVICE_ID 0

#define VERBOSE true
#define VERBOSE_LENGTH 20

extern __global__ void neighborhoods_count_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t num_nodes,
    const dim_t max_neighbors,
    const bool discharge,
    uint32_t* neighbors,
    dim_t* neighbors_offsets
);

extern __global__ void neighborhoods_scatter_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t num_nodes,
    const dim_t* neighbors_offsets,
    uint32_t* neighbors
);

extern __global__ void candidates_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* neighbors,
    const dim_t* neighbors_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* inbound_count,
    const float* hedge_weights,
    const uint32_t* nodes_sizes,
    const uint32_t num_nodes,
    uint32_t* pairs,
    uint32_t* scores
);

extern __global__ void grouping_kernel(
    const uint32_t* pairs,
    const uint32_t* scores,
    const uint32_t* nodes_sizes,
    const uint32_t num_nodes,
    const uint32_t num_repeats,
    slot* group_slots,
    dp_score* d_dp_scores,
    uint32_t* groups
);

extern __global__ void apply_coarsening_hedges_count(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* groups,
    const uint32_t num_hedges,
    const uint32_t max_hedge_size,
    uint32_t *coarse_oversized_hedges,
    dim_t* coarse_hedges_offsets
);

extern __global__ void apply_coarsening_hedges_scatter(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* groups,
    const uint32_t num_hedges,
    const dim_t* coarse_hedges_offsets,
    uint32_t* coarse_hedges
);

extern __global__ void apply_coarsening_neighbors_count(
    const uint32_t* neighbors,
    const dim_t* neighbors_offsets,
    const uint32_t* groups,
    const uint32_t* ungroups,
    const dim_t* ungroups_offsets,
    const uint32_t num_groups,
    const dim_t max_neighbors,
    const bool discharge,
    uint32_t* oversized_coarse_neighbors,
    dim_t* coarse_neighbors_offsets
);

extern __global__ void apply_coarsening_neighbors_scatter(
    const uint32_t* neighbors,
    const dim_t* neighbors_offsets,
    const uint32_t* groups,
    const uint32_t* ungroups,
    const dim_t* ungroups_offsets,
    const uint32_t num_groups,
    const dim_t* coarse_neighbors_offsets,
    uint32_t* coarse_neighbors
);

extern __global__ void apply_coarsening_touching_count(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t num_hedges,
    dim_t* coarse_touching_offsets
) ;

extern __global__ void apply_coarsening_touching_scatter_inbound(
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* inbound_count,
    const uint32_t* ungroups,
    const dim_t* ungroups_offsets,
    const uint32_t num_groups,
    const dim_t* coarse_touching_offsets,
    uint32_t* coarse_touching,
    uint32_t* coarse_inbound_count
);

extern __global__ void apply_coarsening_touching_scatter_outbound(
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* inbound_count,
    const uint32_t* ungroups,
    const dim_t* ungroups_offsets,
    const uint32_t num_groups,
    const dim_t* coarse_touching_offsets,
    const uint32_t* coarse_inbound_count,
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
    const dim_t* hedges_offsets,
    const uint32_t* partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* pins_per_partitions,
    uint32_t* partitions_inbound_sizes
);

extern __global__ void inbound_pins_per_partition_kernel(
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* inbound_pins_per_partitions,
    uint32_t* partitions_inbound_sizes
);

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

extern __global__ void fm_refinement_apply_kernel(
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* moves,
    const uint32_t* move_ranks,
    const uint32_t* nodes_sizes,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t num_good_moves,
    uint32_t* partitions,
    uint32_t* partitions_sizes
    //uint32_t* pins_per_partitions
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
    uint32_t* valid_moves
);

extern __global__ void build_hedge_events_kernel(
    const uint32_t* moves,
    const uint32_t* ranks,
    const uint32_t* partitions,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint32_t* inbound_count,
    const uint32_t num_nodes,
    uint32_t* ev_partition,
    uint32_t* ev_index,
    uint32_t* ev_hedge,
    int32_t* ev_delta
);

extern __global__ void count_inbound_size_events_kernel(
    const uint32_t* partitions_inbound_counts,
    const uint32_t* ev_partition,
    const uint32_t* ev_index,
    const uint32_t* ev_hedge,
    const int32_t* ev_delta,
    uint32_t num_events,
    uint32_t num_partitions,
    uint32_t* inbound_size_events_offsets
);

extern __global__ void build_inbound_size_events_kernel(
    const uint32_t* partitions_inbound_counts,
    const uint32_t* ev_partition,
    const uint32_t* ev_index,
    const uint32_t* ev_hedge,
    const int32_t* ev_delta,
    const uint32_t* inbound_size_events_offsets,
    uint32_t num_events,
    uint32_t num_partitions,
    uint32_t* new_ev_partition,
    uint32_t* new_ev_index,
    int32_t* new_ev_delta
);

extern __global__ void flag_inbound_events_kernel(
    const uint32_t* ev_partition,
    const uint32_t* ev_index,
    const int32_t* ev_delta,
    const uint32_t* partitions_inbound_sizes,
    const uint32_t num_events,
    uint32_t* valid_moves
);

extern __global__ void pack_segments(
    const uint32_t* oversized,
    const dim_t* offsets,
    const uint32_t num_subs,
    const dim_t sub_size,
    uint32_t* out
);

extern __global__ void pack_segments_varsize(
    const uint32_t* oversized,
    const dim_t* oversized_offsets,
    const dim_t* offsets,
    const uint32_t num_subs,
    const dim_t base_sub_size,
    uint32_t* out
);

extern __constant__ uint32_t max_nodes_per_part;
extern __constant__ uint32_t max_inbound_per_part;


using namespace hgraph;
using namespace hwmodel;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void printHelp() {
    std::cout <<
        "Usage:\n"
        "  prog -r <input_file> [-s <output_file>]\n"
        "  prog -h\n\n"
        "Options:\n"
        "  -r <file>   Reload hypergraph from file\n"
        "  -s <file>   Save partitioned hypergraph to file\n"
        "  -c <name>   Constraints set to use (valid ones: truenorth, loihi64, loihi84, loihi1024 - default is loihi64)\n"
        "  -h          Show this help\n";
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
        } else if (arg == "-s") {
            if (i + 1 >= argc) { std::cerr << "Error: -s requires a file path\n"; return 1; }
            save_path = argv[++i];
        } else if (arg == "-c") {
            if (i + 1 >= argc) { std::cerr << "Error: -c requires a config name\n"; return 1; }
            constraints = argv[++i];
        } else { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
    }

    // load hypergraph
    HyperGraph hg(0, {}, {}); // placeholder -> overwritten if "-r" is given
    bool loaded = false;

    if (!load_path.empty()) {
        try {
            if (!std::filesystem::is_regular_file(load_path)) throw std::runtime_error("Failed to load hypergraph, the provided path is not a file.");
            std::filesystem::path file_path(load_path);
            if (file_path.extension() == ".hgr") {
                std::cout << "Loading hypergraph from: " << load_path << " (hMetis format) ...\n";
                std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(load_path)) / (1 << 20) << " MB\n";
                hg = HyperGraph::load_hmetis(load_path);
            } else {
                std::cout << "Loading hypergraph from: " << load_path << " (binary format) ...\n";
                std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(load_path)) / (1 << 20) << " MB\n";
                hg = HyperGraph::load(load_path);
            }
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
    
    if (!hw.checkSnnFit(hg, false, true))
        std::cerr << "WARNING, the hypergraph did not pass the fit check on the given constraints (NOTE: this test admits false negatives) !!\n";

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

    std::cout << "Preparing hypergraph data...\n";

    /*
    * Note: by design, only inbound hedges can be constrained (because their deduplication takes priority over outbound), therefore to support other constraints there are two options:
    * - to constrain outbound hedges, simply swap inbound and outbound hedges
    * - to constrain incident (touching) hedges, make them all inbound (no src)
    * 
    * Important:
    * - no cycles admitted
    * - during execution, hedges and incidence sets will diverge:
    *   - hedges remove duplicates (cycles) between sources and destinations, from the destinations (sources preserved)
    *   - incidence sets (touching) remove duplicates between inbound and outbound, from the outbound (inbound preserved -> for constraint checks)
    */

    uint32_t num_hedges = static_cast<uint32_t>(hg.hedges().size());
    std::vector<dim_t> hedges_offsets; // hedge idx -> hedge start index in the contiguous hedges array
    hedges_offsets.reserve(num_hedges + 1);
    
    // prepare hedge offsets
    // HP: no duplicates per hedge, no self-cycles (keep the src only, arg=false -> still consider the hedge among the src's inbounds, arg=true -> update the inbound set to match)
    hg.deduplicateHyperedges(false);
    for (uint32_t i = 0; i < num_hedges; ++i)
        hedges_offsets.push_back(static_cast<dim_t>(hg.hedges()[i].offset()));
    hedges_offsets.push_back(hg.hedgesFlat().size());
    
    std::vector<uint32_t> touching_hedges;
    std::vector<dim_t> touching_hedges_offsets;
    std::vector<uint32_t> inbound_count;
    touching_hedges.reserve(hg.hedgesFlat().size()); // with one outbound hedge per node, the total number of pins (e*d) is the total number of connections (n*h)
    touching_hedges_offsets.reserve(hg.nodes() + 1);
    inbound_count.reserve(hg.nodes());

    // prepare touching sets
    // HP: no duplicates in either set, eventually duplicates in outbound w.r.t. inbounds will also be lost,
    //     inbounds must come first and their part must be sorted by id (ascending)
    for (uint32_t n = 0; n < hg.nodes(); ++n) {
        auto curr_size = touching_hedges.size();
        touching_hedges_offsets.push_back(curr_size);
        // NOTE: must put in inbounds first!
        for (uint32_t h : hg.inboundSortedIds(n))
            touching_hedges.push_back(h);
        inbound_count.push_back(touching_hedges.size() - curr_size);
        for (uint32_t h : hg.outboundSortedIds(n))
            touching_hedges.push_back(h);
    }
    touching_hedges_offsets.push_back(touching_hedges.size());
    uint32_t touching_hedges_size = touching_hedges.size();
    
    // prepare hyperedge weights
    std::vector<float> hedge_weights(num_hedges);
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedge_weights[i] = hg.hedges()[i].weight();
    }
    
    // total number of distinct nodes (for output indexing)
    const uint32_t num_nodes = hg.nodes(); // nodes count used when allocating outputs

    // estimated max hedge and neighbors count
    dim_t max_hedge_size = std::transform_reduce(std::next(hedges_offsets.begin()), hedges_offsets.end(), hedges_offsets.begin(), dim_t{0}, [](dim_t a, dim_t b) { return std::max(a, b); }, [](dim_t next, dim_t curr) { return next - curr; });
    dim_t max_neighbors = hg.sampleMaxNeighborhoodSize(240); // TODO: is 240 enough here?
    std::cout << "Max hedges estimate set to " << max_hedge_size << ", neighbors estimate set to " << max_neighbors << "\n";

    // constraints
    const uint32_t h_max_nodes_per_part = hw.neuronsPerCore();
    const uint32_t h_max_inbound_per_part = hw.synapsesPerCore();
    const uint32_t max_parts = hw.coresCount(); // not needed in kernels
    const uint32_t target_parts = min(max_parts, (num_nodes + h_max_nodes_per_part - 1) / h_max_nodes_per_part);
    
    std::cout << "Starting timer...\n";
    auto time_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Setting up GPU memory...\n";

    // ============================
    // === CUDA STUFF GOES HERE ===
    
    // device streams
    cudaStream_t compute_stream = nullptr;
    cudaStream_t transfer_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
    int least_priority = 0;
    int greatest_priority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    // give higher priority access to memory bandwidth to the compute kernel
    CUDA_CHECK(cudaStreamCreateWithPriority(&compute_stream, cudaStreamNonBlocking, greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&transfer_stream, cudaStreamNonBlocking, least_priority));
    
    // device pointers
    uint32_t *d_hedges = nullptr;
    dim_t *d_hedges_offsets = nullptr;
    uint32_t *d_neighbors = nullptr;
    dim_t *d_neighbors_offsets = nullptr;
    uint32_t *d_touching = nullptr;
    dim_t *d_touching_offsets = nullptr;
    uint32_t *d_inbound_count = nullptr;
    float *d_hedge_weights = nullptr;
    uint32_t *d_pairs = nullptr;
    float *d_f_scores = nullptr;
    uint32_t *d_u_scores = nullptr;
    slot *d_slots = nullptr;
    dp_score *d_dp_scores = nullptr;
    uint32_t *d_nodes_sizes = nullptr;
    uint32_t *d_partitions_sizes = nullptr;
    uint32_t *d_pins_per_partitions = nullptr;
    uint32_t *d_partitions_inbound_sizes = nullptr;

    // kernel dimensions
    int blocks, threads_per_block, warps_per_block;
    int num_threads_needed, num_warps_needed;
    size_t bytes_per_thread, bytes_per_warp, shared_bytes;
    int blocks_per_SM, max_blocks;

    // memory state
    size_t free_bytes, total_bytes;

    // allocate device memory
    CUDA_CHECK(cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t))); // contigous hedges array (each hedge must be stored as src+destinations, with the src in the first position)
    CUDA_CHECK(cudaMalloc(&d_hedges_offsets, (num_hedges + 1) * sizeof(dim_t))); // hedges_offsets[hedge idx] -> hedge start idx in d_hedges
    CUDA_CHECK(cudaMalloc(&d_touching, touching_hedges.size() * sizeof(uint32_t))); // contigous inbound+outbout sets array (first inbound, then outbound)
    CUDA_CHECK(cudaMalloc(&d_touching_offsets, (num_nodes + 1) * sizeof(dim_t))); // touching_offsets[node idx] -> touching set start idx in d_touching
    CUDA_CHECK(cudaMalloc(&d_inbound_count, num_nodes * sizeof(uint32_t))); // inbound_count[node idx] -> how many hedge of touching[node idx] are inbound (inbound hedges are before inbound_count[node idx], then outbound)
    CUDA_CHECK(cudaMalloc(&d_hedge_weights, num_hedges * sizeof(float))); // hedge_weights[hedge idx] -> weight
    CUDA_CHECK(cudaMalloc(&d_pairs, num_nodes * sizeof(uint32_t) * MAX_CANDIDATES)); // partitions[node idx] -> best neighbor
    CUDA_CHECK(cudaMalloc(&d_f_scores, num_nodes * sizeof(float))); // connection streght for each pair, used during refinement
    CUDA_CHECK(cudaMalloc(&d_u_scores, num_nodes * sizeof(uint32_t) * MAX_CANDIDATES)); // fixed point version of the above, used for the multi-candidates kernel
    CUDA_CHECK(cudaMalloc(&d_slots, num_nodes * sizeof(slot) * MAX_GROUP_SIZE)); // slot to finalize node pairs during grouping (true dtype: "slot")
    CUDA_CHECK(cudaMalloc(&d_dp_scores, num_nodes * sizeof(dp_score))); // dynamic programming score for each node in the tree assuming it connected (with) or not (w/out) to its target
    CUDA_CHECK(cudaMalloc(&d_nodes_sizes, num_nodes * sizeof(uint32_t))); // nodes_size[node idx] -> how many pins the node counts as towards the partition size limit

    // copy to device
    CUDA_CHECK(cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedges_offsets, hedges_offsets.data(), (num_hedges + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching, touching_hedges.data(), touching_hedges.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching_offsets, touching_hedges_offsets.data(), (num_nodes + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedge_weights, hedge_weights.data(), num_hedges * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inbound_count, inbound_count.data(), num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice));
    std::vector<dim_t>().swap(hedges_offsets);
    std::vector<uint32_t>().swap(touching_hedges);
    std::vector<dim_t>().swap(touching_hedges_offsets);
    std::vector<uint32_t>().swap(inbound_count);

    // initialize
    thrust::device_ptr<uint32_t> t_nodes_sizes(d_nodes_sizes);
    thrust::fill(t_nodes_sizes, t_nodes_sizes + num_nodes, 1u); // each initial node counts as 1 (NOTE: can be tuned to give some nodes more "space")

    // copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(max_nodes_per_part, &h_max_nodes_per_part, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(max_inbound_per_part, &h_max_inbound_per_part, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    // wrap up memory duties with a sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // prepare neighborhoods
    // HP: no duplicates in neighbors, no one's own self among one's neighbors
    // uses a two-step method, first just counting, then writing, to allocate exactly the amount of memory needed, since neighborhoods can explode quickly...
    // if there is enough memory, a speedier version is used, that replaced the scatter with a direct pack from the initial oversized allocation!
    uint32_t *d_oversized_neighbors = nullptr;
    dim_t init_max_neighbors = (dim_t)std::ceil(OVERSIZED_SIZE_MULTIPLIER * (float)max_neighbors);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    // check if there could be space to allocate both oversized neighbors and final neighbors at once; with no better guess, use 'max_neighbors' to estimate the final neighbors size...
    bool direct_scatter_neighbors = (num_nodes * init_max_neighbors /*oversized*/ + num_nodes * max_neighbors /*final upper bound*/) * sizeof(uint32_t) + num_nodes * sizeof(dim_t) /*offsets*/ < free_bytes;
    // no pack? can spare space in the oversized buffer equal to the amount of shared memory used for fast deduping
    if (!direct_scatter_neighbors) init_max_neighbors = init_max_neighbors > SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE ? init_max_neighbors - SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE : 0;
    init_max_neighbors = max(init_max_neighbors, (dim_t)GM_MIN_BLOCK_DEDUPE_BUFFER_SIZE);
    if (num_nodes * init_max_neighbors * sizeof(uint32_t) > (1ull << 32))
        std::cout << "Allocating " << std::fixed << std::setprecision(1) << (float)(num_nodes * init_max_neighbors * sizeof(uint32_t)) / (1 << 30) << " GB for neighbors deduplication ...\n";
    CUDA_CHECK(cudaMalloc(&d_oversized_neighbors, num_nodes * init_max_neighbors * sizeof(uint32_t))); // space for spilling deduplication hash-sets
    CUDA_CHECK(cudaMalloc(&d_neighbors_offsets, (num_nodes + 1) * sizeof(dim_t))); // node -> neighbors set start idx in d_neighbors
    thrust::device_ptr<dim_t> t_neigh_offsets(d_neighbors_offsets);
    // launch configuration - neighborhoods count kernel
    blocks = num_nodes;
    threads_per_block = 256; // 256/32 -> 8 warps per block
    std::cout << "Running neighborhoods count kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    // launch - neighborhoods count kernel
    neighborhoods_count_kernel<<<blocks, threads_per_block>>>(
        d_hedges,
        d_hedges_offsets,
        d_touching,
        d_touching_offsets,
        num_nodes,
        init_max_neighbors,
        direct_scatter_neighbors,
        d_oversized_neighbors,
        d_neighbors_offsets
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!direct_scatter_neighbors) CUDA_CHECK(cudaFree(d_oversized_neighbors)); // no pack? free oversized immediately
    // correct the max neighbors count estimate
    auto actual_max_neighbors = thrust::max_element(t_neigh_offsets, t_neigh_offsets + num_nodes + 1);
    dim_t actual_max_neighbors_offset = static_cast<dim_t>(actual_max_neighbors - t_neigh_offsets);
    CUDA_CHECK(cudaMemcpy(&max_neighbors, d_neighbors_offsets + actual_max_neighbors_offset, sizeof(dim_t), cudaMemcpyDeviceToHost));
    std::cout << "Max neighbors estimate corrected to " << max_neighbors << "\n";
    // compute final offsets
    thrust::exclusive_scan(t_neigh_offsets, t_neigh_offsets + (num_nodes + 1), t_neigh_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
    dim_t total_neighbors;
    CUDA_CHECK(cudaMemcpy(&total_neighbors, d_neighbors_offsets + num_nodes, sizeof(dim_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&d_neighbors, total_neighbors * sizeof(uint32_t))); // contigous neighborhood sets array
    if (direct_scatter_neighbors) {
        // pack oversized neighbors in their final tight-fit subarrays
        // launch configuration - neighborhoods pack kernel
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = num_nodes ; // 1 warp per node
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        std::cout << "Running neighborhoods pack kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // launch - neighborhoods scatter kernel
        pack_segments<<<blocks, threads_per_block>>>(
            d_oversized_neighbors,
            d_neighbors_offsets,
            num_nodes,
            init_max_neighbors,
            d_neighbors
        );
    } else {
        // write neighbors at their correct offset
        // launch configuration - neighborhoods scatter kernel
        blocks = num_nodes;
        threads_per_block = 256; // 256/32 -> 8 warps per block
        std::cout << "Running neighborhoods scatter kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // launch - neighborhoods scatter kernel
        neighborhoods_scatter_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            d_touching,
            d_touching_offsets,
            num_nodes,
            d_neighbors_offsets,
            d_neighbors
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (direct_scatter_neighbors) CUDA_CHECK(cudaFree(d_oversized_neighbors)); // pack? free oversized afterwards


    // returns the number of partitions and the pointer to the final partitions device buffer
    std::function<std::tuple<uint32_t, uint32_t*>(const uint32_t, const uint32_t, uint32_t*&, dim_t*&, dim_t, uint32_t*&, dim_t*&, dim_t, uint32_t*&, uint32_t*&)> coarsen_refine_uncoarsen = [&](
        const uint32_t level_idx,
        const uint32_t curr_num_nodes,
        uint32_t*& d_hedges,
        dim_t*& d_hedges_offsets,
        dim_t hedges_size,
        uint32_t*& d_touching,
        dim_t*& d_touching_offsets,
        dim_t touching_size,
        uint32_t*& d_inbound_count,
        uint32_t*& d_nodes_sizes
    ) { // this is a lambda
        std::cout << "Coarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

        /*
        * Flow:
        * 1) coarsen
        *   - propose valid candidate node pairs
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
        *   - propose refinement moves in isolation and rank them
        *   - compute per-move gain as if applied in sequence
        *   - compute per-move validity via a prefix sum of the # of invalid partitions
        *     when applying the sequence of size, hedge, and inbound set events up to its rank
        *   - apply the highest-gain valid subsequence of moves
        *   => return final partitioning to the outer level
        */

        /*
        * Buffers allocated on (and local to) each level:
        * - d_groups
        * - d_ungroups, d_ungroups_offsets
        * - all event buffers for constraint checks
        * Buffers constructed anew before (and passed as args to) each level:
        * - d_hedges, d_hedges_offsets
        * - d_touching, d_touching_offsets, d_inbound_count
        * - d_nodes_sizes / d_groups_sizes
        * Buffers (constructed by and) returned from each level:
        * - d_partitions
        * Buffers updated (globally) in-place after each level:
        * - d_pairs
        * - d_u_scores, d_f_scores
        * - d_slots
        * - d_ranks
        * - d_neighbors, d_neighbors_offsets
        * - d_partitions_sizes
        * - d_pins_per_partitions
        * - d_partitions_inbound_sizes
        * Untouched buffers:
        * - d_hedge_weights
        *
        * TODO:
        * - use (multi)function bits to coarsen/uncoarsen hedges in-place
        * - use (multi)function bits to coarsen/uncoarsen hedges in-place
        * - DO NOT DO THE ABOVE for neighbors (not used during refinement)
        *
        * TODO: could remove some (or even all) the synchronizes
        */

        /*
        * Constraint checks:
        * - coarsening only proposes pairs that, individually, would not violated constraints if grouped
        * - grouping more than two nodes checks if the larger group still fits constraints (TODO)
        * - refinement proposes moves that do not violated constraints if applied in isolation
        * - refinement selects the best subsequence of moves that ends on a valid state
        */

        // zero-out candidates kernel's outputs
        // TODO: could just init. up to curr_num_nodes
        CUDA_CHECK(cudaMemset(d_pairs, 0xFF, num_nodes * sizeof(uint32_t) * MAX_CANDIDATES)); // 0xFF -> UINT32_MAX
        // NOTE: no need to init. "d_u_scores" if we use "d_pairs" to see which locations are valid
        
        // launch configuration - candidates kernel
        // NOTE: choose threads_per_block multiple of WARP_SIZE
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = curr_num_nodes ; // 1 warp per node
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        bytes_per_warp = 2 * HIST_SIZE * sizeof(uint32_t);
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
            d_inbound_count,
            d_hedge_weights,
            d_nodes_sizes,
            curr_num_nodes,
            d_pairs,
            d_u_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // =============================
        // print some temporary results
        #if VERBOSE
        std::vector<uint32_t> pairs_tmp(curr_num_nodes * MAX_CANDIDATES);
        std::vector<uint32_t> scores_tmp(curr_num_nodes * MAX_CANDIDATES);
        std::vector<std::set<uint32_t>> candidates_count(MAX_CANDIDATES);
        CUDA_CHECK(cudaMemcpy(pairs_tmp.data(), d_pairs, curr_num_nodes * sizeof(uint32_t) * MAX_CANDIDATES, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scores_tmp.data(), d_u_scores, curr_num_nodes * sizeof(uint32_t) * MAX_CANDIDATES, cudaMemcpyDeviceToHost));
        std::cout << "Pairing results:";
        for (uint32_t i = 0; i < curr_num_nodes; ++i) {
            if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH))
                std::cout << "\n  node " << i << " ->";
            for (uint32_t j = 0; j < MAX_CANDIDATES; ++j) {
                float score = ((float)scores_tmp[i * MAX_CANDIDATES + j])/FIXED_POINT_SCALE;
                uint32_t target = pairs_tmp[i * MAX_CANDIDATES + j];
                candidates_count[j].insert(target);
                if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH)) {
                    if (target == UINT32_MAX) std::cout << " (" << j << " target=none score=none)";
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
        std::vector<uint32_t>().swap(scores_tmp);
        std::vector<std::set<uint32_t>>().swap(candidates_count);
        #endif
        // =============================

        // zero-out grouping kernel's outputs
        slot init_slot; init_slot.id = UINT32_MAX; init_slot.score = 0u;
        thrust::device_ptr<slot> d_slots_ptr(d_slots);
        // TODO: could lower to just curr_num_nodes
        thrust::fill(d_slots_ptr, d_slots_ptr + curr_num_nodes * MAX_GROUP_SIZE, init_slot); // upper 32 bits to 0x00, lower 32 to 0xFF
        CUDA_CHECK(cudaMemset(d_dp_scores, 0x00, num_nodes * sizeof(dp_score)));
        
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
        uint32_t num_repeats = 1;
        if (blocks > max_blocks) {
            num_repeats = (blocks + max_blocks - 1) / max_blocks;
            std::cout << "NOTE: grouping kernel required blocks=" << blocks << ", but max-blocks=" << max_blocks << ", setting repeats=" << num_repeats << " ...\n";
            blocks = (blocks + num_repeats - 1) / num_repeats;
            if (num_repeats > MAX_REPEATS) {
                std::cout << "ABORTING: grouping kernel required repeats=" << num_repeats << ", but max-repeats=" << MAX_REPEATS << " !!\n";
                abort();
            }
        }
        // launch - grouping kernel
        std::cout << "Running grouping kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        void *kernel_args[] = {
            (void*)&d_pairs,
            (void*)&d_u_scores,
            (void*)&d_nodes_sizes,
            (void*)&curr_num_nodes,
            (void*)&num_repeats,
            (void*)&d_slots,
            (void*)&d_dp_scores,
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
        const uint32_t new_num_nodes = t_headflags.back() + 1;
        // ======================================
        // prepare this level's cumulative groups sizes
        // NOTE: "node sizes" = size of the nodes that entered this level, "group sizes" = cumulative size of groups constructed on this level
        uint32_t *d_groups_sizes = nullptr;
        CUDA_CHECK(cudaMalloc(&d_groups_sizes, new_num_nodes * sizeof(uint32_t))); // group_sizes[group id] = sum of sizes of all nodes in that group
        // extra step: compute cumulative group sizes
        thrust::device_ptr<uint32_t> t_groups_sizes(d_groups_sizes);
        thrust::device_ptr<const uint32_t> t_nodes_sizes(d_nodes_sizes);
        // premute node sizes in "sorted-by-group" order, using indices that already reflect such ordering ( t_indices[i] tells which original idx got sorted in position i )
        auto nodes_sizes_values_begin = thrust::make_permutation_iterator(t_nodes_sizes, t_indices.begin());
        // reduce (sum) nodes_sizes inside each group (group = key, marked by having the same headflag, that by now corresponds to the zero-based group id) ( headflags[i] is the new group ID for sorted position i )
        thrust::reduce_by_key(t_headflags.begin(), t_headflags.end(), nodes_sizes_values_begin, thrust::make_discard_iterator(), t_groups_sizes);
        // => now "d_groups_sizes[idx]" holds the sum of nodes_size over all nodes in group idx, for idx in [0, new_num_nodes)
        // ======================================
        // scatter the new ids back to original positions using the sequence; for sorted position i, original index is t_indices[i]; we want: d_groups[t_indices[i]] = t_headflags[i]
        thrust::scatter(t_headflags.begin(), t_headflags.end(), t_indices.begin(), t_groups);
        // if the number of groups has reached the required threshold, they become the partitions
        // => now "d_groups[idx]" contains the new zero-based group ID for every node

        // ======================================
        // base case, return inital partitioning
        // TODO: could increase the threshold and instead of "become the partitions" run a host-side robust partitioning algorithm
        //       => what this does now is equivalent to using the coarsening algorithm also as the algorithm to perform the initial partitioning
        // NOTE: with the current setup, we stop as soon as we clear "max_parts" and let refinement eventually empty some if they are too many
        //       => and alternative solution could be to always wait for "new_num_nodes == curr_num_nodes" and then enforce "max_parts" to spot failures
        if (new_num_nodes <= target_parts || new_num_nodes == curr_num_nodes) {
            // HERE we repurpose the coarsening routine as the routine for initial partitions:
            // - num_partitions = new_num_nodes
            // - partitions = groups

            // NOTE: d_partitions eventually will coincide with the innermost group each node was part of + refinement moves
            //       => the innermost nodes (groups) count is also the number of partitions

            // NOTE: just like groups, partitions need to ordered, as they be used as indices; however, partitions are few, and if one becomes
            //       empty we can just discard its index and leave a few empty spots in the data structures, it's cheaper to compress at the end

            // neighbors are no longer needed after coarsening is done
            CUDA_CHECK(cudaFree(d_neighbors));
            CUDA_CHECK(cudaFree(d_neighbors_offsets));

            // prepare initial partition sizes
            // NOTE: current groups become the partitions, and so group sizes become partition sizes
            d_partitions_sizes = d_groups_sizes; // partitions_sizes[idx] -> how many nodes (by size) are in the partition

            // NOTE: the inbound counters per partition are just the transposed of pins per partition! No need to compute them separately!
            CUDA_CHECK(cudaMalloc(&d_pins_per_partitions, num_hedges * new_num_nodes * sizeof(uint32_t))); // hedge * num_partitions + partition -> count of pins of "hedge" in that "partition"
            CUDA_CHECK(cudaMalloc(&d_partitions_inbound_sizes, new_num_nodes * sizeof(uint32_t))); // partition -> distinct inbound hedges count for "partition"

            // base case, reached the target number of partitions
            if (new_num_nodes <= target_parts) {
                std::cout << "Minimal initial partitioning built at level " << level_idx << ", remaining nodes=" << curr_num_nodes << ", number of partitions=" << new_num_nodes << "\n";
            } else if (new_num_nodes <= max_parts) {
                std::cout << "Initial partitioning built at level " << level_idx << ", remaining nodes=" << curr_num_nodes << ", number of partitions=" << new_num_nodes << "\n";
                std::cout << "WARNING: the partitioning is valid, but didn't reach the minimal number of partitions (" << target_parts << ")...\n";
            } else { // base case, failure to coarsen further
                std::cout << "FAILED TO COARSEN FURTHER at level " << level_idx << ", remaining nodes=" << curr_num_nodes << " number of partitions=" << new_num_nodes << " max allowed partitions=" << max_parts << "\n";
                std::cout << "WARNING: falling back to returning current groups as individual partitions...\n";
            }

            return std::make_tuple(new_num_nodes, d_groups);
        }
        // ======================================

        // prepare this level's uncoarsening data structures
        uint32_t *d_ungroups = nullptr;
        dim_t *d_ungroups_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ungroups, curr_num_nodes * sizeof(uint32_t))); // ungroups[ungroups_offsets[group id] + i] -> the group's i-th node (its original idx)
        CUDA_CHECK(cudaMalloc(&d_ungroups_offsets, (1 + new_num_nodes) * sizeof(dim_t))); // ungroups_offsets[node idx] -> node's group id (zero-based)
        
        // build reverse multifunction from groups to their original nodes
        // from above, t_indices is the list of node idxs sorted by their group id, hence, the reverse list is simply t_indices, we just need to compute the offsets to reach, from each group id, its original nodes
        CUDA_CHECK(cudaMemcpy(d_ungroups, thrust::raw_pointer_cast(t_indices.data()), curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        thrust::device_ptr<dim_t> t_ungroups_offsets(d_ungroups_offsets);
        // predicate to detect group starts: is_group_start(i) = (i == 0) || (headflags[i] != headflags[i-1])
        auto is_group_start = [heads = t_headflags.begin()] __device__ (uint32_t i) { return (i == 0) || (heads[i] != heads[i - 1]); };
        // counting iterator over sorted positions
        auto t_iter_begin = thrust::make_counting_iterator<uint32_t>(0);
        auto t_iter_end = thrust::make_counting_iterator<uint32_t>(curr_num_nodes);
        // copy positions of (only) group starts directly into ungroups_offsets
        thrust::copy_if(t_iter_begin, t_iter_end, t_iter_begin, t_ungroups_offsets, is_group_start);
        // append the (curr_num_nodes + 1)-th value
        dim_t dim_t_curr_num_nodes = (dim_t)curr_num_nodes;
        CUDA_CHECK(cudaMemcpy(d_ungroups_offsets + new_num_nodes, &dim_t_curr_num_nodes, sizeof(dim_t), cudaMemcpyHostToDevice));
        // free up thrust vectors
        //thrust::device_vector<uint32_t>().swap(t_indices); // DO NOT FREE THIS UP! We need it later for REFINEMENT!
        thrust::device_vector<uint32_t>().swap(t_headflags);

        // =============================
        // print some temporary results
        #if VERBOSE
        std::vector<uint32_t> groups_tmp(curr_num_nodes);
        std::vector<uint32_t> groups_sizes_tmp(new_num_nodes);
        CUDA_CHECK(cudaMemcpy(pairs_tmp.data(), d_pairs, curr_num_nodes * sizeof(uint32_t) * MAX_CANDIDATES, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(groups_tmp.data(), d_groups, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(groups_sizes_tmp.data(), d_groups_sizes, new_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::unordered_map<uint32_t, int> groups_count;
        std::cout << "Grouping results:\n";
        for (uint32_t i = 0; i < curr_num_nodes; ++i) {
            uint32_t group = groups_tmp[i];
            uint32_t group_size = groups_sizes_tmp[group];
            groups_count[group]++;
            if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH)) {
                std::cout << "  node " << i << " ->";
                for (uint32_t j = 0; j < MAX_CANDIDATES; ++j) {
                    uint32_t target = pairs_tmp[i * MAX_CANDIDATES + j];
                    if (target == UINT32_MAX) std::cout << " (" << j << " target=none)";
                    else std::cout << " (" << j << " target=" << target << ")";
                }
                std::cout << " group=" << group << " group_size=" << group_size << "\n";
            }
        }
        for (uint32_t i = 0; i < new_num_nodes; ++i) {
            uint32_t group_size = groups_sizes_tmp[i];
            if (group_size > h_max_nodes_per_part)
                std::cout << "  WARNING, max group size constraint (" << h_max_nodes_per_part << ") violated by group=" << i << " with group_size=" << group_size << " !!\n";
        }
        int max_gs = groups_count.empty() ? 0 : std::max_element(groups_count.begin(), groups_count.end(), [](auto &a, auto &b){ return a.second < b.second; })->second;
        std::cout << "Groups count: " << groups_count.size() << ", Max group size: " << max_gs << "\n";
        std::vector<uint32_t>().swap(pairs_tmp);
        std::vector<uint32_t>().swap(groups_tmp);
        std::unordered_map<uint32_t, int>().swap(groups_count);
        #endif
        // =============================

        // update the maximum hedges and neighbors estimate by scaling it by new_num_nodes/curr_num_nodes
        float scale = (float)new_num_nodes / curr_num_nodes;
        max_hedge_size = std::ceil(max_hedge_size * scale);
        max_neighbors = std::ceil(max_neighbors * scale);
        std::cout << "Max hedges estimate updated to " << max_hedge_size << ", neighbors estimate updated to " << max_neighbors << "\n";

        // prepare coarse neighbors buffers
        uint32_t *d_coarse_neighbors = nullptr;
        uint32_t *d_coarse_oversized_neighbors = nullptr;
        dim_t *d_coarse_neighbors_offsets = nullptr;
        dim_t curr_max_neighbors = (dim_t)(OVERSIZED_SIZE_MULTIPLIER * (float)max_neighbors); // add a bit of safety-room to compensate for the flat scaling by 'new_num_nodes / curr_num_nodes'
        // if there is enough memory for the full oversized buffer, SM dischard included, a speedier version is used, that replaced the scatter with a direct pack from the initial oversized allocation!
        cudaMemGetInfo(&free_bytes, &total_bytes);
        // NOTE: no need to check if there could be space to allocate both oversized neighbors and final neighbors at once, if the oversized fits, then the new neighbors are allocated either after the oversized is freed, or after the original neighbors are freed
        bool direct_scatter_coarse_neighbors = (curr_num_nodes * curr_max_neighbors /*oversized (SM included)*/ + new_num_nodes * max_neighbors /*final upper bound*/) * sizeof(uint32_t) + new_num_nodes * sizeof(dim_t) /*offsets*/ < free_bytes;
        // no pack? can spare space in the oversized buffer equal to the amount of shared memory used for fast deduping
        if (!direct_scatter_coarse_neighbors) curr_max_neighbors = curr_max_neighbors * MAX_GROUP_SIZE > MAX_SM_WARP_DEDUPE_BUFFER_SIZE ? curr_max_neighbors - MAX_SM_WARP_DEDUPE_BUFFER_SIZE / MAX_GROUP_SIZE : 0; // save the spaced for the duplicates caught in SM
        curr_max_neighbors = max(curr_max_neighbors, (dim_t)MIN_GM_WARP_DEDUPE_BUFFER_SIZE); // just some ensurance...
        if (curr_num_nodes * curr_max_neighbors * sizeof(uint32_t) > (1ull << 32))
            std::cout << "Allocating " << std::fixed << std::setprecision(1) << (float)(curr_num_nodes * curr_max_neighbors * sizeof(uint32_t)) / (1 << 30) << " GB for neighbors deduplication ...\n";
        CUDA_CHECK(cudaMalloc(&d_coarse_oversized_neighbors, curr_num_nodes * curr_max_neighbors * sizeof(uint32_t))); // space for spilling deduplication hash-sets
        CUDA_CHECK(cudaMalloc(&d_coarse_neighbors_offsets, (1 + new_num_nodes) * sizeof(dim_t))); // NOTE: the number nodes decreases!
        CUDA_CHECK(cudaMemset(d_coarse_neighbors_offsets, 0x00, sizeof(dim_t))); // init. the first offset at 0
        thrust::device_ptr<dim_t> t_coarse_neigh_offsets(d_coarse_neighbors_offsets);
        // launch configuration - coarsening kernel (neighbors - count)
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = new_num_nodes ; // 1 warp per group
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        bytes_per_warp = MAX_SM_WARP_DEDUPE_BUFFER_SIZE * sizeof(uint32_t);
        shared_bytes = warps_per_block * bytes_per_warp;
        // launch - coarsening kernel (neighbors - count)
        std::cout << "Running coarsening kernel (neighbors - count) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_neighbors_count<<<blocks, threads_per_block, shared_bytes>>>(
            d_neighbors,
            d_neighbors_offsets,
            d_groups,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            curr_max_neighbors,
            direct_scatter_coarse_neighbors,
            d_coarse_oversized_neighbors,
            d_coarse_neighbors_offsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!direct_scatter_coarse_neighbors) CUDA_CHECK(cudaFree(d_coarse_oversized_neighbors));
        if (direct_scatter_coarse_neighbors) { CUDA_CHECK(cudaFree(d_neighbors)); CUDA_CHECK(cudaFree(d_neighbors_offsets)); }
        // correct the max neighbors count estimate
        auto actual_coarse_max_neighbors = thrust::max_element(t_coarse_neigh_offsets, t_coarse_neigh_offsets + new_num_nodes + 1);
        dim_t actual_coarse_max_neighbors_offset = static_cast<dim_t>(actual_coarse_max_neighbors - t_coarse_neigh_offsets);
        CUDA_CHECK(cudaMemcpy(&max_neighbors, d_coarse_neighbors_offsets + actual_coarse_max_neighbors_offset, sizeof(dim_t), cudaMemcpyDeviceToHost));
        std::cout << "Max neighbors estimate corrected to " << max_neighbors << "\n";
        // compute final offsets
        thrust::inclusive_scan(t_coarse_neigh_offsets, t_coarse_neigh_offsets + (new_num_nodes + 1), t_coarse_neigh_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
        dim_t new_neighbors_size = 0; // last value in the inclusive scan = full reduce = total number of neighbors among all sets
        CUDA_CHECK(cudaMemcpy(&new_neighbors_size, d_coarse_neighbors_offsets + new_num_nodes, sizeof(dim_t), cudaMemcpyDeviceToHost));
        // NOTE: rebuilding neighbors from scratch makes no sense: if the "oversized" buffer could fit, no reason the new neighbors shouldn't!
        // this alloc should never fail, since it occurs in the space left by either the oversized buffer or previous neighbors
        CUDA_CHECK(cudaMalloc(&d_coarse_neighbors, new_neighbors_size * sizeof(uint32_t)));
        if (direct_scatter_coarse_neighbors) {
            // pack oversized coarse neighbors in their final tight-fit subarrays
            // launch configuration - coarsening kernel (neighbors - pack)
            threads_per_block = 128; // 128/32 -> 4 warps per block
            warps_per_block = threads_per_block / WARP_SIZE;
            num_warps_needed = new_num_nodes; // 1 warp per group
            blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            std::cout << "Running coarsening kernel (neighbors - pack) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            // launch - neighborhoods scatter kernel
            pack_segments_varsize<<<blocks, threads_per_block>>>(
                d_coarse_oversized_neighbors,
                d_ungroups_offsets, // once more, ungroup offsets provide the offsets for the oversized buffer too
                d_coarse_neighbors_offsets,
                new_num_nodes,
                curr_max_neighbors,
                d_coarse_neighbors
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_coarse_oversized_neighbors));
        } else {
            // launch configuration - coarsening kernel (neighbors - scatter)
            threads_per_block = 128; // 128/32 -> 4 warps per block
            warps_per_block = threads_per_block / WARP_SIZE;
            num_warps_needed = new_num_nodes ; // 1 warp per group
            blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - coarsening kernel (neighbors - scatter)
            std::cout << "Running coarsening kernel (neighbors - scatter) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            apply_coarsening_neighbors_scatter<<<blocks, threads_per_block, shared_bytes>>>(
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
        }
        d_neighbors = d_coarse_neighbors;
        d_neighbors_offsets = d_coarse_neighbors_offsets;

        // prepare coarse hedges buffers
        uint32_t *d_coarse_hedges = nullptr;
        uint32_t *d_coarse_oversized_hedges = nullptr;
        dim_t *d_coarse_hedges_offsets = nullptr;
        dim_t curr_max_hedge_size = (dim_t)(OVERSIZED_SIZE_MULTIPLIER * (float)max_hedge_size);
        curr_max_hedge_size = curr_max_hedge_size > MAX_SM_WARP_DEDUPE_BUFFER_SIZE ? curr_max_hedge_size - MAX_SM_WARP_DEDUPE_BUFFER_SIZE : 0;
        curr_max_hedge_size = max(curr_max_hedge_size, (dim_t)MIN_GM_WARP_DEDUPE_BUFFER_SIZE);
        if (num_hedges * curr_max_hedge_size * sizeof(uint32_t) > (1ull << 32))
            std::cout << "Allocating " << std::fixed << std::setprecision(1) << (float)(num_hedges * curr_max_hedge_size * sizeof(uint32_t)) / (1 << 30) << " GB for hedges deduplication ...\n";
        CUDA_CHECK(cudaMalloc(&d_coarse_oversized_hedges, num_hedges * curr_max_hedge_size * sizeof(uint32_t))); // space for spilling deduplication hash-sets
        CUDA_CHECK(cudaMalloc(&d_coarse_hedges_offsets, (1 + num_hedges) * sizeof(dim_t))); // NOTE: the number of hedges never decreases (for now), unlike that of nodes!
        CUDA_CHECK(cudaMemset(d_coarse_hedges_offsets, 0x00, sizeof(dim_t))); // init. the first offset at 0
        thrust::device_ptr<dim_t> t_coarse_hedges_offsets(d_coarse_hedges_offsets);
        // launch configuration - coarsening kernel (hedges - count)
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = num_hedges; // 1 warp per hedge
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        bytes_per_warp = MAX_SM_WARP_DEDUPE_BUFFER_SIZE * sizeof(uint32_t);
        shared_bytes = warps_per_block * bytes_per_warp;
        // launch - coarsening kernel (hedges - count)
        std::cout << "Running coarsening kernel (hedges - count) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_hedges_count<<<blocks, threads_per_block, shared_bytes>>>(
            d_hedges,
            d_hedges_offsets,
            d_groups,
            num_hedges,
            curr_max_hedge_size,
            d_coarse_oversized_hedges,
            d_coarse_hedges_offsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_coarse_oversized_hedges));
        // correct the max hedge size estimate
        auto actual_coarse_max_hedge_size = thrust::max_element(t_coarse_hedges_offsets, t_coarse_hedges_offsets + num_hedges + 1);
        dim_t actual_coarse_max_hedge_size_offset = static_cast<dim_t>(actual_coarse_max_hedge_size - t_coarse_hedges_offsets);
        CUDA_CHECK(cudaMemcpy(&max_hedge_size, d_coarse_hedges_offsets + actual_coarse_max_hedge_size_offset, sizeof(dim_t), cudaMemcpyDeviceToHost));
        std::cout << "Max hedges estimate corrected to " << max_hedge_size << "\n";
        // NOTE: the scan wants the last index EXCLUDED, while the memcopy wants the last index exactly! That's why we use here the +1, and not later!
        thrust::inclusive_scan(t_coarse_hedges_offsets, t_coarse_hedges_offsets + (num_hedges + 1), t_coarse_hedges_offsets); // in-place exclusive scan (the last element collects the full reduce)
        dim_t new_hedges_size = 0; // last value in the inclusive scan = full reduce = total number of pins among all hedges
        CUDA_CHECK(cudaMemcpy(&new_hedges_size, d_coarse_hedges_offsets + num_hedges, sizeof(dim_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMalloc(&d_coarse_hedges, new_hedges_size * sizeof(uint32_t)));
        // launch configuration - coarsening kernel (hedges - scatter) - same as coarsening kernel (hedges - count)
        // launch - coarsening kernel (hedges - scatter)
        std::cout << "Running coarsening kernel (hedges - scatter) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_hedges_scatter<<<blocks, threads_per_block, shared_bytes>>>(
            d_hedges,
            d_hedges_offsets,
            d_groups,
            num_hedges,
            d_coarse_hedges_offsets,
            d_coarse_hedges
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // prepare coarse touching buffers
        uint32_t *d_coarse_touching = nullptr;
        uint32_t *d_coarse_touching_buffer = nullptr;
        dim_t *d_coarse_touching_offsets = nullptr;
        uint32_t *d_coarse_inbound_count = nullptr;
        CUDA_CHECK(cudaMalloc(&d_coarse_touching_offsets, (1 + new_num_nodes) * sizeof(dim_t))); // NOTE: the number nodes decreases!
        CUDA_CHECK(cudaMemset(d_coarse_touching_offsets, 0x00, (1 + new_num_nodes) * sizeof(dim_t))); // remember to leave the first offset at 0
        CUDA_CHECK(cudaMalloc(&d_coarse_inbound_count, new_num_nodes * sizeof(uint32_t)));
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
        thrust::device_ptr<dim_t> t_coarse_touching_offsets(d_coarse_touching_offsets);
        thrust::inclusive_scan(t_coarse_touching_offsets, t_coarse_touching_offsets + (new_num_nodes + 1), t_coarse_touching_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
        dim_t new_touching_size = 0; // last value in the inclusive scan = full reduce = total number of touching hedges among all sets
        CUDA_CHECK(cudaMemcpy(&new_touching_size, d_coarse_touching_offsets + new_num_nodes, sizeof(dim_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMalloc(&d_coarse_touching, new_touching_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_coarse_touching_buffer, new_touching_size * sizeof(uint32_t)));
        // launch configuration - coarsening kernel (touching - scatter - inbound)
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = new_num_nodes; // 1 warp per group
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        bytes_per_warp = MAX_SM_WARP_DEDUPE_BUFFER_SIZE * sizeof(uint32_t);
        shared_bytes = warps_per_block * bytes_per_warp;
        // launch - coarsening kernel (touching - scatter - inbound)
        std::cout << "Running coarsening kernel (touching - scatter - inbound) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_touching_scatter_inbound<<<blocks, threads_per_block, shared_bytes>>>(
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            d_coarse_touching_offsets,
            d_coarse_touching,
            d_coarse_inbound_count
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // sort each inbound touching set
        cub::DoubleBuffer<uint32_t> c_coarse_touching_double_buffer(d_coarse_touching, d_coarse_touching_buffer);
        void* c_temp_storage = nullptr;
        size_t c_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(c_temp_storage, c_storage_bytes, c_coarse_touching_double_buffer, new_touching_size, new_num_nodes, d_coarse_touching_offsets, d_coarse_touching_offsets + 1, /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream*/0);
        std::cout << "CUB segmented sort requiring " << std::fixed << std::setprecision(3) << (float)(new_touching_size * sizeof(uint32_t)) / (1 << 30) << " GB of pong-buffer and " << std::fixed << std::setprecision(3) << ((float)c_storage_bytes) / (1 << 20) << " MB of temporary storage ...\n";
        cudaMalloc(&c_temp_storage, c_storage_bytes);
        cub::DeviceSegmentedRadixSort::SortKeys(c_temp_storage, c_storage_bytes, c_coarse_touching_double_buffer, new_touching_size, new_num_nodes, d_coarse_touching_offsets, d_coarse_touching_offsets + 1, /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream*/0);
        if (c_coarse_touching_double_buffer.Current() != d_coarse_touching) {
            uint32_t* tmp = d_coarse_touching_buffer;
            d_coarse_touching_buffer = d_coarse_touching;
            d_coarse_touching = tmp;
        }
        // launch configuration - coarsening kernel (touching - scatter - outbound) - same as coarsening kernel (touching - scatter - inbound)
        // launch - coarsening kernel (touching - scatter - outbound)
        std::cout << "Running coarsening kernel (touching - scatter - outbound) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_touching_scatter_outbound<<<blocks, threads_per_block, shared_bytes>>>(
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            d_coarse_touching_offsets,
            d_coarse_inbound_count,
            d_coarse_touching
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_coarse_touching_buffer));
        CUDA_CHECK(cudaFree(c_temp_storage));
        
        // spill non-coarse data structures to host
        std::vector<uint32_t> h_hedges;
        std::vector<dim_t> h_hedges_offsets;
        std::vector<uint32_t> h_touching;
        std::vector<dim_t> h_touching_offsets;
        if (level_idx < SAVE_MEMORY_UP_TO_LEVEL) {
            // TODO: make these async, move everything out of the default stream and use a "compute" and a "transfer" stream
            h_hedges.reserve(hedges_size);
            h_hedges_offsets.reserve(num_hedges + 1);
            h_touching.reserve(touching_size);
            h_touching_offsets.reserve(curr_num_nodes + 1);
            std::cout << "Spilling " << std::fixed << std::setprecision(3) << (float)((hedges_size + touching_size) * sizeof(uint32_t) + (num_hedges + 1 + curr_num_nodes + 1) * sizeof(dim_t)) / (1 << 30) << " GB from device to host at level " << level_idx << " ...\n";
            CUDA_CHECK(cudaMemcpy(h_hedges.data(), d_hedges, hedges_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_hedges_offsets.data(), d_hedges_offsets, (num_hedges + 1) * sizeof(dim_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_touching.data(), d_touching, touching_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_touching_offsets.data(), d_touching_offsets, (curr_num_nodes + 1) * sizeof(dim_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_hedges));
            CUDA_CHECK(cudaFree(d_hedges_offsets));
            CUDA_CHECK(cudaFree(d_touching));
            CUDA_CHECK(cudaFree(d_touching_offsets));
        }
        
        // ======================================
        // recursive call, go down one more level
        auto [num_partitions, d_coarse_partitions] = coarsen_refine_uncoarsen(
            level_idx + 1,
            new_num_nodes,
            d_coarse_hedges,
            d_coarse_hedges_offsets,
            new_hedges_size,
            d_coarse_touching,
            d_coarse_touching_offsets,
            new_touching_size,
            d_coarse_inbound_count,
            d_groups_sizes
        );
        // ======================================

        std::cout << "Uncoarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

        // un-spill non-coarse data structures to device
        if (level_idx < SAVE_MEMORY_UP_TO_LEVEL) {
            std::cout << "Unspilling " << std::fixed << std::setprecision(3) << (float)((hedges_size + touching_size) * sizeof(uint32_t) + (num_hedges + 1 + curr_num_nodes + 1) * sizeof(dim_t)) / (1 << 30) << " GB from host to device at level " << level_idx << " ...\n";
            CUDA_CHECK(cudaMalloc(&d_hedges, hedges_size * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_hedges_offsets, (num_hedges + 1) * sizeof(dim_t)));
            CUDA_CHECK(cudaMalloc(&d_touching, touching_size * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_touching_offsets, (curr_num_nodes + 1) * sizeof(dim_t)));
            CUDA_CHECK(cudaMemcpy(d_hedges, h_hedges.data(), hedges_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_hedges_offsets, h_hedges_offsets.data(), (num_hedges + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_touching, h_touching.data(), touching_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_touching_offsets, h_touching_offsets.data(), (curr_num_nodes + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
            std::vector<uint32_t>().swap(h_hedges);
            std::vector<dim_t>().swap(h_hedges_offsets);
            std::vector<uint32_t>().swap(h_touching);
            std::vector<dim_t>().swap(h_touching_offsets);
        }

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
        CUDA_CHECK(cudaFree(d_coarse_inbound_count));
        CUDA_CHECK(cudaFree(d_groups_sizes));
        CUDA_CHECK(cudaFree(d_coarse_partitions)); // allocated at the next inner level, freed here!

        // =============================
        // print some temporary results
        #if VERBOSE
        std::vector<uint32_t> partitions_tmp(curr_num_nodes);
        CUDA_CHECK(cudaMemcpy(partitions_tmp.data(), d_partitions, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::vector<uint32_t> partitions_sizes_tmp(num_partitions);
        CUDA_CHECK(cudaMemcpy(partitions_sizes_tmp.data(), d_partitions_sizes, num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::unordered_map<uint32_t, int> part_count;
        std::cout << "Partitioning results:\n";
        for (uint32_t i = 0; i < curr_num_nodes; ++i) {
            uint32_t part = partitions_tmp[i];
            part_count[part]++;
            if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH))
                std::cout << "  node " << i << " -> " << part << "\n";
        }
        for (uint32_t i = 0; i < num_partitions; ++i) {
            uint32_t part_size = partitions_sizes_tmp[i];
            if (part_size > h_max_nodes_per_part)
                std::cout << "  WARNING, max partition size constraint (" << h_max_nodes_per_part << ") violated by part=" << i << " with part_size=" << part_size << " !!\n";
        }
        int max_ps = part_count.empty() ? 0 : std::max_element(part_count.begin(), part_count.end(), [](auto &a, auto &b){ return a.second < b.second; })->second;
        std::cout << "Non-empty partitions count: " << part_count.size() << ", Max partition size: " << max_ps << "\n";
        std::vector<uint32_t>().swap(partitions_tmp);
        std::vector<uint32_t>().swap(partitions_sizes_tmp);
        std::unordered_map<uint32_t, int>().swap(part_count);
        #endif
        // =============================

        std::cout << "Refining level " << level_idx << ", remaining nodes=" << curr_num_nodes << " number of partitions=" << num_partitions << "\n";

        // prepare this level's pins per partition
        CUDA_CHECK(cudaMemset(d_pins_per_partitions, 0x00, num_hedges * num_partitions * sizeof(uint32_t)));
        // while computing pins per partition also compute the distinct inbound counts per partition (number of pins with a count > 0)
        CUDA_CHECK(cudaMemset(d_partitions_inbound_sizes, 0x00, num_partitions * sizeof(uint32_t)));
        
        // TODO: if you are struggling for memory, store both pins per partition in a compact form
        // TODO: compute both pins per partition (inbound only and not) via a highly optimized map+histogram pattern
        //       => map from edges of nodes to hedges of partitions, then histogram for each hedge in // (by key)

        // launch configuration - pins per partition kernel
        // TODO: could update this in-place instead of recomputing it each time by going over 'touching' for moved nodes when applying the refinement!
        // HARDER: if we change pins per partition to represent only destination (inbound) pins after we computed gains, also need to revert it to represent all pins...
        // => If we do this, uncomment "pins_per_partitions" in "fm_refinement_apply_kernel"
        // TODO: maybe it would be faster to build pins per partition with 'touching', by going one block per partition, 256 threads digesting touching hedge with
        //       an hash-map in shared memory, then dumped to global with one streak of atomics?
        // => call something like "apply_coarsening_touching_count" at the innermost level using partitions as groups to compute the initial pins per partition?
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
            d_pins_per_partitions,
            d_partitions_inbound_sizes // as of here, this will be incorrect (also including outbounds)
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // zero-out fm-ref gains kernel's outputs
        CUDA_CHECK(cudaMemset(d_pairs, 0xFF, curr_num_nodes * sizeof(uint32_t))); // 0xFF -> UINT32_MAX
        // NOTE: no need to init. "d_f_scores" if we use "d_pairs" to see which locations are valid

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
            d_nodes_sizes,
            d_partitions_sizes,
            num_hedges,
            curr_num_nodes,
            num_partitions,
            // NOTE: repurposing those from the candidates kernel!
            d_pairs, // -> moves: pairs[node] -> partition the node wants to join
            d_f_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // sort scores and build an array of ranks (node id -> his move's idx in sorted scores)
        //thrust::device_vector<int> t_indices(curr_num_nodes); // temporary sequence sorted alongside scores -> ALREADY DECLARED FOR COARSEING, reuse!
        thrust::sequence(t_indices.begin(), t_indices.end());
        uint32_t *d_ranks = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ranks, curr_num_nodes * sizeof(uint32_t))); // node -> number of touching hedges seen as of now
        thrust::device_ptr<uint32_t> t_ranks(d_ranks);
        thrust::device_ptr<float> t_scores(d_f_scores);
        // sort scores according to scores themselves and indices in the same way
        // use node ids as a tie-breaker when sorting moves
        auto rank_keys_begin = thrust::make_zip_iterator(thrust::make_tuple(t_scores, t_indices.begin()));
        auto rank_keys_end = rank_keys_begin + curr_num_nodes;
        thrust::sort(rank_keys_begin, rank_keys_end, [] __device__ (const thrust::tuple<float, uint32_t>& a, const thrust::tuple<float, uint32_t>& b) {
                float sa = thrust::get<0>(a), sb = thrust::get<0>(b);
                if (sa > sb) return true; // highest score first
                if (sa < sb) return false;
                return thrust::get<1>(a) < thrust::get<1>(b); // deterministic tie-break
        });
        thrust::scatter(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(curr_num_nodes), t_indices.begin(), t_ranks); // invert the permutation such that: ranks[original_index] = sorted_position
        // free up thrust vectors
        thrust::device_vector<uint32_t>().swap(t_indices);
        // launch configuration - fm-ref cascade kernel => same as "fm-ref gains kernel"
        // compute shared memory per block (bytes)
        bytes_per_warp = 0; //TODO
        shared_bytes = warps_per_block * bytes_per_warp;
        // launch - fm-ref cascade kernel
        std::cout << "Running fm-ref cascade kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
            d_f_scores
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // not re-sorting the scores array means you have the array ordered as per the initial scores,
        // but now, this scan updates the scores "as if all previous moves were applied"!
        thrust::inclusive_scan(t_scores, t_scores + curr_num_nodes, t_scores); // in-place (we don't need scores anymore anyway)
        // Remember: moves never get re-ranked (re-sorted) after the first time with in-isolation gains. Keep them like that and just find the valid sequence of maximum gain! This is an heuristics!
        // ======================================
        // extra step: compute moves validity by size (same HP as the kernel above: all previous higher-gain moves will be applied)
        // explode each move into two events, one decrementing and incrementing the size of the src and dst partition respectively
        // => seeing each move as two distinct events makes us able to identify sequences of useful events first, then moves
        uint32_t *d_size_events_partition = nullptr, *d_size_events_index = nullptr;
        int32_t *d_size_events_delta = nullptr;
        const uint32_t num_size_events = 2 * curr_num_nodes;
        CUDA_CHECK(cudaMalloc(&d_size_events_partition, num_size_events * sizeof(uint32_t))); // size_events_partition[ev] -> partition affected by the event
        CUDA_CHECK(cudaMalloc(&d_size_events_index, num_size_events * sizeof(uint32_t))); // size_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        CUDA_CHECK(cudaMalloc(&d_size_events_delta, num_size_events * sizeof(int32_t))); // size_events_delta[ev] -> size variation brought by the event
        thrust::device_ptr<uint32_t> t_size_events_partition(d_size_events_partition);
        thrust::device_ptr<uint32_t> t_size_events_index(d_size_events_index);
        thrust::device_ptr<int32_t> t_size_events_delta(d_size_events_delta);
        // launch configuration - build size events kernel
        threads_per_block = 128;
        num_threads_needed = curr_num_nodes; // 1 thread per move
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - build size events kernel
        std::cout << "Running build size events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // TODO: could filter out null moves (target = -1)?
        build_size_events_kernel<<<blocks, threads_per_block>>>(
            d_pairs,
            d_ranks,
            d_partitions,
            d_nodes_sizes,
            curr_num_nodes,
            d_size_events_partition,
            d_size_events_index,
            d_size_events_delta
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // sort events by (partition, rank) [in lexicographical order for the tuple] and carry size_events_delta along
        auto size_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_size_events_partition, t_size_events_index));
        auto size_events_key_end = size_events_key_begin + num_size_events;
        thrust::sort_by_key(size_events_key_begin, size_events_key_end, t_size_events_delta);
        // inclusive scan inside each key (= partition) on the event deltas => for each event we get the cumulative size delta for that partition at that point in the sequence
        thrust::inclusive_scan_by_key(t_size_events_partition, t_size_events_partition + num_size_events, t_size_events_delta, t_size_events_delta);
        // now mark moves that would violate size constraint if the sequence were to end on them
        uint32_t *d_valid_moves = nullptr;
        CUDA_CHECK(cudaMalloc(&d_valid_moves, curr_num_nodes * sizeof(uint32_t))); // valid_move[rank idx] -> 1 if applying all moves up to the idx one in the ordered sequence gives a valid state
        CUDA_CHECK(cudaMemset(d_valid_moves, 0u, curr_num_nodes * sizeof(uint32_t)));
        thrust::device_ptr<uint32_t> t_valid_moves(d_valid_moves);
        // launch configuration - flag size events kernel
        threads_per_block = 128;
        num_threads_needed = num_size_events; // 1 thread per event
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - flag size events kernel
        std::cout << "Running flag size events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        flag_size_events_kernel<<<blocks, threads_per_block>>>(
            d_size_events_partition,
            d_size_events_index,
            d_size_events_delta,
            d_partitions_sizes,
            num_size_events,
            d_valid_moves
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_size_events_partition));
        CUDA_CHECK(cudaFree(d_size_events_index));
        CUDA_CHECK(cudaFree(d_size_events_delta));
        // compute, as of each event, the cumulative number of partitions that are invalid by summing the count of those made/unmade invalid at each event
        thrust::inclusive_scan(t_valid_moves, t_valid_moves + curr_num_nodes, t_valid_moves);
        // ======================================
        // preparatory step: update pins per partition into inbound (only) pins partition
        // simultaneously, also correct the calculation for partitions_inbound_sizes by removing outbounds
        // launch configuration - inbound pins per partition kernel
        threads_per_block = 256;
        num_threads_needed = num_hedges; // 1 thread per hedge
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - inbound pins per partition kernel
        std::cout << "Running inbound pins per partition kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // NOTE: inbound-only version of the above used for constraints checks...
        inbound_pins_per_partition_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            d_partitions,
            num_hedges,
            num_partitions,
            d_pins_per_partitions, // from now it represents inbound sets only
            d_partitions_inbound_sizes
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // ======================================
        // extra step: compute moves validity by inbound set cardinality (same HP as the kernel above: all previous higher-gain moves will be applied)
        // explode each move into two events for every inbound hedge of the moved node, one decrementing and one incrementing the hedge's
        // occurrencies in the src partition's inbound set and dst partition's inbound set respectively
        // => results in n*h events (better than the n*h*p volume of conditions/counters we need to check)
        uint32_t *d_inbound_count_events_partition = nullptr, *d_inbound_count_events_index = nullptr, *d_inbound_count_events_hedge = nullptr;
        int32_t *d_inbound_count_events_delta = nullptr;
        const uint32_t num_inbound_count_events = 2 * touching_size; // TODO: this is slightly larger than truly needed, because we just use inbound hedges...
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_partition, num_inbound_count_events * sizeof(uint32_t))); // inbound_count_events_partition[ev] -> partition affected by the event
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_index, num_inbound_count_events * sizeof(uint32_t))); // inbound_count_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_hedge, num_inbound_count_events * sizeof(uint32_t))); // d_inbound_count_events_hedge[ev] -> hedge involved in the event
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_delta, num_inbound_count_events * sizeof(int32_t))); // inbound_count_events_delta[ev] -> inbound_count variation brought by the event
        CUDA_CHECK(cudaMemset(d_inbound_count_events_partition, 0xFF, num_inbound_count_events * sizeof(uint32_t))); // => use inbound_count_events_partition being UINT32_MAX to spot invalid events
        thrust::device_ptr<uint32_t> t_inbound_count_events_partition(d_inbound_count_events_partition);
        thrust::device_ptr<uint32_t> t_inbound_count_events_index(d_inbound_count_events_index);
        thrust::device_ptr<uint32_t> t_inbound_count_events_hedge(d_inbound_count_events_hedge);
        thrust::device_ptr<int32_t> t_inbound_count_events_delta(d_inbound_count_events_delta);
        // launch configuration - build hedge events kernel
        threads_per_block = 128; // 128/32 -> 4 warps per block
        warps_per_block = threads_per_block / WARP_SIZE;
        num_warps_needed = curr_num_nodes ; // 1 warp per move
        blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - build hedge events kernel
        std::cout << "Running build hedge events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // TODO: could filter out null moves (target = -1)?
        build_hedge_events_kernel<<<blocks, threads_per_block>>>(
            d_pairs,
            d_ranks,
            d_partitions,
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            curr_num_nodes,
            d_inbound_count_events_partition,
            d_inbound_count_events_index,
            d_inbound_count_events_hedge,
            d_inbound_count_events_delta
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // sort events by (partition, hedge, rank) [in lexicographical order for the tuple] and carry events_delta along
        // the resulting array will have events sorted by partition, and inside each partition sorted by hedge, and inside each hedge sorted by rank!
        auto count_events_sort_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_inbound_count_events_partition, t_inbound_count_events_hedge, t_inbound_count_events_index));
        auto count_events_sort_key_end = count_events_sort_key_begin + num_inbound_count_events;
        thrust::sort_by_key(count_events_sort_key_begin, count_events_sort_key_end, t_inbound_count_events_delta);
        // inclusive scan by key of the deltas, the key being (partition, hedge) -> we now have the total number of times each hedge appears in the inbound set as of each move (in order of rank)
        auto count_events_scan_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_inbound_count_events_partition, t_inbound_count_events_hedge));
        auto count_events_scan_key_end = count_events_scan_key_begin + num_inbound_count_events;
        thrust::inclusive_scan_by_key(count_events_scan_key_begin, count_events_scan_key_end, t_inbound_count_events_delta, t_inbound_count_events_delta);
        // new array of events, one event for each time the counter of an hedge in the inbound set (+ the overall inbounds per partition counter) goes from 0 to >0,
        // the event carrying a +1 to the inbound set size, one event for each time the counter of an hedge goes from >0 to 0 carrying a -1 to the inbound set size for that partition
        uint32_t *d_inbound_size_events_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_offsets, (num_inbound_count_events + 1) * sizeof(uint32_t))); // inbound_size_events_offsets[event idx] -> initially a flag of whether each event will produce an increase/decrese in inbound counts, after the scan it becomes the offset of each new event
        CUDA_CHECK(cudaMemset(d_inbound_size_events_offsets, 0u, (num_inbound_count_events + 1) * sizeof(uint32_t)));
        // launch configuration - count inbound events kernel
        threads_per_block = 128;
        num_threads_needed = num_inbound_count_events; // 1 thread per hedge event
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - count inbound events kernel
        std::cout << "Running count inbound events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        count_inbound_size_events_kernel<<<blocks, threads_per_block>>>(
            d_pins_per_partitions,
            d_inbound_count_events_partition,
            d_inbound_count_events_index,
            d_inbound_count_events_hedge,
            d_inbound_count_events_delta,
            num_inbound_count_events,
            num_partitions,
            d_inbound_size_events_offsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // transform the counts in offsets with a scan and find the total count of new size events
        thrust::device_ptr<uint32_t> t_inbound_size_events_offsets(d_inbound_size_events_offsets);
        thrust::inclusive_scan(t_inbound_size_events_offsets, t_inbound_size_events_offsets + num_inbound_count_events + 1, t_inbound_size_events_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
        uint32_t num_inbound_size_events = 0; // last value in the inclusive scan = full reduce = total number of touching hedges among all sets
        CUDA_CHECK(cudaMemcpy(&num_inbound_size_events, d_inbound_size_events_offsets + num_inbound_count_events, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t *d_inbound_size_events_partition = nullptr, *d_inbound_size_events_index = nullptr;
        int32_t *d_inbound_size_events_delta = nullptr;
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_partition, num_inbound_size_events * sizeof(uint32_t))); // inbound_size_events_partition[ev] -> partition affected by the event
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_index, num_inbound_size_events * sizeof(uint32_t))); // inbound_size_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_delta, num_inbound_size_events * sizeof(int32_t))); // inbound_size_events_delta[ev] -> inbound set size variation brought by the event
        thrust::device_ptr<uint32_t> t_inbound_size_events_partition(d_inbound_size_events_partition);
        thrust::device_ptr<uint32_t> t_inbound_size_events_index(d_inbound_size_events_index);
        thrust::device_ptr<int32_t> t_inbound_size_events_delta(d_inbound_size_events_delta);
        // launch configuration - build inbound events kernel
        threads_per_block = 128;
        num_threads_needed = num_inbound_count_events; // 1 thread per hedge event
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - build inbound events kernel
        std::cout << "Running build inbound events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        build_inbound_size_events_kernel<<<blocks, threads_per_block>>>(
            d_pins_per_partitions,
            d_inbound_count_events_partition,
            d_inbound_count_events_index,
            d_inbound_count_events_hedge,
            d_inbound_count_events_delta,
            d_inbound_size_events_offsets,
            num_inbound_count_events,
            num_partitions,
            d_inbound_size_events_partition,
            d_inbound_size_events_index,
            d_inbound_size_events_delta
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_inbound_count_events_partition));
        CUDA_CHECK(cudaFree(d_inbound_count_events_index));
        CUDA_CHECK(cudaFree(d_inbound_count_events_hedge));
        CUDA_CHECK(cudaFree(d_inbound_count_events_delta));
        CUDA_CHECK(cudaFree(d_inbound_size_events_offsets));
        // sort events by (partition, rank) [in lexicographical order for the tuple] and carry inbound_size_events_delta along
        auto inbound_size_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_inbound_size_events_partition, t_inbound_size_events_index));
        auto inbound_size_events_key_end = inbound_size_events_key_begin + num_inbound_size_events;
        thrust::sort_by_key(inbound_size_events_key_begin, inbound_size_events_key_end, t_inbound_size_events_delta);
        // inclusive scan inside each key (= partition) on the event deltas => for each event we get the cumulative size delta for that partition's inbound set at that point in the sequence
        thrust::inclusive_scan_by_key(t_inbound_size_events_partition, t_inbound_size_events_partition + num_inbound_size_events, t_inbound_size_events_delta, t_inbound_size_events_delta);
        // now mark moves that would violate the inbound set size constraint if the sequence were to end on them
        uint32_t *d_inbound_valid_moves = nullptr;
        CUDA_CHECK(cudaMalloc(&d_inbound_valid_moves, curr_num_nodes * sizeof(uint32_t))); // valid_move[rank idx] -> 1 if applying all moves up to the idx one in the ordered sequence gives a valid state
        CUDA_CHECK(cudaMemset(d_inbound_valid_moves, 0u, curr_num_nodes * sizeof(uint32_t)));
        thrust::device_ptr<uint32_t> t_inbound_valid_moves(d_inbound_valid_moves);
        // launch configuration - flag inbound events kernel
        threads_per_block = 128;
        num_threads_needed = num_inbound_size_events; // 1 thread per event
        blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - flag inbound events kernel
        std::cout << "Running flag inbound events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        flag_inbound_events_kernel<<<blocks, threads_per_block>>>(
            d_inbound_size_events_partition,
            d_inbound_size_events_index,
            d_inbound_size_events_delta,
            d_partitions_inbound_sizes,
            num_inbound_size_events,
            d_inbound_valid_moves
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_inbound_size_events_partition));
        CUDA_CHECK(cudaFree(d_inbound_size_events_index));
        CUDA_CHECK(cudaFree(d_inbound_size_events_delta));
        // compute, as of each event, the cumulative number of partitions that are invalid by summing the count of those made/unmade invalid at each event
        thrust::inclusive_scan(t_inbound_valid_moves, t_inbound_valid_moves + curr_num_nodes, t_inbound_valid_moves);
        // ======================================
        // find the move in the sequence that yields both the highest gain and a valid state (when all moves before it are applied)
        // index space 0..curr_num_nodes - 1
        auto idx_begin = thrust::make_counting_iterator<uint32_t>(0);
        // functor masking invalid endpoints in the sequence => invalid moves get a -inf score
        // NOTE: valid_moves => the move is valid when the counter is 0!
        masked_value_functor masked_scores { thrust::raw_pointer_cast(t_scores), thrust::raw_pointer_cast(t_valid_moves), thrust::raw_pointer_cast(t_inbound_valid_moves) };
        auto masked_begin = thrust::make_transform_iterator(idx_begin, masked_scores);
        auto masked_end = masked_begin + curr_num_nodes;
        // max over valid endpoints only, find the point in the sequence of moves where applying them further never nets a higher gain in a valid state
        auto best_iterator_entry = thrust::max_element(masked_begin, masked_end);
        const uint32_t best_rank = static_cast<uint32_t>(best_iterator_entry - masked_begin);
        const uint32_t num_good_moves = best_rank + 1; // "+1" to make this the improving moves count, rather than the last improving move's idx
        // validity double-check (if there were no valid moves...)
        uint32_t size_validity, inbounds_validity;
        CUDA_CHECK(cudaMemcpy(&size_validity, d_valid_moves + best_rank, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&inbounds_validity, d_inbound_valid_moves + best_rank, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (size_validity == 0 && inbounds_validity == 0) {
            // launch configuration - fm-ref apply kernel
            threads_per_block = 128;
            num_threads_needed = curr_num_nodes; // 1 thread per move to apply
            blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - fm-ref apply kernel
            std::cout << "Running fm-ref apply (" << num_good_moves << " good moves) kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            fm_refinement_apply_kernel<<<blocks, threads_per_block>>>(
                d_touching,
                d_touching_offsets,
                d_pairs,
                d_ranks,
                d_nodes_sizes,
                num_hedges,
                curr_num_nodes,
                num_partitions,
                num_good_moves,
                d_partitions,
                d_partitions_sizes
                //d_pins_per_partitions
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            std::cout << "WARNING: no valid refinement move found on level " << level_idx << " !!\n";
        }
        CUDA_CHECK(cudaFree(d_ranks));
        CUDA_CHECK(cudaFree(d_valid_moves));
        CUDA_CHECK(cudaFree(d_inbound_valid_moves));
        
        return std::make_tuple(num_partitions, d_partitions);
    };


    // START: the multi-level recursive refinement routine, down we go!
    auto [num_partitions, d_partitions] = coarsen_refine_uncoarsen(
        0, // first level
        num_nodes,
        d_hedges,
        d_hedges_offsets,
        hg.hedgesFlat().size(),
        d_touching,
        d_touching_offsets,
        touching_hedges_size,
        d_inbound_count,
        d_nodes_sizes
    );

    // final partitions rework: merge small ones and make partition ids zero-based
    thrust::device_ptr<uint32_t> t_partitions(d_partitions);
    thrust::device_ptr<uint32_t> t_partitions_sizes(d_partitions_sizes);
    thrust::device_ptr<uint32_t> t_partitions_inbound_sizes(d_partitions_inbound_sizes);

    // greedily merge small partitions
    // => checking inbound constraints w/out deduping, with a straight sum, to make it fast
    thrust::device_vector<uint32_t> t_part_index(num_partitions);
    thrust::sequence(t_part_index.begin(), t_part_index.end());
    // extract small partitions (size < K)
    thrust::device_vector<uint32_t> t_small_parts(num_partitions);
    auto small_end = thrust::copy_if(t_part_index.begin(), t_part_index.end(), t_small_parts.begin(), [=] __host__ __device__ (uint32_t p) { return t_partitions_sizes[p] < SMALL_PART_MERGE_SIZE_THRESHOLD; });
    t_small_parts.resize(small_end - t_small_parts.begin());
    uint32_t smallest_part_size = thrust::reduce(t_partitions_sizes, t_partitions_sizes + num_partitions, UINT32_MAX, thrust::minimum<uint32_t>());
    std::cout << "Smallest partition size: " << smallest_part_size << "\n";
    if (!t_small_parts.empty()) {
        std::cout << "Partitions compression over " << t_small_parts.size() << " partitions ...\n";
        // stable sort small partitions with key (size, inbound, id)
        thrust::stable_sort(t_small_parts.begin(), t_small_parts.end(), [=] __host__ __device__ (uint32_t a, uint32_t b) { uint32_t sa = t_partitions_sizes[a]; uint32_t sb = t_partitions_sizes[b]; if (sa != sb) return sa < sb; uint32_t ia = t_partitions_inbound_sizes[a]; uint32_t ib = t_partitions_inbound_sizes[b]; if (ia != ib) return ia < ib; return a < b; });
        // greedy grouping scan for constraints
        thrust::device_vector<constraints_state> t_constraints_states(t_small_parts.size());
        thrust::transform(t_small_parts.begin(), t_small_parts.end(), t_constraints_states.begin(), [=] __host__ __device__ (uint32_t p) { return constraints_state{ t_partitions_sizes[p], t_partitions_inbound_sizes[p], 0u }; });
        thrust::inclusive_scan(t_constraints_states.begin(), t_constraints_states.end(), t_constraints_states.begin(), [=] __host__ __device__ (const constraints_state& a, const constraints_state& b) { if (a.s + b.s <= h_max_nodes_per_part && a.i + b.i <= h_max_inbound_per_part) return constraints_state{ a.s + b.s, a.i + b.i, a.g }; return constraints_state{ b.s, b.i, a.g + 1 }; });
        // get the id of each node of a group
        thrust::device_vector<uint32_t> t_groups(t_constraints_states.size());
        thrust::transform(t_constraints_states.begin(), t_constraints_states.end(), t_groups.begin(), [] __host__ __device__ (const constraints_state& s) { return s.g; });
        // map groups to a representative partition id (lowest id in the group); groups are already contiguous, a single reduce-by-key is enough
        thrust::device_vector<uint32_t> t_rep_ids(t_groups.size());
        auto rep_end = thrust::reduce_by_key(t_groups.begin(), t_groups.end(), t_small_parts.begin(), thrust::make_discard_iterator(), t_rep_ids.begin(), thrust::equal_to<uint32_t>(), thrust::minimum<uint32_t>());
        t_rep_ids.resize(rep_end.second - t_rep_ids.begin());
        // build the map from partition id to the representative node
        thrust::device_vector<uint32_t> pid_map(num_partitions);
        thrust::sequence(pid_map.begin(), pid_map.end());
        thrust::device_vector<uint32_t> new_pids(t_small_parts.size());
        thrust::gather(t_groups.begin(), t_groups.end(), t_rep_ids.begin(), new_pids.begin());
        thrust::scatter(new_pids.begin(), new_pids.end(), t_small_parts.begin(), pid_map.begin());
        // update partitions
        uint32_t* pid_map_ptr = thrust::raw_pointer_cast(pid_map.data());
        thrust::transform(t_partitions, t_partitions + num_nodes, t_partitions, [pid_map_ptr] __host__ __device__ (uint32_t p) { return pid_map_ptr[p]; });
    } else
        std::cout << "Partitions compression not performed ...\n";

    // make d_partitions zero-based again, if we emptied some partitions... (same logic as that used for d_groups)
    thrust::device_vector<uint32_t> t_indices(num_nodes);
    thrust::sequence(t_indices.begin(), t_indices.end());
    thrust::sort_by_key(t_partitions, t_partitions + num_nodes, t_indices.begin());
    thrust::device_vector<uint32_t> t_headflags(num_nodes);
    t_headflags[0] = 0;
    thrust::transform(t_partitions + 1, t_partitions + num_nodes, t_partitions, t_headflags.begin() + 1, [] __device__ (uint32_t curr, uint32_t prev) { return curr != prev ? 1u : 0u; });
    thrust::inclusive_scan(t_headflags.begin(), t_headflags.end(), t_headflags.begin());
    const uint32_t new_num_partitions = t_headflags.back() + 1;
    thrust::scatter(t_headflags.begin(), t_headflags.end(), t_indices.begin(), t_partitions);

    // copy back results
    std::vector<uint32_t> partitions(num_nodes);
    CUDA_CHECK(cudaMemcpy(partitions.data(), d_partitions, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // =============================
    // print some example outputs
    #if VERBOSE
    std::set<uint32_t> part_count;
    std::cout << "Final partitioning results:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        uint32_t part = partitions[i];
        part_count.insert(part);
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            if (part == UINT32_MAX) std::cout << "node " << i << " -> part=none\n";
            else std::cout << "node " << i << " ->" << " part=" << part << "\n";
        }
    }
    std::cout << "Partitions count: " << part_count.size() << " (plus " << num_partitions - part_count.size() << " empty ones)" << "\n";
    if (new_num_partitions != part_count.size())
        std::cout << "WARNING, distinct partitions count (" << part_count.size() << ") does not match the computed number of partitions when zero-ing their ids (" << new_num_partitions << ") !!\n";
    std::set<uint32_t>().swap(part_count);
    #endif
    // =============================

    // cleanup device memory
    CUDA_CHECK(cudaFree(d_hedges));
    CUDA_CHECK(cudaFree(d_hedges_offsets));
    //CUDA_CHECK(cudaFree(d_neighbors)); // should have already been freed at the innermost recursion level
    //CUDA_CHECK(cudaFree(d_neighbors_offsets));
    CUDA_CHECK(cudaFree(d_touching));
    CUDA_CHECK(cudaFree(d_touching_offsets));
    CUDA_CHECK(cudaFree(d_inbound_count));
    CUDA_CHECK(cudaFree(d_hedge_weights));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_f_scores));
    CUDA_CHECK(cudaFree(d_u_scores));
    CUDA_CHECK(cudaFree(d_slots));
    CUDA_CHECK(cudaFree(d_nodes_sizes));
    CUDA_CHECK(cudaFree(d_partitions));
    CUDA_CHECK(cudaFree(d_partitions_sizes));
    CUDA_CHECK(cudaFree(d_pins_per_partitions));
    CUDA_CHECK(cudaFree(d_partitions_inbound_sizes));

    // final sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaStreamDestroy(transfer_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));

    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << "Stopping timer...\n";

    // === CUDA STUFF ENDS HERE ===
    // ============================

    std::cerr << "CUDA section: complete; proceeding with partitioning results validation and evalution...\n";

    double total_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
    std::cout << "Total execution time: " << std::fixed << std::setprecision(3) << total_ms << " ms\n";

    if (hw.checkPartitionValidity(hg, partitions, true)) {
        auto partitioned_hg = hg.getPartitionsHypergraph(partitions, false, true);
        //auto metrics = hw.getAllMetrics(partitioned_hg, placement);
        float partitioning_cost = partitioned_hg.totalWeight();
        auto synaptic_reuse = hw.synapticReuse(hg, partitions);
        std::cout << "Partitioned hypergraph:\n";
        std::cout << "  Nodes:       " << partitioned_hg.nodes() << "\n";
        std::cout << "  Hyperedges:  " << partitioned_hg.hedges().size() << "\n";
        std::cout << "  Total pins:  " << partitioned_hg.hedgesFlat().size() << "\n";
        std::cout << "  Total Spike Frequency: " << partitioned_hg.totalSpikeFrequency() << "\n";
        std::cout << "  Synaptic reuse:        " << std::fixed << std::setprecision(3) << synaptic_reuse.ar_mean << " ar. mean, " << synaptic_reuse.geo_mean << " geo. mean\n";
        
        // save hypergraph
        if (!save_path.empty()) {
            if (!loaded) {
                std::cerr << "Error: -s used without loading a hypergraph first.\n";
                return 1;
            }
            try {
                // TODO: apply the partitioning before saving!
                partitioned_hg.save(save_path);
                std::cout << "Partitioned hypergraph saved to " << save_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error saving file: " << e.what() << "\n";
                return 1;
            }
        }
    } else {
        std::cerr << "WARNING, invalid partitining !!\n";
    }

    return 0;
}
