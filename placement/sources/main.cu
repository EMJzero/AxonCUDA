#include <tuple>
#include <string>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <algorithm>
#include <filesystem>

#include <cuda_runtime.h>

#include "thruster.cuh"

#include <cub/cub.cuh>

#include "hgraph.hpp"
#include "nmhardware.hpp"
#include "runconfig_plc.hpp"

#include "utils.cuh"
#include "utils_plc.cuh"
#include "data_types.cuh"
#include "data_types_plc.cuh"
#include "defines_plc.cuh"
#include "placement.cuh"
#include "ordering.cuh"
#include "prep.cuh"

using namespace hgraph;
using namespace hwmodel;
using namespace hwgeom;
using namespace config_plc;


int main(int argc, char** argv) {
    if (argc == 1) {
        printHelp();
        return 0;
    }

    // parse CLI args
    runconfig cfg = parseArgs(argc, argv);

    // load hypergraph
    HyperGraph hg = loadHgraph(cfg);

    // setup the hardware model
    HardwareModel hw = setupNMH(cfg);

    // print statistics
    std::cout << "Loaded hypergraph:\n";
    std::cout << "  Nodes:      " << hg.nodes() << "\n";
    std::cout << "  Hyperedges: " << hg.hedges().size() << "\n";
    std::cout << "  Total pins: " << hg.hedgesFlat().size() << "\n";
    std::cout << "  Total connections weight: " << std::fixed << std::setprecision(3) << hg.connectivity() << "\n";

    // print hardware details
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

    // build incidence sets on the host only if explicitly required
    if (!cfg.device_touching_construction)
        hg.buildIncidenceSets();

    uint32_t num_hedges = static_cast<uint32_t>(hg.hedges().size());
    std::vector<dim_t> hedges_offsets; // hedge idx -> hedge start index in the contiguous hedges array
    std::vector<uint32_t> srcs_count;
    hedges_offsets.reserve(num_hedges + 1);
    //srcs_count.reserve(num_hedges);

    // prepare hedge offsets
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedges_offsets.push_back(static_cast<dim_t>(hg.hedges()[i].offset()));
        //srcs_count.push_back(hg.hedges()[i].src_count());
    }
    hedges_offsets.push_back(hg.hedgesFlat().size());

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

    std::cout << "Starting timer...\n";
    auto time_start = std::chrono::high_resolution_clock::now();
    cudaEvent_t d_time_start, d_time_stop;
    CUDA_CHECK(cudaEventCreate(&d_time_start));
    CUDA_CHECK(cudaEventCreate(&d_time_stop));
    CUDA_CHECK(cudaEventRecord(d_time_start));

    // ============================
    // === CUDA STUFF GOES HERE ===

    std::cout << "Setting up GPU memory...\n";

    // device pointers
    // hypergraph
    uint32_t *d_hedges = nullptr; // contigous hedges array (each hedge must be stored as src+destinations, with the src in the first position)
    dim_t *d_hedges_offsets = nullptr; // hedges_offsets[hedge idx] -> hedge start idx in d_hedges
    //uint32_t *d_srcs_count = nullptr; // srcs_count[hedge idx] -> number of sources of hedge idx
    uint32_t *d_touching = nullptr; // contigous inbound+outbout sets array (first inbound, then outbound)
    dim_t *d_touching_offsets = nullptr; // touching_offsets[node idx] -> touching set start idx in d_touching
    //uint32_t *d_inbound_count = nullptr; // inbound_count[node idx] -> how many hedge of touching[node idx] are inbound (inbound hedges are before inbound_count[node idx], then outbound)
    float *d_hedge_weights = nullptr; // hedge_weights[hedge idx] -> weight
    // placement
    coords *d_placement = nullptr; // placement[node idx] -> x and y placement coordinates of node
    uint32_t *d_inv_placement = nullptr; // inv_placement[y * h_max_width + x] -> idx of the node occupying such place, or UINT32_MAX

    // allocate device memory
    CUDA_CHECK(cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hedges_offsets, (num_hedges + 1) * sizeof(dim_t)));
    //CUDA_CHECK(cudaMalloc(&d_srcs_count, num_hedges * sizeof(uint32_t)));
    //CUDA_CHECK(cudaMalloc(&d_touching, touching_hedges.size() * sizeof(uint32_t)));
    //CUDA_CHECK(cudaMalloc(&d_touching_offsets, (num_nodes + 1) * sizeof(dim_t)));
    //CUDA_CHECK(cudaMalloc(&d_inbound_count, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hedge_weights, num_hedges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_placement, num_nodes * sizeof(coords)));
    CUDA_CHECK(cudaMalloc(&d_inv_placement, h_max_width * h_max_height * sizeof(uint32_t)));

    // copy to device
    CUDA_CHECK(cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedges_offsets, hedges_offsets.data(), (num_hedges + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
    //CUDA_CHECK(cudaMemcpy(d_srcs_count, srcs_count.data(), num_hedges * sizeof(uint32_t), cudaMemcpyHostToDevice));
    //CUDA_CHECK(cudaMemcpy(d_touching, touching_hedges.data(), touching_hedges.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    //CUDA_CHECK(cudaMemcpy(d_touching_offsets, touching_hedges_offsets.data(), (num_nodes + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedge_weights, hedge_weights.data(), num_hedges * sizeof(float), cudaMemcpyHostToDevice));

    // thrust pointers
    //thrust::device_ptr<dim_t> t_touching_offsets(d_touching_offsets);
    //thrust::device_ptr<uint32_t> t_inbound_count(d_inbound_count);

    // initialize
    // each initial node has one outbound hyperedge -> init. inbound counts to the number of touching - 1
    //thrust::transform(t_touching_offsets + 1, t_touching_offsets + 1 + num_nodes, t_touching_offsets, t_inbound_count, [] __device__ (int next, int curr) { return next - curr - 1; });

    // copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(max_width, &h_max_width, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(max_height, &h_max_height, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    // wrap up memory duties with a sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // prepare touching sets
    if (cfg.device_touching_construction) {
        std::tie(d_touching, d_touching_offsets) = buildTouching(
            cfg,
            d_hedges,
            d_hedges_offsets,
            num_nodes,
            num_hedges
        );
    } else {
        std::tie(d_touching, d_touching_offsets) = buildTouchingHost(
            hg
        );
    }

    std::cout << "Starting core timer...\n";
    cudaEvent_t d_time_core_start, d_time_core_stop;
    CUDA_CHECK(cudaEventCreate(&d_time_core_start));
    CUDA_CHECK(cudaEventCreate(&d_time_core_stop));
    CUDA_CHECK(cudaEventRecord(d_time_core_start));

    // initial placement
    uint32_t* d_order_idx = nullptr;  // order_idx[node] -> position in the 1D ordering for node
    if (cfg.feedforward_order) {
        CUDA_CHECK(cudaMalloc(&d_order_idx, num_nodes * sizeof(uint32_t)));
        std::cout << "Ordering nodes (sequential - might take a while) ...\n";
        std::vector<uint32_t> nodes_order_idx = hg.feedForwardOrder();
        CUDA_CHECK(cudaMemcpy(d_order_idx, nodes_order_idx.data(), num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice));
    } else {
        std::cout << "Ordering nodes (parallel - recursive bisection) ...\n";
        d_order_idx = locality_ordering(
            num_nodes,
            num_hedges,
            d_hedges,
            d_hedges_offsets,
            d_hedge_weights,
            d_touching,
            d_touching_offsets,
            cfg.seed
        );
    }

    // generate a 1D to 2D map for lattice points
    std::vector<coords> init_placement = hilbertPlacement(num_nodes, h_max_width, h_max_height);
    CUDA_CHECK(cudaMemcpy(d_placement, init_placement.data(), num_nodes * sizeof(coords), cudaMemcpyHostToDevice));

    // assign (scatter) to the nodes their respective placement following both 1D orders (from ordered nodes through the lattice points map)
    coords *d_tmp_placement = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp_placement, num_nodes * sizeof(coords)));
    thrust::device_ptr<coords> t_placement(d_placement);
    thrust::device_ptr<coords> t_tmp_placement(d_tmp_placement);
    thrust::device_ptr<uint32_t> t_order_idx(d_order_idx);
    thrust::scatter(t_placement, t_placement + num_nodes, t_order_idx, t_tmp_placement);
    CUDA_CHECK(cudaFree(d_order_idx));
    CUDA_CHECK(cudaFree(d_placement));
    d_placement = d_tmp_placement;
    
    // =============================
    // print some temporary results
    #if VERBOSE
    CUDA_CHECK(cudaMemcpy(init_placement.data(), d_placement, num_nodes * sizeof(coords), cudaMemcpyDeviceToHost));
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
    {
        // launch configuration - inverse placement kernel
        int threads_per_block = 128;
        int num_threads_needed = num_nodes; // 1 thread per node
        int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - inverse placement kernel
        std::cout << "Running inverse placement kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        inverse_placement_kernel<<<blocks, threads_per_block>>>(
            d_placement,
            num_nodes,
            d_inv_placement
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

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

    // run force-directed refinement
    force_directed_refinement(
        props,
        d_hedges,
        d_hedges_offsets,
        d_touching,
        d_touching_offsets,
        d_hedge_weights,
        num_nodes,
        d_placement,
        d_inv_placement
    );

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
    //CUDA_CHECK(cudaFree(d_srcs_count));
    CUDA_CHECK(cudaFree(d_touching));
    CUDA_CHECK(cudaFree(d_touching_offsets));
    //CUDA_CHECK(cudaFree(d_inbound_count));
    CUDA_CHECK(cudaFree(d_hedge_weights));
    CUDA_CHECK(cudaFree(d_placement));
    CUDA_CHECK(cudaFree(d_inv_placement));

    // final sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(d_time_stop));
    CUDA_CHECK(cudaEventSynchronize(d_time_stop));
    float d_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d_total_ms, d_time_start, d_time_stop));
    CUDA_CHECK(cudaEventDestroy(d_time_start));
    CUDA_CHECK(cudaEventDestroy(d_time_stop));

    CUDA_CHECK(cudaEventRecord(d_time_core_stop));
    CUDA_CHECK(cudaEventSynchronize(d_time_core_stop));
    float d_core_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d_core_ms, d_time_core_start, d_time_core_stop));
    CUDA_CHECK(cudaEventDestroy(d_time_core_start));
    CUDA_CHECK(cudaEventDestroy(d_time_core_stop));

    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << "Stopping timer...\n";

    // === CUDA STUFF ENDS HERE ===
    // ============================

    std::cerr << "CUDA section: complete; proceeding with placement results validation and evalution...\n";

    double total_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
    // core: excluding the initialization of hedges and incidence sets in device memory
    std::cout << "Total device core execution time: " << std::fixed << std::setprecision(3) << d_core_ms << " ms\n";
    std::cout << "Total device execution time: " << std::fixed << std::setprecision(3) << d_total_ms << " ms\n";
    std::cout << "Total host execution time: " << std::fixed << std::setprecision(3) << total_ms << " ms\n";

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
        saveResult(cfg, h_placement);
    } else {
        std::cerr << "WARNING, invalid placement !!\n";
    }

    return 0;
}
