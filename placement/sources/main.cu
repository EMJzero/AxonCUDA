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

#include <omp.h>

#include "hgraph.hpp"
#include "curves.hpp"
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

    std::cout << "Using settings:\n";
    std::cout << "  Seed:                            " << cfg.seed << "\n";
    std::cout << "  Force-directed iterations:       " << cfg.fd_iterations << "\n";
    std::cout << "  Force-directed candidates count: " << cfg.candidates_count << "\n";
    std::cout << "  Multi-start attempts:            ";
    if (cfg.multi_start_override == UINT32_MAX) std::cout << "<max-occupancy>\n";
    else std::cout << cfg.multi_start_override << "\n";
    std::cout << "  Host threads count:              ";
    if (cfg.num_host_threads == UINT32_MAX) std::cout << "<max-occupancy>\n";
    else std::cout << cfg.num_host_threads << "\n";
    std::cout << "  Label propagation repeats:       " << cfg.labelprop_repeats << "\n";
    std::cout << "  Space-filling curve:             " << SFCtoString(cfg.space_filling_curve) << "\n";
    std::cout << "  Flags: " << (cfg.device_touching_construction ? "dtc " : "") << (cfg.feedforward_order ? "ff " : "") << "\n";

    if (hg.nodes() > hw.coresAlongX() * hw.coresAlongY()) {
        ERR(cfg) std::cerr << "ERROR, the hypergraph has more nodes (" << hg.nodes() << ") than the 2D lattice has points (" << hw.coresAlongX() * hw.coresAlongY() << "), placement would fail !!\n";
        return 1;
    }
    
    std::cout << "CUDA device:\n";
    
    // get device properties
    int device_cnt;
    cudaGetDeviceCount(&device_cnt);
    std::cout << "  Found " << device_cnt << " devices: using device " << DEVICE_ID << "\n";
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, DEVICE_ID);
    std::cout << "  Device name:            " << props.name << "\n";
    std::cout << "  GPU clock rate:         " << props.clockRate / 1e3 << " MHz\n";
    std::cout << "  Available VRAM:         " << std::fixed << std::setprecision(1) << (float)(props.totalGlobalMem) / (1 << 30) << " GB\n";
    const float peak_bandwidth = 2.0f * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6;
    std::cout << "  Peak VRAM bandwidth:    " << peak_bandwidth << " GB/s\n";
    std::cout << "  SM count:               " << props.multiProcessorCount << "\n";
    std::cout << "  Max. threads / SM:      " << props.maxThreadsPerMultiProcessor << "\n";
    std::cout << "  Max. threads / block:   " << props.maxThreadsPerBlock << "\n";
    std::cout << "  MAx. registers / block: " << props.regsPerBlock << "\n";
    std::cout << "  Max. grid size:         " << props.maxGridSize[0] << " x " << props.maxGridSize[1] << " x " << props.maxGridSize[2] << "\n";
    std::cout << "  Max. block size:        " << props.maxThreadsDim[0] << " x " << props.maxThreadsDim[1] << " x " << props.maxThreadsDim[2] << "\n";
    std::cout << "  Shared mem. per block:  " << std::fixed << std::setprecision(1) << (float)(props.sharedMemPerBlock) / (1 << 10) << " KB\n";
    
    INFO(cfg) std::cout << "Preparing hypergraph data...\n";

    // build incidence sets on the host only if explicitly required
    if (!cfg.device_touching_construction || cfg.feedforward_order)
        hg.buildIncidenceSets();

    uint32_t num_hedges = static_cast<uint32_t>(hg.hedges().size());
    std::vector<dim_t> hedges_offsets; // hedge idx -> hedge start index in the contiguous hedges array
    std::vector<uint32_t> srcs_count;
    hedges_offsets.reserve(num_hedges + 1);
    srcs_count.reserve(num_hedges);

    // prepare hedge offsets
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedges_offsets.push_back(static_cast<dim_t>(hg.hedges()[i].offset()));
        srcs_count.push_back(hg.hedges()[i].src_count());
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

    INFO(cfg) std::cout << "Starting timer...\n";
    auto time_start = std::chrono::high_resolution_clock::now();
    cudaEvent_t d_time_start, d_time_stop;
    CUDA_CHECK(cudaEventCreate(&d_time_start));
    CUDA_CHECK(cudaEventCreate(&d_time_stop));
    CUDA_CHECK(cudaEventRecord(d_time_start));

    // ============================
    // === CUDA STUFF GOES HERE ===

    INFO(cfg) std::cout << "Setting up GPU memory...\n";

    // device pointers
    // hypergraph
    uint32_t *d_hedges = nullptr; // contigous hedges array (each hedge must be stored as src+destinations, with the src in the first position)
    dim_t *d_hedges_offsets = nullptr; // hedges_offsets[hedge idx] -> hedge start idx in d_hedges
    uint32_t *d_srcs_count = nullptr; // srcs_count[hedge idx] -> number of sources of hedge idx
    uint32_t *d_touching = nullptr; // contigous inbound+outbout sets array (first inbound, then outbound)
    dim_t *d_touching_offsets = nullptr; // touching_offsets[node idx] -> touching set start idx in d_touching
    //uint32_t *d_inbound_count = nullptr; // inbound_count[node idx] -> how many hedge of touching[node idx] are inbound (inbound hedges are before inbound_count[node idx], then outbound)
    float *d_hedge_weights = nullptr; // hedge_weights[hedge idx] -> weight
    
    // best placement
    coords *d_best_placement = nullptr; // placement[node idx] -> x and y placement coordinates of node
    uint32_t *d_best_inv_placement = nullptr; // inv_placement[y * h_max_width + x] -> idx of the node occupying such place, or UINT32_MAX
    // |
    // best results
    float best_whops = FLT_MAX; // total weight of hedge hops in the current best solution
    // |
    // 1D to 2D map
    coords *d_1dto2d_placement = nullptr; // 1dto2d_placement[node idx] -> x and y coordinates for the node's sequence mapping from 1D to 2D
    
    // allocate device memory
    CUDA_CHECK(cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hedges_offsets, (num_hedges + 1) * sizeof(dim_t)));
    CUDA_CHECK(cudaMalloc(&d_srcs_count, num_hedges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hedge_weights, num_hedges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_placement, num_nodes * sizeof(coords)));
    CUDA_CHECK(cudaMalloc(&d_best_inv_placement, h_max_width * h_max_height * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_1dto2d_placement, num_nodes * sizeof(coords)));

    // copy to device
    CUDA_CHECK(cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedges_offsets, hedges_offsets.data(), (num_hedges + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_srcs_count, srcs_count.data(), num_hedges * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedge_weights, hedge_weights.data(), num_hedges * sizeof(float), cudaMemcpyHostToDevice));

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
            cfg,
            hg
        );
    }

    // generate a 1D to 2D map for lattice points
    std::vector<coords> h_1dto2d_placement;
    if (cfg.space_filling_curve == SpaceFillingCurve::HILB)
        h_1dto2d_placement = hilbertPlacement(num_nodes, h_max_width, h_max_height, cfg.verbose_info);
    else if (cfg.space_filling_curve == SpaceFillingCurve::SNAK)
        h_1dto2d_placement = snakePlacement(num_nodes, h_max_width, h_max_height, true, cfg.verbose_info);
    else if (cfg.space_filling_curve == SpaceFillingCurve::ZORD)
        h_1dto2d_placement = zorderPlacement(num_nodes, h_max_width, h_max_height, cfg.verbose_info);
    else if (cfg.space_filling_curve == SpaceFillingCurve::QUAD)
        h_1dto2d_placement = quadPlacement(num_nodes, h_max_width, h_max_height, cfg.verbose_info);
    CUDA_CHECK(cudaMemcpy(d_1dto2d_placement, h_1dto2d_placement.data(), num_nodes * sizeof(coords), cudaMemcpyHostToDevice));
    thrust::device_ptr<const coords> t_1dto2d_placement(d_1dto2d_placement);

    // determine multistart count to maximally occupy the GPU
    uint32_t multi_start_count = cfg.multi_start_override;
    if (cfg.feedforward_order && multi_start_count > 1) {
        multi_start_count = 1u;
        ERR(cfg) std::cout << "WARNING, feedforward ordering doesn't support multi-start, forcing multi-start count to 1 !!\n";
    } else if (multi_start_count == UINT32_MAX) {
        // HP: every kernel uses at most one warp per node
        //int max_warps_per_SM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        //int total_warp_capacity = prop.multiProcessorCount * max_warps_per_SM;
        // TODO: this is a good-enough guess for now, eventually:
        // - use "cudaOccupancyMaxActiveBlocksPerMultiprocessor" on every kernel at the start to get an exact "active_blocks_per_SM"!
        // - then compute "active_warps_per_SM = active_blocks_per_SM * (THREADS_PER_BLOCK / WARP_SIZE)"
        const uint32_t active_warps_per_SM = props.maxThreadsPerMultiProcessor / props.warpSize;
        const uint32_t total_warp_capacity = props.multiProcessorCount * active_warps_per_SM;
        multi_start_count = std::max(1u, total_warp_capacity / num_nodes);
        INFO(cfg) std::cout << "Setting multi-start count for maximum occupancy to: " << multi_start_count << " (" << active_warps_per_SM << " warps per SM * " << props.multiProcessorCount << " SMs / " << num_nodes << " nodes)\n";
    }
    uint32_t num_threads = cfg.num_host_threads;
    if (num_threads == UINT32_MAX) {
        num_threads = multi_start_count;
        INFO(cfg) std::cout << "Setting num-threads equal to multi-start count: " << multi_start_count << "\n";
    }
    if (multi_start_count > 2u * omp_get_max_threads()) {
        ERR(cfg) std::cout << "WARNING, it is suggested to decrease num-threads from " << num_threads << " to " << 2 * omp_get_max_threads() << ", as not to exceed 2x the number of available CPUs (" <<  omp_get_max_threads() << ") !!\n";
    }

    // TODO:
    // In the parallel multi-start section, put "#opm master" before LOG and LAUNCH, to let only the master thread output to CLI
    // => nah, rather, add "(tid = X)" in each log!

    // setup a stream per thread
    omp_set_num_threads(num_threads);
    INFO(cfg) std::cout << "Spawning " << num_threads << " OpenMP threads and matching CUDA streams ...\n";
    std::vector<cudaStream_t> streams(num_threads);
    for (uint32_t i = 0; i < num_threads; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    INFO(cfg) std::cout << "Starting core timer...\n";
    cudaEvent_t d_time_core_start, d_time_core_stop;
    CUDA_CHECK(cudaEventCreate(&d_time_core_start));
    CUDA_CHECK(cudaEventCreate(&d_time_core_stop));
    CUDA_CHECK(cudaEventRecord(d_time_core_start));

    float avg_attempt_time = 0.0f;

    #pragma omp parallel for default(shared) reduction(+:avg_attempt_time)
    for (uint32_t start = 0; start < multi_start_count; start++) {
        // recover your stream
        int tid = omp_get_thread_num();
        cudaStream_t stream = streams[tid];
        auto thrust_exec = thrust::cuda::par.on(stream);

        const uint32_t seed = cfg.seed + start;

        INFO(cfg) std::cout TID(tid) << "Beginning attempt " << start << " ...\n";

        // prepare thread-local solution
        coords *d_placement = nullptr; // placement[node idx] -> x and y placement coordinates of node
        uint32_t *d_inv_placement = nullptr; // inv_placement[y * h_max_width + x] -> idx of the node occupying such place, or UINT32_MAX
        CUDA_CHECK(cudaMallocAsync(&d_placement, num_nodes * sizeof(coords), stream));
        CUDA_CHECK(cudaMallocAsync(&d_inv_placement, h_max_width * h_max_height * sizeof(uint32_t), stream));

        INFO(cfg) std::cout TID(tid) << "Starting attempt timer...\n";
        cudaEvent_t d_time_attempt_start, d_time_attempt_stop;
        CUDA_CHECK(cudaEventCreate(&d_time_attempt_start));
        CUDA_CHECK(cudaEventCreate(&d_time_attempt_stop));
        CUDA_CHECK(cudaEventRecord(d_time_attempt_start, stream));

        // initial placement
        uint32_t* d_order_idx = nullptr; // order_idx[node] -> position in the 1D ordering for node
        if (cfg.feedforward_order) {
            CUDA_CHECK(cudaMallocAsync(&d_order_idx, num_nodes * sizeof(uint32_t), stream));
            INFO(cfg) std::cout TID(tid) << "Ordering nodes (sequential - might take a while) ...\n";
            std::vector<uint32_t> nodes_order_idx = hg.feedForwardOrder();
            CUDA_CHECK(cudaMemcpyAsync(d_order_idx, nodes_order_idx.data(), num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        } else {
            INFO(cfg) std::cout TID(tid) << "Ordering nodes (parallel - recursive bisection) ...\n";
            d_order_idx = locality_ordering(
                cfg,
                num_nodes,
                num_hedges,
                hg.hedgesFlat().size(),
                d_hedges,
                d_hedges_offsets,
                d_hedge_weights,
                d_touching,
                d_touching_offsets,
                seed,
                stream,
                tid
            );
        }

        // assign to each node its respective placement following both 1D orders (from ordered nodes to the lattice points map)
        thrust::device_ptr<coords> t_placement(d_placement);
        thrust::device_ptr<uint32_t> t_order_idx(d_order_idx);
        thrust::gather(thrust_exec, t_order_idx, t_order_idx + num_nodes, t_1dto2d_placement, t_placement);
        CUDA_CHECK(cudaFreeAsync(d_order_idx, stream));
        
        // =============================
        // print some temporary results
        LOG(cfg) {
            std::vector<coords> init_placement(num_nodes);
            CUDA_CHECK(cudaMemcpyAsync(init_placement.data(), d_placement, num_nodes * sizeof(coords), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<Coord2D> h_init_placement(num_nodes);
            for (uint32_t i = 0; i < num_nodes; i++) {
                h_init_placement[i] = Coord2D(
                    init_placement[i].x,
                    init_placement[i].y
                );
            }
            
            if (hw.checkPlacementValidity(hg, h_init_placement, true)) {
                auto metrics = hw.getAllMetrics(hg, h_init_placement);
                std::cout TID(tid) << "Initial placement metrics:\n";
                std::cout TID(tid) << "  Energy:        " << std::fixed << std::setprecision(3) << metrics.energy.value() << "\n";
                std::cout TID(tid) << "  Avg. latency:  " << std::fixed << std::setprecision(3) << metrics.avg_latency.value() << "\n";
                std::cout TID(tid) << "  Max. latency:  " << std::fixed << std::setprecision(3) << metrics.max_latency.value() << "\n";
                std::cout TID(tid) << "  Avg. congestion:  " << std::fixed << std::setprecision(3) << metrics.avg_congestion.value() << "\n";
                std::cout TID(tid) << "  Max. congestion:  " << std::fixed << std::setprecision(3) << metrics.max_congestion.value() << "\n";
                std::cout TID(tid) << "  Connections locality:\n";
                std::cout TID(tid) << "    Flat:     " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean << " ar. mean, " << metrics.connections_locality.value().geo_mean << " geo. mean\n";
                std::cout TID(tid) << "    Weighted: " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean_weighted << " ar. mean, " << metrics.connections_locality.value().geo_mean_weighted << " geo. mean\n";
            } else {
                std::cerr TID(tid) << "ERROR, invalid initial placement !!\n";
                abort(); // should never happen
            }
            std::vector<Coord2D>().swap(h_init_placement);

            std::cout TID(tid) << "Initial placement:\n";
            for (uint32_t i = 0; i < num_nodes; ++i) {
                if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
                    const coords place = init_placement[i];
                    std::cout TID(tid) << "  node " << i << " -> x=" << place.x << " y=" << place.y << "\n";
                }
            }
        }
        // =============================

        // initialize inverse placement
        CUDA_CHECK(cudaMemsetAsync(d_inv_placement, 0xFF, h_max_width * h_max_height * sizeof(uint32_t), stream));
        {
            // launch configuration - inverse placement kernel
            int threads_per_block = 128;
            int num_threads_needed = num_nodes; // 1 thread per node
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - inverse placement kernel
            LAUNCH(cfg) TID(tid) RUN << "inverse placement kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            inverse_placement_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_placement,
                num_nodes,
                d_inv_placement
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // =============================
        // print some temporary results
        LOG(cfg) {
            std::vector<uint32_t> inv_place_tmp(h_max_width * h_max_height);
            CUDA_CHECK(cudaMemcpyAsync(inv_place_tmp.data(), d_inv_placement, h_max_width * h_max_height * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::cout TID(tid) << "Initial inverse placement:\n";
            printMatrixHex16(inv_place_tmp.data(), h_max_width, h_max_height, VERBOSE_LENGTH, VERBOSE_LENGTH);
            std::vector<uint32_t>().swap(inv_place_tmp);
        }
        // =============================

        // run force-directed refinement
        forceDirectedRefinement(
            cfg,
            props,
            d_hedges,
            d_hedges_offsets,
            d_touching,
            d_touching_offsets,
            d_hedge_weights,
            num_nodes,
            d_placement,
            d_inv_placement,
            stream,
            tid
        );

        // grade present solution
        float src_dst_distance;
        float steiner_span;
        std::tie(src_dst_distance, steiner_span) = getLocalityMetrics(
            cfg,
            d_placement,
            d_hedges,
            d_hedges_offsets,
            d_srcs_count,
            d_hedge_weights,
            num_hedges,
            stream,
            tid
        );
        // TODO: tune these two coefficients
        float curr_whops = 0.4 * src_dst_distance + 0.6 * steiner_span; // lower is better

        // update current best solution
        #pragma omp critical
        {
            if (curr_whops < best_whops) {
                best_whops = curr_whops;
                CUDA_CHECK(cudaMemcpyAsync(d_best_placement, d_placement, num_nodes * sizeof(coords), cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_best_inv_placement, d_inv_placement, h_max_width * h_max_height * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                INFO(cfg) std::cout TID(tid) << "Updated best placement: whops=" << std::fixed << std::setprecision(3) << curr_whops << "\n";
            } else {
                INFO(cfg) std::cout TID(tid) << "Discarded placement: whops=" << std::fixed << std::setprecision(3) << curr_whops << " < " << best_whops << "\n";
            }
        }

        CUDA_CHECK(cudaFreeAsync(d_placement, stream));
        CUDA_CHECK(cudaFreeAsync(d_inv_placement, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaEventRecord(d_time_attempt_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(d_time_attempt_stop));
        float d_attempt_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d_attempt_ms, d_time_attempt_start, d_time_attempt_stop));
        CUDA_CHECK(cudaEventDestroy(d_time_attempt_start));
        CUDA_CHECK(cudaEventDestroy(d_time_attempt_stop));
        avg_attempt_time += d_attempt_ms;
    }

    // copy back results
    std::vector<coords> placement(num_nodes);
    CUDA_CHECK(cudaMemcpy(placement.data(), d_best_placement, num_nodes * sizeof(coords), cudaMemcpyDeviceToHost));

    if (best_whops == FLT_MAX)
        ERR(cfg) std::cerr << "WARNING, the final mapping still has whops=FLT_MAX, no mapping seems to have been constructed !!\n";

    // =============================
    // print some example outputs
    LOG(cfg) {
        std::cout << "Final placement:\n";
        for (uint32_t i = 0; i < num_nodes; ++i) {
            if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
                coords place = placement[i];
                std::cout << "  node " << i << " -> x=" << place.x << " y=" << place.y << "\n";
            }
        }
        std::vector<uint32_t> final_inv_place_tmp(h_max_width * h_max_height);
        CUDA_CHECK(cudaMemcpy(final_inv_place_tmp.data(), d_best_inv_placement, h_max_width * h_max_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::cout << "Final inverse placement:\n";
        printMatrixHex16(final_inv_place_tmp.data(), h_max_width, h_max_height, VERBOSE_LENGTH, VERBOSE_LENGTH);
    }
    // =============================
    
    // cleanup device memory
    CUDA_CHECK(cudaFree(d_hedges));
    CUDA_CHECK(cudaFree(d_hedges_offsets));
    CUDA_CHECK(cudaFree(d_srcs_count));
    CUDA_CHECK(cudaFree(d_touching));
    CUDA_CHECK(cudaFree(d_touching_offsets));
    //CUDA_CHECK(cudaFree(d_inbound_count));
    CUDA_CHECK(cudaFree(d_hedge_weights));
    CUDA_CHECK(cudaFree(d_best_placement));
    CUDA_CHECK(cudaFree(d_best_inv_placement));

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
    INFO(cfg) std::cout << "Stopping timer...\n";

    // === CUDA STUFF ENDS HERE ===
    // ============================

    INFO(cfg) std::cout << "CUDA section: complete; proceeding with placement results validation and evalution...\n";

    double total_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
    avg_attempt_time /= multi_start_count;
    // core: excluding the initialization of hedges and incidence sets in device memory
    std::cout << "Average device attempt execution time: " << std::fixed << std::setprecision(3) << avg_attempt_time << " ms\n";
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

    if (hw.checkPlacementValidity(hg, h_placement, cfg.verbose_errs_and_warns)) {
        auto metrics = hw.getAllMetrics(hg, h_placement);
        std::cout << "Placement metrics:\n";
        std::cout << "  Energy:        " << std::fixed << std::setprecision(3) << metrics.energy.value() << "\n";
        std::cout << "  Avg. latency:  " << std::fixed << std::setprecision(3) << metrics.avg_latency.value() << "\n";
        std::cout << "  Max. latency:  " << std::fixed << std::setprecision(3) << metrics.max_latency.value() << "\n";
        std::cout << "  Avg. congestion:  " << std::fixed << std::setprecision(3) << metrics.avg_congestion.value() << "\n";
        std::cout << "  Max. congestion:  " << std::fixed << std::setprecision(3) << metrics.max_congestion.value() << "\n";
        std::cout << "  Connections locality:\n";
        std::cout << "    Flat:     " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean << " ar. mean, " << metrics.connections_locality.value().geo_mean << " geo. mean\n";
        std::cout << "    Weighted: " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean_weighted << " ar. mean, " << metrics.connections_locality.value().geo_mean_weighted << " geo. mean\n";

        // save hypergraph
        saveResult(cfg, h_placement);
    } else {
        ERR(cfg) std::cerr << "WARNING, invalid placement !!\n";
    }

    return 0;
}
