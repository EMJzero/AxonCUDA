#include <tuple>
#include <chrono>
#include <concepts>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <algorithm>
#include <filesystem>

#include <omp.h>

#include "hgraph.hpp"
#include "curves.hpp"
#include "topology.hpp"
#include "nmhardware.hpp"
#include "runconfig_plc.hpp"

#include "utils.hpp"
#include "prims.hpp"
#include "eval_instr.hpp"
#include "utils_plc.hpp"
#include "data_types.hpp"
#include "data_types_plc.hpp"
#include "defines_plc.hpp"
#include "placement.hpp"
#include "ordering.hpp"
#include "prep.hpp"

using namespace hgraph;
using namespace hwmodel;
using namespace topology;
using namespace curve;
using namespace config_plc;


// first "model name" in /proc/cpuinfo
static std::string cpuModelName() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line))
        if (line.rfind("model name", 0) == 0)
            return line.substr(line.find(':') + 2);
    return "unknown";
}

int main(int argc, char** argv) {
    if (argc == 1) {
        printHelp();
        return 0;
    }

    // parse CLI args
    runconfig cfg = parseArgs(argc, argv);
    omp_set_num_threads((int)cfg.threads);

    // load hypergraph
    HyperGraph hg = loadHgraph(cfg);

    // print statistics
    std::cout << "Loaded hypergraph:\n";
    std::cout << "  Nodes:      " << hg.nodes() << "\n";
    std::cout << "  Hyperedges: " << hg.hedges().size() << "\n";
    std::cout << "  Total pins: " << hg.hedgesFlat().size() << "\n";
    std::cout << "  Total connections weight: " << std::fixed << std::setprecision(3) << hg.connectivity() << "\n";

    // topology-templated main routine
    auto place_routine = [&]<Topology T>() -> int {

        // setup the hardware model
        HardwareModel<T> hw = setupNMH<T>(cfg);

        // print hardware details
        std::cout << "Using hardware model \"" << hw.name() << "\":\n";
        std::cout << "  Topology:                 " << topologyToString(cfg.topology) << "\n";
        std::cout << "  Neurons per core:         " << hw.neuronsPerCore() << "\n";
        std::cout << "  Inbound axons per core:   " << hw.inboundPerCore() << "\n";
        std::cout << "  Synapses (pins) per core: " << hw.pinsPerCore() << "\n";
        std::cout << "  Cores per dim:            " << hw.coresAlongDim(0);
        for (uint32_t dim = 1; dim < T::dimensions; dim++)
            std::cout << ", " << hw.coresAlongDim(dim);
        std::cout << " (" << hw.coresCount() << " tot.)" << "\n";
        std::cout << "  Routing energy, latency: " << std::fixed << std::setprecision(3) << hw.energyPerRouting() << " pJ, " << hw.latencyPerRouting() << " ns\n";
        std::cout << "  Wire energy, latency:    " << std::fixed << std::setprecision(3) << hw.energyPerWire() << " pJ, " << hw.latencyPerWire() << " ns\n";

        std::cout << "Using settings:\n";
        std::cout << "  Seed:                            " << cfg.seed << "\n";
        std::cout << "  Force-directed iterations:       " << cfg.fd_iterations << "\n";
        std::cout << "  Force-directed candidates count: " << cfg.candidates_count << "\n";
        std::cout << "  Multi-start attempts:            ";
        if (cfg.multi_start_override == UINT32_MAX) std::cout << "<one per thread>\n";
        else std::cout << cfg.multi_start_override << "\n";
        std::cout << "  Multi-start batch size:          ";
        if (cfg.batch_size == UINT32_MAX) std::cout << "<multi-start count>\n";
        else std::cout << cfg.batch_size << "\n";
        std::cout << "  Label propagation repeats:       " << cfg.labelprop_repeats << "\n";
        std::cout << "  Space-filling curve:             " << SFCtoString(cfg.space_filling_curve) << "\n";
        std::cout << "  Routing policies:                " << routingPoliciesToString(cfg.unicast_metrics, cfg.xy_multicast_metrics, cfg.steiner_multicast_metrics) << "\n";
        std::cout << "  Flags: " << (cfg.parallel_touching_construction ? "ptc " : "") << (cfg.feedforward_order ? "ff " : "") << "\n";

        if (hg.nodes() > hw.coresCount()) {
            ERR(cfg) std::cerr << "ERROR, the hypergraph has more nodes (" << hg.nodes() << ") than the 2D lattice has points (" << hw.coresCount() << "), placement would fail !!\n";
            return 1;
        }

        if (cfg.candidates_count > T::neighborsCount()) {
            INFO(cfg) std::cout << "WARNING: lowering candidates count (-cnc) to " << T::neighborsCount() << ", which is the maximum neighbors count in the current topology !!\n";
            cfg.candidates_count = T::neighborsCount();
        }

        std::cout << "CPU:\n";
        std::cout << "  Model name:             " << cpuModelName() << "\n";
        std::cout << "  Hardware threads:       " << omp_get_num_procs() << "\n";
        std::cout << "  OpenMP threads:         " << cfg.threads << "\n";

        INFO(cfg) std::cout << "Preparing hypergraph data...\n";

        // build incidence sets while preparing the hypergraph, unless explicitly required to do it in parallel
        if (!cfg.parallel_touching_construction || cfg.feedforward_order)
            hg.buildIncidenceSets();

        const uint32_t num_hedges = static_cast<uint32_t>(hg.hedges().size());
        const dim_t hedges_size = hg.hedgesFlat().size();

        // total number of distinct nodes (for output indexing)
        const uint32_t num_nodes = hg.nodes(); // nodes count used when allocating outputs

        INFO(cfg) std::cout << "Starting timer...\n";
        auto time_start = std::chrono::high_resolution_clock::now();

        // Target topologies support is limited to those for which an exact close formula for distance between any two points exists!
        // => other topologies could be supported if they have at least a closed approximate formula, and the approximation is acceptable.
        // =>=> if a topology must be supported regardless, the best fallback is a lookup-based distance function with all pre-computed distances.
        // |
        // What to implement to support a topology:
        // - take the topology's intrinsic dimension and define the identifier of each point as a tuple with "# intrinsic dimensions" coordinates
        // - define an iterator for the set of points adjacent (directly connected, distance = 1) to any given one
        // - define a function to compute the distance between any two points in constant time (possibly closed-form, small lookup acceptable)
        // - define a 1D->topology locality-preserving mapping function for the initial placement ordering projection

        using Coord = Coord_t<T>;

        // topology / constraints
        topo<T> = hw.topology();
        if (static_cast<uint64_t>(topo<T>.extent().volume()) >= UINT32_MAX) {
            ERR(cfg) std::cerr << "ABORTING: the topology has " << topo<T>.extent().volume() << " places, overflowing the place idx space !!\n";
            abort();
        }
        const uint32_t volume = static_cast<uint32_t>(topo<T>.extent().volume());

        INFO(cfg) std::cout << "Setting up memory...\n";

        // hypergraph to be placed
        buffer<uint32_t> hedges(hedges_size); // contigous hedges array (each hedge must be stored as src+destinations, with the src in the first position)
        buffer<dim_t> hedges_offsets((dim_t)num_hedges + 1); // hedges_offsets[hedge idx] -> hedge start idx in hedges
        buffer<uint32_t> srcs_count(num_hedges); // srcs_count[hedge idx] -> number of sources of hedge idx
        buffer<float> hedge_weights(num_hedges); // hedge_weights[hedge idx] -> weight
        buffer<uint32_t> touching; // contigous inbound+outbout sets array (first inbound, then outbound)
        buffer<dim_t> touching_offsets; // touching_offsets[node idx] -> touching set start idx in touching

        // best placement
        buffer<Coord> best_placement(num_nodes); // placement[node idx] -> x and y placement coordinates of node
        buffer<uint32_t> best_inv_placement(volume); // inv_placement[flat coord idx] -> idx of the node occupying such place, or UINT32_MAX
        // |
        // best results
        float best_whops = FLT_MAX; // total weight of hedge hops in the current best solution

        // fill the hypergraph's arrays
        par_copy(hedges.data(), hg.hedgesFlat().data(), hedges_size);
        #pragma omp parallel for schedule(static) if(num_hedges > PARALLEL_GRAIN)
        for (uint32_t i = 0; i < num_hedges; ++i) {
            hedges_offsets[i] = static_cast<dim_t>(hg.hedges()[i].offset());
            srcs_count[i] = hg.hedges()[i].src_count();
            hedge_weights[i] = hg.hedges()[i].weight();
        }
        hedges_offsets[num_hedges] = hedges_size;

        // prepare touching sets
        EVP_PUSH("setup_touching");
        if (cfg.parallel_touching_construction) {
            std::tie(touching, touching_offsets) = buildTouching(
                cfg,
                hedges.data(),
                hedges_offsets.data(),
                num_nodes,
                num_hedges
            );
        } else {
            std::tie(touching, touching_offsets) = buildTouchingHost(
                cfg,
                hg
            );
        }
        EVP_POP(); // setup_touching

        // generate a 1D to 2D map for lattice points
        const std::vector<Coord> curve_1dto2d_placement = generatePlacementCurve<T>(
            cfg.space_filling_curve,
            num_nodes,
            topo<T>,
            cfg.verbose_info
        ); // curve_1dto2d_placement[position] -> coordinates of the position-th point of the node sequence mapped from 1D to 2D

        // determine multistart count
        uint32_t multi_start_count = cfg.multi_start_override;
        if (cfg.feedforward_order && multi_start_count > 1) {
            multi_start_count = 1u;
            ERR(cfg) std::cout << "WARNING, feedforward ordering doesn't support multi-start, forcing multi-start count to 1 !!\n";
        } else if (multi_start_count == UINT32_MAX) {
            // NOTE: in CUDA the default count is the one that maximally occupies the GPU, here it is one per thread
            multi_start_count = cfg.threads;
            INFO(cfg) std::cout << "Setting multi-start count to one per thread: " << multi_start_count << "\n";
        }
        // refine multi-start attempts in batches, every kernel handles a whole batch at once
        uint32_t batch_size = cfg.batch_size;
        if (batch_size == UINT32_MAX) {
            batch_size = multi_start_count;
            INFO(cfg) std::cout << "Setting batch size equal to multi-start count: " << batch_size << "\n";
        }
        batch_size = std::min(batch_size, multi_start_count);
        // node idxs are batch-flat inside every refinement structure, they must stay clear of the empty-cell sentinels
        if (static_cast<uint64_t>(batch_size) * num_nodes >= UINT32_MAX - T::neighborsCount()) {
            ERR(cfg) std::cerr << "ABORTING: batch size " << batch_size << " over " << num_nodes << " nodes overflows the batch-flat node idx space !!\n";
            abort();
        }

        INFO(cfg) std::cout << "Refining " << multi_start_count << " multi-start attempts in batches of " << batch_size << " ...\n";
        const int tid = 0;

        // batch solutions
        buffer<Coord> placement(static_cast<dim_t>(batch_size) * num_nodes); // placement[batch-flat node idx] -> placement coordinates of the node
        buffer<uint32_t> inv_placement(static_cast<dim_t>(batch_size) * volume); // inv_placement[start*volume + flat coord idx] -> batch-flat idx of the node occupying such place, or UINT32_MAX
        // |
        // batch scores
        std::vector<float> src_dst_distance(batch_size); // src_dst_distance[start] -> weighted avg. hedge max src-dst manhattan distance of that attempt
        std::vector<float> steiner_span(batch_size); // steiner_span[start] -> weighted avg. hedge Steiner tree span of that attempt

        INFO(cfg) std::cout << "Starting core timer...\n";
        auto time_core_start = std::chrono::high_resolution_clock::now();

        double avg_attempt_time = 0.0;

        for (uint32_t batch_begin = 0; batch_begin < multi_start_count; batch_begin += batch_size) {
            const uint32_t curr_batch = std::min(batch_size, multi_start_count - batch_begin);

            INFO(cfg) std::cout TID(tid) << "Beginning attempts [" << batch_begin << ", " << batch_begin + curr_batch << ") ...\n";

            INFO(cfg) std::cout TID(tid) << "Starting batch timer...\n";
            auto time_batch_start = std::chrono::high_resolution_clock::now();

            // initial placement
            EVP_PUSH("initial_placement");
            buffer<uint32_t> order_idx; // order_idx[node] -> position in its multi-start's 1D ordering for node
            if (cfg.feedforward_order) {
                // NOTE: feed-forward ordering is deterministic, so it forces the multi-start count - and thus the batch - down to 1
                INFO(cfg) std::cout TID(tid) << "Ordering nodes (sequential - might take a while) ...\n";
                const std::vector<uint32_t> nodes_order_idx = hg.feedForwardOrder();
                order_idx = buffer<uint32_t>(num_nodes);
                par_copy(order_idx.data(), nodes_order_idx.data(), num_nodes);
            } else {
                INFO(cfg) std::cout TID(tid) << "Ordering nodes (parallel - recursive bisection) ...\n";
                order_idx = locality_ordering(
                    cfg,
                    num_nodes,
                    curr_batch,
                    num_hedges,
                    hedges_size,
                    hedges.data(),
                    hedges_offsets.data(),
                    hedge_weights.data(),
                    touching.data(),
                    touching_offsets.data(),
                    cfg.seed + batch_begin
                );
            }

            // assign to each node its respective placement following both 1D orders (from ordered nodes to the lattice points map)
            // => order_idx holds positions local to each multi-start, so the shared 1D-to-(N)D map serves the whole batch
            par_gather(order_idx.data(), static_cast<dim_t>(curr_batch) * num_nodes, curve_1dto2d_placement.data(), placement.data());
            order_idx.release();

            for (uint32_t start = 0; start < curr_batch; start++) {
                // =============================
                // print some temporary results
                LOG(cfg) {
                    const std::vector<Coord> init_placement(placement.data() + start * num_nodes, placement.data() + (start + 1) * num_nodes);

                    if (hw.checkPlacementValidity(hg, init_placement, true)) {
                        if (cfg.unicast_metrics) {
                            DBG(cfg) std::cout TID(tid) << "Computing initial placement unicast metrics...\n";
                            auto metrics = hw.getAllUnicastMetrics(hg, init_placement);
                            std::cout TID(tid) << "Initial placement unicast metrics:\n";
                            std::cout TID(tid) << "  Energy:        " << std::fixed << std::setprecision(3) << metrics.energy.value() << "\n";
                            std::cout TID(tid) << "  Avg. latency:  " << std::fixed << std::setprecision(3) << metrics.avg_latency.value() << "\n";
                            std::cout TID(tid) << "  Max. latency:  " << std::fixed << std::setprecision(3) << metrics.max_latency.value() << "\n";
                            std::cout TID(tid) << "  Avg. congestion:  " << std::fixed << std::setprecision(3) << metrics.avg_congestion.value() << "\n";
                            std::cout TID(tid) << "  Max. congestion:  " << std::fixed << std::setprecision(3) << metrics.max_congestion.value() << "\n";
                            std::cout TID(tid) << "  Connections locality:\n";
                            std::cout TID(tid) << "    Flat:     " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean << " ar. mean, " << metrics.connections_locality.value().geo_mean << " geo. mean\n";
                            std::cout TID(tid) << "    Weighted: " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean_weighted << " ar. mean, " << metrics.connections_locality.value().geo_mean_weighted << " geo. mean\n";
                        }
                    } else {
                        std::cerr TID(tid) << "ERROR, invalid initial placement !!\n";
                        abort(); // should never happen
                    }

                    std::cout TID(tid) << "Initial placement:\n";
                    for (uint32_t i = 0; i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH); ++i)
                        std::cout TID(tid) << "  node " << i << " -> " << init_placement[i].toString() << "\n";
                }
                // =============================
            }

            // initialize inverse placement
            par_fill<uint32_t>(inv_placement.data(), static_cast<dim_t>(curr_batch) * volume, UINT32_MAX);
            LAUNCH(cfg) TID(tid) RUN << "inverse placement kernel (threads=" << cfg.threads << ") ...\n";
            inverse_placement_kernel<T>(
                placement.data(),
                num_nodes,
                curr_batch,
                volume,
                inv_placement.data()
            );

            // =============================
            // print some temporary results
            LOG(cfg) {
                std::cout TID(tid) << "Initial inverse placement (attempt " << batch_begin << "):\n";
                uint32_t neigh_rotator = 0u;
                Coord place;
                place.setAll(0);
                for (uint32_t i = 0; i < std::min<uint32_t>(volume, VERBOSE_LENGTH); ++i) {
                    std::cout TID(tid) << "  plc " << place.toString() << " -> " << inv_placement[topo<T>.flattenedIdx(place)] << "\n";
                    while (!topo<T>.contains(topo<T>.neighbor(place, neigh_rotator)))
                        neigh_rotator = (neigh_rotator + 1) % T::neighborsCount();
                    place = topo<T>.neighbor(place, neigh_rotator);
                    neigh_rotator = (neigh_rotator + 1) % T::neighborsCount(); // keep advancing so non-mesh topologies (e.g. Arbitrary) don't just bounce between 2 nodes
                }
            }
            // =============================

            EVP_POP(); // initial_placement

            // run force-directed refinement over the whole batch
            EVP_PUSH("fd_refinement");
            forceDirectedRefinement<T>(
                cfg,
                hedges.data(),
                hedges_offsets.data(),
                touching.data(),
                touching_offsets.data(),
                hedge_weights.data(),
                num_nodes,
                curr_batch,
                volume,
                placement.data(),
                inv_placement.data()
            );
            EVP_POP(); // fd_refinement

            // grade every solution in the batch
            EVP_PUSH("grade");
            getLocalityMetrics<T>(
                cfg,
                placement.data(),
                hedges.data(),
                hedges_offsets.data(),
                srcs_count.data(),
                hedge_weights.data(),
                num_hedges,
                num_nodes,
                curr_batch,
                src_dst_distance.data(),
                steiner_span.data()
            );
            EVP_POP(); // grade

            // update current best solution
            for (uint32_t start = 0; start < curr_batch; start++) {
                // TODO: tune these two coefficients
                const float curr_whops = 0.4 * src_dst_distance[start] + 0.6 * steiner_span[start]; // lower is better

                if (curr_whops < best_whops) {
                    best_whops = curr_whops;
                    par_copy(best_placement.data(), placement.data() + start * num_nodes, num_nodes);
                    // the winner's node idxs are batch-flat, rebase them - this is the result the rest of the tool reads
                    const uint32_t* start_inv_placement = inv_placement.data() + static_cast<dim_t>(start) * volume;
                    const uint32_t nodes_base = start * num_nodes;
                    #pragma omp parallel for schedule(static) if(volume > PARALLEL_GRAIN)
                    for (uint32_t i = 0; i < volume; i++)
                        best_inv_placement[i] = start_inv_placement[i] == UINT32_MAX ? UINT32_MAX : start_inv_placement[i] - nodes_base;
                    INFO(cfg) std::cout TID(tid) << "Updated best placement: attempt=" << batch_begin + start << " whops=" << std::fixed << std::setprecision(3) << curr_whops << "\n";
                } else {
                    INFO(cfg) std::cout TID(tid) << "Discarded placement: attempt=" << batch_begin + start << " whops=" << std::fixed << std::setprecision(3) << curr_whops << " > " << best_whops << "\n";
                }
            }

            auto time_batch_end = std::chrono::high_resolution_clock::now();
            avg_attempt_time += std::chrono::duration<double, std::milli>(time_batch_end - time_batch_start).count();
        }

        placement.release();
        inv_placement.release();

        // copy out the results
        const std::vector<Coord> final_placement(best_placement.data(), best_placement.data() + num_nodes);

        if (best_whops == FLT_MAX)
            ERR(cfg) std::cerr << "WARNING, the final mapping still has whops=FLT_MAX, no mapping seems to have been constructed !!\n";

        // =============================
        // print some example outputs
        LOG(cfg) {
            std::cout << "Final placement:\n";
            for (uint32_t i = 0; i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH); ++i)
                std::cout << "  node " << i << " -> " << final_placement[i].toString() << "\n";
            std::cout << "Final inverse placement:\n";
            uint32_t neigh_rotator = 0u;
            Coord place;
            place.setAll(0);
            for (uint32_t i = 0; i < std::min<uint32_t>(volume, VERBOSE_LENGTH); ++i) {
                std::cout << "  plc " << place.toString() << " -> " << best_inv_placement[topo<T>.flattenedIdx(place)] << "\n";
                while (!topo<T>.contains(topo<T>.neighbor(place, neigh_rotator)))
                    neigh_rotator = (neigh_rotator + 1) % T::neighborsCount();
                place = topo<T>.neighbor(place, neigh_rotator);
                neigh_rotator = (neigh_rotator + 1) % T::neighborsCount(); // keep advancing so non-mesh topologies (e.g. Arbitrary) don't just bounce between 2 nodes
            }
        }
        // =============================

        auto time_end = std::chrono::high_resolution_clock::now();
        INFO(cfg) std::cout << "Stopping timer...\n";

        INFO(cfg) std::cout << "Parallel section: complete; proceeding with placement results validation and evalution...\n";

        const double total_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
        // core: excluding the initialization of hedges and incidence sets
        const double core_ms = std::chrono::duration<double, std::milli>(time_end - time_core_start).count();
        avg_attempt_time /= multi_start_count;
        std::cout << "Average attempt execution time: " << std::fixed << std::setprecision(3) << avg_attempt_time << " ms\n";
        std::cout << "Total core execution time: " << std::fixed << std::setprecision(3) << core_ms << " ms\n";
        std::cout << "Total execution time: " << std::fixed << std::setprecision(3) << total_ms << " ms\n";
        INFO(cfg) EVP_REPORT();

        EVP_PUSH("postproc_metrics");
        if (hw.checkPlacementValidity(hg, final_placement, cfg.verbose_errs_and_warns)) {
            if (cfg.unicast_metrics) {
                DBG(cfg) std::cout << "Computing placement unicast metrics...\n";
                auto uc_metrics = hw.getAllUnicastMetrics(hg, final_placement);
                std::cout << "Placement unicast metrics:\n";
                std::cout << "  Energy:        " << std::fixed << std::setprecision(3) << uc_metrics.energy.value() << "\n";
                std::cout << "  Avg. latency:  " << std::fixed << std::setprecision(3) << uc_metrics.avg_latency.value() << "\n";
                std::cout << "  Max. latency:  " << std::fixed << std::setprecision(3) << uc_metrics.max_latency.value() << "\n";
                std::cout << "  Avg. congestion:  " << std::fixed << std::setprecision(3) << uc_metrics.avg_congestion.value() << "\n";
                std::cout << "  Max. congestion:  " << std::fixed << std::setprecision(3) << uc_metrics.max_congestion.value() << "\n";
                std::cout << "  Connections locality:\n";
                std::cout << "    Flat:     " << std::fixed << std::setprecision(3) << uc_metrics.connections_locality.value().ar_mean << " ar. mean, " << uc_metrics.connections_locality.value().geo_mean << " geo. mean\n";
                std::cout << "    Weighted: " << std::fixed << std::setprecision(3) << uc_metrics.connections_locality.value().ar_mean_weighted << " ar. mean, " << uc_metrics.connections_locality.value().geo_mean_weighted << " geo. mean\n";
            }

            if (cfg.xy_multicast_metrics) {
                DBG(cfg) std::cout << "Computing placement XY-multicast metrics...\n";
                auto xy_metrics = hw.getAllXYMulticastMetrics(hg, final_placement);
                std::cout << "Placement XY-multicast metrics:\n";
                if (xy_metrics.energy.has_value()) std::cout << "  Energy:          " << std::fixed << std::setprecision(3) << xy_metrics.energy.value() << "\n";
                else std::cout << "  Energy:          N/A (not implemented for this topology)\n";
                std::cout << "  Avg. latency:    " << std::fixed << std::setprecision(3) << xy_metrics.avg_latency.value() << "\n";
                if (xy_metrics.avg_congestion.has_value()) std::cout << "  Avg. congestion: " << std::fixed << std::setprecision(3) << xy_metrics.avg_congestion.value() << "\n";
                else std::cout << "  Avg. congestion: N/A (not implemented for this topology)\n";
                if (xy_metrics.max_congestion.has_value()) std::cout << "  Max. congestion: " << std::fixed << std::setprecision(3) << xy_metrics.max_congestion.value() << "\n";
                else std::cout << "  Max. congestion: N/A (not implemented for this topology)\n";
            }

            if (cfg.steiner_multicast_metrics) {
                DBG(cfg) std::cout << "Computing placement Steiner-multicast metrics...\n";
                auto mc_metrics = hw.getAllSteinerMulticastMetrics(hg, final_placement);
                std::cout << "Placement Steiner-multicast metrics:\n";
                if (mc_metrics.energy.has_value()) std::cout << "  Energy:          " << std::fixed << std::setprecision(3) << mc_metrics.energy.value() << "\n";
                else std::cout << "  Energy:          N/A (not implemented for this topology)\n";
                std::cout << "  Avg. latency:    " << std::fixed << std::setprecision(3) << mc_metrics.avg_latency.value() << "\n";
                if (mc_metrics.avg_congestion.has_value()) std::cout << "  Avg. congestion: " << std::fixed << std::setprecision(3) << mc_metrics.avg_congestion.value() << "\n";
                else std::cout << "  Avg. congestion: N/A (not implemented for this topology)\n";
                if (mc_metrics.max_congestion.has_value()) std::cout << "  Max. congestion: " << std::fixed << std::setprecision(3) << mc_metrics.max_congestion.value() << "\n";
                else std::cout << "  Max. congestion: N/A (not implemented for this topology)\n";
                std::cout << "  Evaluation fraction: " << std::fixed << std::setprecision(3) << mc_metrics.evaluation_fraction << "\n";
            }

            // save hypergraph
            saveResult<T>(cfg, final_placement);
        } else {
            ERR(cfg) std::cerr << "WARNING, invalid placement !!\n";
        }
        EVP_POP(); // postproc_metrics

        return 0;
    }; // place_routine end

    // dispatch based on topology
    switch (cfg.topology) {
        case TargetTopology::LATTICE2D:
            return place_routine.template operator()<Lattice<2>>();
        case TargetTopology::TORUS6D:
            return place_routine.template operator()<Torus<6>>();
        case TargetTopology::ARBITRARY:
            return place_routine.template operator()<ArbitraryGraph>();
        default:
            throw std::runtime_error("Topology not yet implemented *-* !");
    }
}
