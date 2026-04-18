#include <tuple>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <unordered_map>

#include "thruster.cuh"

#include "runconfig_plc.hpp"

#include "utils.cuh"
#include "utils_plc.cuh"
#include "defines_plc.cuh"
#include "placement.cuh"

void forceDirectedRefinement(
    const runconfig &cfg,
    const cudaDeviceProp props,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const uint32_t* d_touching,
    const dim_t* d_touching_offsets,
    const float* d_hedge_weights,
    const uint32_t num_nodes,
    coords* d_placement,
    uint32_t* d_inv_placement,
    const cudaStream_t stream,
    const int tid
) {
    auto thrust_exec = thrust::cuda::par.on(stream);
    // device pointers
    // refinement structures
    float *d_forces = nullptr; // forces[4*node idx + 0 for dx, + 1 for sx, +2 for up, +2 for down] -> direction of the node's two proposed moves
    uint32_t *d_pairs = nullptr; // pairs[4*node idx + 0..] -> nodes the current one wants to swap with, ordered by decreasing score
    uint32_t *d_scores = nullptr; // scores[4*node idx + 0..] -> score with which node wants to pair with other nodes
    slot *d_swap_slots = nullptr; // slot to finalize node pairs while computing exclusive swaps (true dtype: "slot")
    uint32_t *d_swap_flags = nullptr; // swap_flag[node idx] -> set to 1 for the lower-id of each node in a swap-pair, in order to create swap-events
    // events structures
    swap *d_ev_swaps = nullptr; // ev_swaps[event idx] -> pair of nodes involved in the event's swap
    float *d_ev_scores = nullptr; // ev_scores[event idx] -> score (cost gain) achieved by the event's swap
    uint32_t *d_nodes_rank = nullptr; // node_rank[node idx] -> rank (index) in the sorted events by score of the node

    CUDA_CHECK(cudaMallocAsync(&d_forces, 4 * num_nodes * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_pairs, cfg.candidates_count * num_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scores, cfg.candidates_count * num_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_swap_slots, num_nodes * sizeof(slot), stream));
    CUDA_CHECK(cudaMallocAsync(&d_swap_flags, (num_nodes + 1) * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ev_swaps, num_nodes * sizeof(swap), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ev_scores, num_nodes * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_nodes_rank, num_nodes * sizeof(uint32_t), stream));

    // thrust pointers
    thrust::device_ptr<slot> t_swap_slots(d_swap_slots);
    thrust::device_ptr<uint32_t> t_swap_flags(d_swap_flags);
    thrust::device_ptr<swap> t_ev_swaps(d_ev_swaps);
    thrust::device_ptr<float> t_ev_scores(d_ev_scores);

    for (uint32_t iter = 0; iter < cfg.fd_iterations; iter++) {
        INFO(cfg) std::cout TID(tid) << "Force-directed refinement, iteration " << iter << "\n";

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

        {
            // launch configuration - forces kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_nodes ; // 1 warp per node
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - forces kernel
            LAUNCH(cfg) TID(tid) RUN << "forces kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            forces_kernel<<<blocks, threads_per_block, 0, stream>>>(
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
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // =============================
        // print some temporary results
        LOG(cfg) logForces(
            d_forces,
            num_nodes,
            stream,
            tid
        );
        // =============================

        {
            // launch configuration - tensions kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_nodes ; // 1 warp per node
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - tensions kernel
            LAUNCH(cfg) TID(tid) RUN << "tensions kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            tensions_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_placement,
                d_inv_placement,
                d_forces,
                num_nodes,
                cfg.candidates_count,
                d_pairs,
                d_scores
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // =============================
        // print some temporary results
        LOG(cfg) logTensions(
            cfg,
            d_pairs,
            d_scores,
            num_nodes,
            stream,
            tid
        );
        // =============================

        // zero-out swap slots and flags
        slot init_slot; init_slot.id = UINT32_MAX; init_slot.score = 0u;
        thrust::fill(thrust_exec, t_swap_slots, t_swap_slots + num_nodes, init_slot); // upper 32 bits to 0x00, lower 32 to 0xFF
        CUDA_CHECK(cudaMemsetAsync(d_swap_flags, 0x00, (num_nodes + 1) * sizeof(uint32_t), stream));

        {
            // launch configuration - exclusive swaps kernel
            int threads_per_block = 256;
            int num_threads_needed = num_nodes; // 1 thread per node
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            size_t bytes_per_thread = 0; //TODO
            size_t shared_bytes = threads_per_block * bytes_per_thread;
            // additional checks for the cooperative kernel mode
            int blocks_per_SM = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_SM, exclusive_swaps_kernel, threads_per_block, shared_bytes);
            int max_blocks = blocks_per_SM * props.multiProcessorCount;
            if (blocks > max_blocks) {
                const uint32_t num_repeats = (blocks + max_blocks - 1) / max_blocks;
                INFO(cfg) std::cout TID(tid) << "NOTE: exclusive swaps kernel required blocks=" << blocks << ", but max-blocks=" << max_blocks << ", setting repeats=" << num_repeats << " ...\n";
                blocks = (blocks + num_repeats - 1) / num_repeats;
                if (num_repeats > MAX_SWAPS_MATCHING_REPEATS) {
                    ERR(cfg) std::cerr TID(tid) << "ABORTING: exclusive swaps kernel required repeats=" << num_repeats << ", but max-repeats=" << MAX_SWAPS_MATCHING_REPEATS << " !!\n";
                    abort();
                }
            }
            // launch - exclusive swaps kernel
            LAUNCH(cfg) TID(tid) RUN << "exclusive swaps kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            void *kernel_args[] = {
                (void*)&d_pairs,
                (void*)&d_scores,
                (void*)&num_nodes,
                (void*)&cfg.candidates_count,
                (void*)&d_swap_slots,
                (void*)&d_swap_flags
            };
            cudaLaunchCooperativeKernel((void*)exclusive_swaps_kernel, blocks, threads_per_block, kernel_args, shared_bytes, stream);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // =============================
        // print some temporary results
        LOG(cfg) logSwapPairs(
            d_swap_slots,
            d_swap_flags,
            num_nodes,
            stream,
            tid
        );
        // =============================

        // scan flags to give each event its offset
        thrust::exclusive_scan(thrust_exec, t_swap_flags, t_swap_flags + (num_nodes + 1), t_swap_flags);
        uint32_t num_events;
        CUDA_CHECK(cudaMemcpyAsync(&num_events, d_swap_flags + num_nodes, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        INFO(cfg) std::cout TID(tid) << "Number of events produced: " << num_events << " ...\n";
        if (num_events > (num_nodes + 1) / 2)
            ERR(cfg) std::cerr TID(tid) << "WARNING, there are more events (" << num_events << ") than half the node (" << (num_nodes + 1) / 2 << "), this >may< an undesirable situation ...\n";
        else if (num_events == 0) {
            INFO(cfg) std::cout TID(tid) << "Stopping with no events (viable swaps), on iteration " << iter << "\n";
            break;
        }

        /*
        * IDEA, generate events kernel(s):
        * - extract node1 (lowest id), node2, score in events
        * - no need to rank events, just sort them by score and carry nodes along
        * - now allocate ranks, one entry per node, initialized to UINT32_MAX, then from each event write to its two nodes the event's index (rank)
        * - update the gain of each event in-sequence:
        *   - one warp per event, visit each of the node's touching hedges, one pin per thread in the warp
        *   - for each pin, see its rank, and update its position accordingly
        * - find the maximum gain subsequence and apply it
        */

        {
            // launch configuration - events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_nodes; // 1 thread per node
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - events kernel
            LAUNCH(cfg) TID(tid) RUN << "events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            swap_events_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_swap_slots,
                d_swap_flags,
                num_nodes,
                d_ev_swaps,
                d_ev_scores
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // sort (ascending) events by score while carrying swapped nodes along
        thrust::sort_by_key(thrust_exec, t_ev_scores, t_ev_scores + num_events, t_ev_swaps, thrust::greater<float>());
        CUDA_CHECK(cudaMemsetAsync(d_nodes_rank, 0xFF, num_nodes * sizeof(uint32_t), stream));

        // =============================
        // print some temporary results
        LOG(cfg) logEvents(
            d_ev_swaps,
            d_ev_scores,
            num_nodes,
            "sorted - in isolation",
            stream,
            tid
        );
        // =============================

        {
            // launch configuration - scatter ranks kernel
            int threads_per_block = 128;
            int num_threads_needed = num_events; // 1 thread per event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - scatter ranks kernel
            LAUNCH(cfg) TID(tid) RUN << "scatter ranks kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            scatter_ranks_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_ev_swaps,
                num_events,
                d_nodes_rank
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        {
            // launch configuration - cascade kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_events ; // 1 warp per event
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - cascade kernel
            LAUNCH(cfg) TID(tid) RUN << "cascade kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            cascade_kernel<<<blocks, threads_per_block, 0, stream>>>(
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
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // =============================
        // print some temporary results
        LOG(cfg) logEvents(
            d_ev_swaps,
            d_ev_scores,
            num_nodes,
            "cascade - in sequence",
            stream,
            tid
        );
        // =============================

        // scan the new scores, find the maximum gain subsequence
        thrust::inclusive_scan(thrust_exec, t_ev_scores, t_ev_scores + num_events, t_ev_scores);
        auto best_ev_pos = thrust::max_element(thrust_exec, t_ev_scores, t_ev_scores + num_events);
        const uint32_t best_ev = static_cast<uint32_t>(best_ev_pos - t_ev_scores);
        const uint32_t num_good_swaps = best_ev + 1;
        float gain;
        CUDA_CHECK(cudaMemcpyAsync(&gain, d_ev_scores + best_ev, sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // stop if the sequence has length 0 or the gain is too low
        if (num_good_swaps == 0 || gain < 0.001f) {
            INFO(cfg) {
                if (num_good_swaps == 0) std::cout TID(tid) << "Stopping with no further improving swaps, on iteration " << iter << "\n";
                else std::cout TID(tid) << "Stopping with gain" << std::fixed << std::setprecision(3) << gain << " ( < 10^-3) on iteration " << iter << "\n";
            }
            break;
        } else {
            INFO(cfg) std::cout TID(tid) << "Number of good swaps performed: " << num_good_swaps << " ...\n";
        }

        // update placement and inv_placement
        {
            // launch configuration - apply swaps kernel
            int threads_per_block = 128;
            int num_threads_needed = num_good_swaps; // 1 thread per swap
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - apply swaps kernel
            LAUNCH(cfg) TID(tid) RUN << "apply swaps kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            apply_swaps_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_ev_swaps,
                num_good_swaps,
                d_placement,
                d_inv_placement
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    CUDA_CHECK(cudaFreeAsync(d_forces, stream));
    CUDA_CHECK(cudaFreeAsync(d_pairs, stream));
    CUDA_CHECK(cudaFreeAsync(d_scores, stream));
    CUDA_CHECK(cudaFreeAsync(d_swap_slots, stream));
    CUDA_CHECK(cudaFreeAsync(d_swap_flags, stream));
    CUDA_CHECK(cudaFreeAsync(d_ev_swaps, stream));
    CUDA_CHECK(cudaFreeAsync(d_ev_scores, stream));
    CUDA_CHECK(cudaFreeAsync(d_nodes_rank, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// return the weighted average hedge max src-dst manhattan distance and weighted average hedge Steiner tree span (<2x upper bound)
std::tuple<float, float> getLocalityMetrics(
    const runconfig &cfg,
    const coords* d_placement,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const uint32_t* d_srcs_count,
    const float* d_hedge_weights,
    const uint32_t num_hedges,
    const cudaStream_t stream,
    const int tid
) {
    /*
    * IDEA:
    * - NOTE: this is a good-guess/temporary solution
    * - kernel that computes, for each hedge, the max src-dst manhattan distance (proxy for latency)
    * - kernel that computes, for each hedge, an upper estimate of the min Steiner tree span
    * - weight each estimate by the spiking frequency and reduce across hedges
    * - merge the two reduced values into a single quality metric based on the ratio between E_T, E_R, and L_T, L_R
    *
    * Feasible Steiner approximation:
    * - you cannot "ricochet" off of other lattice points
    * - for each node, add to the total distance the distance between it, and the closes among all other nodes in the same hedge
    *   => this the facto creates a minimum spanning tree over the hedge's pins, connecting each pin to the other one closes to it
    *     => the "minimum spanning tree" is defined over a higher-level fully connected graph where only lattice points occupied by
    *        the hedge's pins exist, and each edge is weighted by the manhattan distance between said points
    *     => hence, with the minimum spanning tree, you always pay the minimum path distance between node pairs, even if you reuse links
    *   => the minimum spanning tree will surely have span <= 2x the minimum Steiner tree
    * - minimum spanning tree complexity: iterate over hedges, over pins of each, and for each pin over pins again (to find the closest), e+d^2
    *   => little issue:
    *     - a pin already part of the minimum spanning tree must not be reconsidered by subsequent nodes...
    *     => Prim-style algoritm, expanding sequentially the connected spanning tree inside each hedge's graph
    */
    
    float *d_result = nullptr; // result[hedge idx] -> temporary result for the hedge
    CUDA_CHECK(cudaMallocAsync(&d_result, 4 * num_hedges * sizeof(float), stream));
    thrust::device_ptr<float> t_result(d_result);
    
    // compute the max src-dst manhattan distance per hedge
    {
        // launch configuration - max src-dst distance kernel
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = num_hedges ; // 1 warp per hedge
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - max src-dst distance kernel
        LAUNCH(cfg) TID(tid) RUN << "max src-dst distance kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        max_src_dst_distance_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_placement,
            d_hedges,
            d_hedges_offsets,
            d_srcs_count,
            d_hedge_weights,
            num_hedges,
            d_result // <- alredy multiply by hedge weight
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    float total_src_dst_distance = thrust::reduce(thrust::device, t_result, t_result + num_hedges, 0.0f, thrust::plus<float>());

    // compute the Steiner tree span upper bound per hedge, given by the weighted spanning tree overs its pins' complete graph
    {
        // launch configuration - min spanning tree weight kernel
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = num_hedges ; // 1 warp per hedge
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - min spanning tree weight kernel
        LAUNCH(cfg) TID(tid) RUN << "min spanning tree weight kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        min_spanning_tree_weight_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_placement,
            d_hedges,
            d_hedges_offsets,
            d_hedge_weights,
            num_hedges,
            d_result // <- alredy multiply by hedge weight
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    float total_steiner_span = thrust::reduce(thrust::device, t_result, t_result + num_hedges, 0.0f, thrust::plus<float>());

    return std::make_tuple(total_src_dst_distance, total_steiner_span);
}


// LOGGING

void logForces(
    const float *d_forces,
    const uint32_t num_nodes,
    const cudaStream_t stream,
    const int tid
) {
    std::vector<float> forces_tmp(num_nodes * 4);
    CUDA_CHECK(cudaMemcpyAsync(forces_tmp.data(), d_forces, num_nodes * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout TID(tid) << "Forces:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            std::cout TID(tid) << "  node " << i << " ->";
            std::cout << " (" << forces_tmp[4 * i + LEFT] << " LEFT)";
            std::cout << " (" << forces_tmp[4 * i + RIGHT] << " RIGHT)";
            std::cout << " (" << forces_tmp[4 * i + UP] << " UP)";
            std::cout << " (" << forces_tmp[4 * i + DOWN] << " DOWN)\n";
        }
    }
}

void logTensions(
    const runconfig &cfg,
    const uint32_t *d_pairs,
    const uint32_t *d_scores,
    const uint32_t num_nodes,
    const cudaStream_t stream,
    const int tid
) {
    std::vector<uint32_t> pairs_tmp(num_nodes * cfg.candidates_count);
    std::vector<uint32_t> scores_tmp(num_nodes * cfg.candidates_count);
    CUDA_CHECK(cudaMemcpyAsync(pairs_tmp.data(), d_pairs, num_nodes * sizeof(uint32_t) * cfg.candidates_count, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(scores_tmp.data(), d_scores, num_nodes * sizeof(uint32_t) * cfg.candidates_count, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::unordered_map<uint32_t, int> groups_count;
    std::cout TID(tid) << "Tensions:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            std::cout TID(tid) << "  node " << i << " ->";
            for (uint32_t j = 0; j < cfg.candidates_count; ++j) {
                uint32_t target = pairs_tmp[i * cfg.candidates_count + j];
                uint32_t score = scores_tmp[i * cfg.candidates_count + j];
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
}

void logSwapPairs(
    const slot *d_swap_slots,
    const uint32_t *d_swap_flags,
    const uint32_t num_nodes,
    const cudaStream_t stream,
    const int tid
) {
    std::vector<slot> slots_tmp(num_nodes);
    std::vector<uint32_t> flags_tmp(num_nodes); // leave out the last flag, fine
    CUDA_CHECK(cudaMemcpyAsync(slots_tmp.data(), d_swap_slots, num_nodes * sizeof(slot), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(flags_tmp.data(), d_swap_flags, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout TID(tid) << "Swap pairs:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            slot node_slot = slots_tmp[i];
            if (node_slot.id == UINT32_MAX) std::cout TID(tid) << "  node " << i << " -> target=none score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
            else if (node_slot.id == UINT32_MAX - LEFT) std::cout TID(tid) << "  node " << i << " -> target=LEFT score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
            else if (node_slot.id == UINT32_MAX - RIGHT) std::cout TID(tid) << "  node " << i << " -> target=RIGHT score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
            else if (node_slot.id == UINT32_MAX - UP) std::cout TID(tid) << "  node " << i << " -> target=UP score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
            else if (node_slot.id == UINT32_MAX - DOWN) std::cout TID(tid) << "  node " << i << " -> target=DOWN score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
            else std::cout TID(tid) << "  node " << i << " -> target=" << node_slot.id << " score=" << node_slot.score << " flag=" << flags_tmp[i] << "\n";
        }
    }
}

void logEvents(
    const swap *d_ev_swaps,
    const float *d_ev_scores,
    const uint32_t num_nodes,
    const  std::string flare,
    const cudaStream_t stream,
    const int tid
) {
    std::vector<swap> ev_swaps_tmp(num_nodes);
    std::vector<float> ev_scores_tmp(num_nodes);
    CUDA_CHECK(cudaMemcpyAsync(ev_swaps_tmp.data(), d_ev_swaps, num_nodes * sizeof(swap), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(ev_scores_tmp.data(), d_ev_scores, num_nodes * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout TID(tid) << "Events (" << flare << "):\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            swap ev_swap = ev_swaps_tmp[i];
            float ev_score = ev_scores_tmp[i];
            std::cout TID(tid) << "  event " << i << " -> lo=" << ev_swap.lo << " hi=" << ev_swap.hi << " score=" << ev_score << "\n";
        }
    }
}
