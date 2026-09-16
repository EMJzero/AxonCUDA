#include <tuple>
#include <vector>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include <cub/cub.cuh>

#include "thruster.cuh"

#include "topology.hpp"
#include "runconfig_plc.hpp"

#include "utils.cuh"
#include "utils_plc.cuh"
#include "defines_plc.cuh"
#include "placement.cuh"

template<Topology T>
void forceDirectedRefinement(
    const runconfig &cfg,
    const cudaDeviceProp props,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const uint32_t* d_touching,
    const dim_t* d_touching_offsets,
    const float* d_hedge_weights,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    Coord_t<T>* d_placement,
    uint32_t* d_inv_placement,
    const cudaStream_t stream,
    const int tid
) {
    auto thrust_exec = thrust::cuda::par.on(stream);

    // the whole batch of multi-starts is refined by one launch per step
    // => every per-multi-start array below is "batch_size" contiguous segments, multi-start "b" owning [b*size, (b+1)*size)
    const uint32_t batch_nodes = batch_size * num_nodes;

    // device pointers
    // refinement structures
    float *d_forces = nullptr; // forces[neighborsCount*node idx + neigh_idx] -> gain from moving the node towards its neigh_idx-th neighbor
    uint32_t *d_pairs = nullptr; // pairs[4*node idx + 0..] -> nodes the current one wants to swap with, ordered by decreasing score
    uint32_t *d_scores = nullptr; // scores[4*node idx + 0..] -> score with which node wants to pair with other nodes
    slot *d_swap_slots = nullptr; // slot to finalize node pairs while computing exclusive swaps (true dtype: "slot")
    // events structures
    // NOTE: events are not compacted, node idx "i" owns event slot "i" and empty slots carry -FLT_MAX, sorting behind every real event
    // => no device-to-host readback is needed to size any launch, and the loop below runs entirely without synchronization
    swap *d_ev_swaps = nullptr; // ev_swaps[event idx] -> pair of nodes involved in the event's swap
    float *d_ev_scores = nullptr; // ev_scores[event idx] -> score (cost gain) achieved by the event's swap
    swap *d_ev_swaps_buffer = nullptr; // pong buffer for the events' segmented sort
    float *d_ev_scores_buffer = nullptr; // pong buffer for the events' segmented sort
    uint32_t *d_ev_offsets = nullptr; // ev_offsets[start] -> first event slot of that multi-start
    uint32_t *d_nodes_rank = nullptr; // node_rank[node idx] -> rank (index) in the sorted events by score of the node, local to its multi-start
    // batch control structures
    uint32_t *d_num_good_swaps = nullptr; // num_good_swaps[start] -> length of that multi-start's improving event prefix
    uint8_t *d_active = nullptr; // active[start] -> 0 once that multi-start converged, its threads return immediately from then on

    CUDA_CHECK(cudaMallocAsync(&d_forces, T::neighborsCount() * batch_nodes * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_pairs, cfg.candidates_count * batch_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scores, cfg.candidates_count * batch_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_swap_slots, batch_nodes * sizeof(slot), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ev_swaps, batch_nodes * sizeof(swap), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ev_scores, batch_nodes * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ev_swaps_buffer, batch_nodes * sizeof(swap), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ev_scores_buffer, batch_nodes * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ev_offsets, (batch_size + 1) * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_nodes_rank, batch_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_num_good_swaps, batch_size * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_active, batch_size * sizeof(uint8_t), stream));

    // thrust pointers
    thrust::device_ptr<slot> t_swap_slots(d_swap_slots);
    thrust::device_ptr<uint32_t> t_ev_offsets(d_ev_offsets);

    // every multi-start owns the same number of event slots -> fixed-length segments for the sort below
    thrust::transform(
        thrust_exec,
        thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(batch_size + 1),
        t_ev_offsets,
        [num_nodes] __device__ (uint32_t start) { return start * num_nodes; }
    );

    // every multi-start starts out active, and retires itself from 'prefix_gain_kernel' once it stops improving
    CUDA_CHECK(cudaMemsetAsync(d_active, 0x01, batch_size * sizeof(uint8_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_num_good_swaps, 0x00, batch_size * sizeof(uint32_t), stream));

    // size the events' segmented sort scratch once, it only depends on the batch shape
    void *c_ev_sort_storage = nullptr;
    size_t c_ev_sort_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        c_ev_sort_storage, c_ev_sort_storage_bytes,
        d_ev_scores, d_ev_scores_buffer,
        d_ev_swaps, d_ev_swaps_buffer,
        batch_nodes, batch_size,
        d_ev_offsets, d_ev_offsets + 1,
        0, sizeof(float) * 8, stream
    ));
    CUDA_CHECK(cudaMallocAsync(&c_ev_sort_storage, c_ev_sort_storage_bytes, stream));

    // pinned, so the periodic activity check does not stall the host on a pageable copy
    uint8_t *h_active = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_active, batch_size * sizeof(uint8_t)));

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
            int num_warps_needed = batch_nodes ; // 1 warp per node
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - forces kernel
            LAUNCH(cfg) TID(tid) RUN << "forces kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            forces_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
                d_hedges,
                d_hedges_offsets,
                d_touching,
                d_touching_offsets,
                d_hedge_weights,
                d_placement,
                num_nodes,
                batch_size,
                d_active,
                d_forces
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // =============================
        // print some temporary results
        LOG(cfg) logForces<T>(
            d_forces,
            batch_nodes,
            stream,
            tid
        );
        // =============================

        {
            // launch configuration - tensions kernel
            int threads_per_block = 128;
            int num_threads_needed = batch_nodes; // 1 thread per node
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - tensions kernel
            LAUNCH(cfg) TID(tid) RUN << "tensions kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            tensions_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
                d_placement,
                d_inv_placement,
                d_forces,
                num_nodes,
                batch_size,
                volume,
                d_active,
                cfg.candidates_count,
                d_pairs,
                d_scores
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // =============================
        // print some temporary results
        LOG(cfg) logTensions<T>(
            cfg,
            d_pairs,
            d_scores,
            batch_nodes,
            stream,
            tid
        );
        // =============================

        // zero-out swap slots
        slot init_slot; init_slot.id = UINT32_MAX; init_slot.score = 0u;
        thrust::fill(thrust_exec, t_swap_slots, t_swap_slots + batch_nodes, init_slot); // upper 32 bits to 0x00, lower 32 to 0xFF

        {
            // launch configuration - exclusive swaps kernel
            int threads_per_block = 256;
            int num_threads_needed = batch_nodes; // 1 thread per node
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            size_t bytes_per_thread = 0; //TODO
            size_t shared_bytes = threads_per_block * bytes_per_thread;
            // additional checks for the cooperative kernel mode
            int blocks_per_SM = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_SM, exclusive_swaps_kernel<T>, threads_per_block, shared_bytes);
            int max_blocks = blocks_per_SM * props.multiProcessorCount;
            if (blocks > max_blocks) {
                const uint32_t num_repeats = (blocks + max_blocks - 1) / max_blocks;
                INFO(cfg) std::cout TID(tid) << "NOTE: exclusive swaps kernel required blocks=" << blocks << ", but max-blocks=" << max_blocks << ", setting repeats=" << num_repeats << " ...\n";
                blocks = (blocks + num_repeats - 1) / num_repeats;
                if (num_repeats > MAX_SWAPS_MATCHING_REPEATS) {
                    ERR(cfg) std::cerr TID(tid) << "ABORTING: exclusive swaps kernel required repeats=" << num_repeats << ", but max-repeats=" << MAX_SWAPS_MATCHING_REPEATS << " !!\n";
                    abort();
                }
                // every repeat appends its own walk to the same per-thread path, before the downward pass unwinds them all
                if (num_repeats * num_nodes > SWAPS_PATH_SIZE) {
                    ERR(cfg) std::cerr TID(tid) << "ABORTING: exclusive swaps kernel required up to " << num_repeats * num_nodes << " path slots per thread, but max-path-size=" << SWAPS_PATH_SIZE << ", lower the batch size !!\n";
                    abort();
                }
            }
            // launch - exclusive swaps kernel
            // NOTE: sizing the grid off the item count keeps every block non-empty, which the grid.sync()-es inside require
            LAUNCH(cfg) TID(tid) RUN << "exclusive swaps kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            void *kernel_args[] = {
                (void*)&d_pairs,
                (void*)&d_scores,
                (void*)&num_nodes,
                (void*)&batch_size,
                (void*)&d_active,
                (void*)&cfg.candidates_count,
                (void*)&d_swap_slots
            };
            CUDA_CHECK(cudaLaunchCooperativeKernel((void*)exclusive_swaps_kernel<T>, blocks, threads_per_block, kernel_args, shared_bytes, stream));
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // =============================
        // print some temporary results
        LOG(cfg) logSwapPairs<T>(
            d_swap_slots,
            batch_nodes,
            stream,
            tid
        );
        // =============================

        {
            // launch configuration - events kernel
            int threads_per_block = 128;
            int num_threads_needed = batch_nodes; // 1 thread per node
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - events kernel
            LAUNCH(cfg) TID(tid) RUN << "events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            swap_events_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
                d_swap_slots,
                num_nodes,
                batch_size,
                d_active,
                d_ev_swaps,
                d_ev_scores
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // sort (descending) each multi-start's events by score while carrying swapped nodes along
        // NOTE: segmented, and not batch-wide, so that a multi-start's event order never depends on what it is batched with
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
            c_ev_sort_storage, c_ev_sort_storage_bytes,
            d_ev_scores, d_ev_scores_buffer,
            d_ev_swaps, d_ev_swaps_buffer,
            batch_nodes, batch_size,
            d_ev_offsets, d_ev_offsets + 1,
            0, sizeof(float) * 8, stream
        ));
        std::swap(d_ev_scores, d_ev_scores_buffer);
        std::swap(d_ev_swaps, d_ev_swaps_buffer);
        CUDA_CHECK(cudaMemsetAsync(d_nodes_rank, 0xFF, batch_nodes * sizeof(uint32_t), stream));

        // =============================
        // print some temporary results
        LOG(cfg) logEvents(
            d_ev_swaps,
            d_ev_scores,
            batch_nodes,
            "sorted - in isolation",
            stream,
            tid
        );
        // =============================

        {
            // launch configuration - scatter ranks kernel
            int threads_per_block = 128;
            int num_threads_needed = batch_nodes; // 1 thread per event slot
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - scatter ranks kernel
            LAUNCH(cfg) TID(tid) RUN << "scatter ranks kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            scatter_ranks_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
                d_ev_swaps,
                num_nodes,
                batch_size,
                d_active,
                d_nodes_rank
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        {
            // launch configuration - resolve empty conflicts kernel
            int threads_per_block = 128;
            int num_threads_needed = batch_nodes; // 1 thread per event slot
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - resolve empty conflicts kernel
            LAUNCH(cfg) TID(tid) RUN << "resolve empty conflicts kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            resolve_empty_conflicts_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
                d_placement,
                d_inv_placement,
                d_nodes_rank,
                num_nodes,
                batch_size,
                volume,
                d_active,
                d_ev_swaps
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        {
            // launch configuration - cascade kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = batch_nodes ; // 1 warp per event slot
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - cascade kernel
            LAUNCH(cfg) TID(tid) RUN << "cascade kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            cascade_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
                d_hedges,
                d_hedges_offsets,
                d_touching,
                d_touching_offsets,
                d_hedge_weights,
                d_placement,
                d_ev_swaps,
                d_nodes_rank,
                num_nodes,
                batch_size,
                d_active,
                d_ev_scores
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // =============================
        // print some temporary results
        LOG(cfg) logEvents(
            d_ev_swaps,
            d_ev_scores,
            batch_nodes,
            "cascade - in sequence",
            stream,
            tid
        );
        // =============================

        {
            // launch configuration - prefix gain kernel
            int threads_per_block = PREFIX_GAIN_THREADS;
            int blocks = batch_size; // 1 block per multi-start
            // launch - prefix gain kernel
            LAUNCH(cfg) TID(tid) RUN << "prefix gain kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            prefix_gain_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_ev_scores,
                num_nodes,
                d_num_good_swaps,
                d_active
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // update placement and inv_placement
        {
            // launch configuration - apply swaps kernel
            int threads_per_block = 128;
            int num_threads_needed = batch_nodes; // 1 thread per event slot
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - apply swaps kernel
            LAUNCH(cfg) TID(tid) RUN << "apply swaps kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            apply_swaps_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
                d_ev_swaps,
                d_num_good_swaps,
                num_nodes,
                batch_size,
                volume,
                d_active,
                d_placement,
                d_inv_placement
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // multi-starts retire themselves on the device, so only check on the host every so often
        if ((iter + 1) % FD_ACTIVE_CHECK_PERIOD == 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_active, d_active, batch_size * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint32_t active_count = 0u;
            for (uint32_t start = 0; start < batch_size; start++) active_count += h_active[start];
            INFO(cfg) std::cout TID(tid) << "Multi-starts still improving after iteration " << iter << ": " << active_count << "/" << batch_size << "\n";
            if (active_count == 0u) {
                INFO(cfg) std::cout TID(tid) << "Stopping, the whole batch converged on iteration " << iter << "\n";
                break;
            }
        }
    }

    CUDA_CHECK(cudaFreeHost(h_active));
    CUDA_CHECK(cudaFreeAsync(d_forces, stream));
    CUDA_CHECK(cudaFreeAsync(d_pairs, stream));
    CUDA_CHECK(cudaFreeAsync(d_scores, stream));
    CUDA_CHECK(cudaFreeAsync(d_swap_slots, stream));
    CUDA_CHECK(cudaFreeAsync(d_ev_swaps, stream));
    CUDA_CHECK(cudaFreeAsync(d_ev_scores, stream));
    CUDA_CHECK(cudaFreeAsync(d_ev_swaps_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(d_ev_scores_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(d_ev_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(d_nodes_rank, stream));
    CUDA_CHECK(cudaFreeAsync(d_num_good_swaps, stream));
    CUDA_CHECK(cudaFreeAsync(d_active, stream));
    CUDA_CHECK(cudaFreeAsync(c_ev_sort_storage, stream));
    DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
}

// per multi-start, return the weighted average hedge max src-dst manhattan distance and weighted average hedge Steiner tree span (<2x upper bound)
template<Topology T>
void getLocalityMetrics(
    const runconfig &cfg,
    const Coord_t<T>* d_placement,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const uint32_t* d_srcs_count,
    const float* d_hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    float* d_src_dst_distance, // src_dst_distance[start] -> weighted avg. hedge max src-dst manhattan distance of that multi-start
    float* d_steiner_span, // steiner_span[start] -> weighted avg. hedge Steiner tree span of that multi-start
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

    auto thrust_exec = thrust::cuda::par.on(stream);

    // the whole batch is graded by one launch per metric, then reduced back down to one figure per multi-start
    const uint32_t batch_hedges = batch_size * num_hedges;

    float *d_result = nullptr; // result[hedge idx] -> temporary result for the hedge
    CUDA_CHECK(cudaMallocAsync(&d_result, batch_hedges * sizeof(float), stream));
    uint32_t *d_hedge_offsets = nullptr; // hedge_offsets[start] -> first result slot of that multi-start
    CUDA_CHECK(cudaMallocAsync(&d_hedge_offsets, (batch_size + 1) * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_hedge_offsets(d_hedge_offsets);

    // every multi-start is graded over the same hedges -> fixed-length segments for the reductions below
    thrust::transform(
        thrust_exec,
        thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(batch_size + 1),
        t_hedge_offsets,
        [num_hedges] __device__ (uint32_t start) { return start * num_hedges; }
    );

    // NOTE: reductions are segmented, and not batch-wide, so that a multi-start's score never depends on what it is batched with
    void *c_reduce_storage = nullptr;
    size_t c_reduce_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        c_reduce_storage, c_reduce_storage_bytes,
        d_result, d_src_dst_distance,
        batch_size, d_hedge_offsets, d_hedge_offsets + 1,
        stream
    ));
    CUDA_CHECK(cudaMallocAsync(&c_reduce_storage, c_reduce_storage_bytes, stream));
    
    // compute the max (or tot) src-dst manhattan distance per hedge
    {
        // launch configuration - max (or tot) src-dst distance kernel
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = batch_hedges ; // 1 warp per hedge
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - max (or tot) src-dst distance kernel
        //LAUNCH(cfg) TID(tid) RUN << "max src-dst distance kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        LAUNCH(cfg) TID(tid) RUN << "tot src-dst distance kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        //max_src_dst_distance_kernel<<<blocks, threads_per_block, 0, stream>>>(
        tot_src_dst_distance_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
            d_placement,
            d_hedges,
            d_hedges_offsets,
            d_srcs_count,
            d_hedge_weights,
            num_hedges,
            num_nodes,
            batch_size,
            d_result // <- alredy multiplied by hedge weight
        );
        DBG(cfg) CUDA_CHECK(cudaGetLastError());
        DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        c_reduce_storage, c_reduce_storage_bytes,
        d_result, d_src_dst_distance,
        batch_size, d_hedge_offsets, d_hedge_offsets + 1,
        stream
    ));

    // compute the Steiner tree span upper bound per hedge, given by the weighted spanning tree overs its pins' complete graph
    {
        // launch configuration - min spanning tree weight kernel
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = batch_hedges ; // 1 warp per hedge
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - min spanning tree weight kernel
        LAUNCH(cfg) TID(tid) RUN << "min spanning tree weight kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        min_spanning_tree_weight_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
            d_placement,
            d_hedges,
            d_hedges_offsets,
            d_hedge_weights,
            num_hedges,
            num_nodes,
            batch_size,
            d_result // <- alredy multiply by hedge weight
        );
        DBG(cfg) CUDA_CHECK(cudaGetLastError());
        DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        c_reduce_storage, c_reduce_storage_bytes,
        d_result, d_steiner_span,
        batch_size, d_hedge_offsets, d_hedge_offsets + 1,
        stream
    ));

    CUDA_CHECK(cudaFreeAsync(d_result, stream));
    CUDA_CHECK(cudaFreeAsync(d_hedge_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(c_reduce_storage, stream));
}


// LOGGING

template<Topology T>
void logForces(
    const float *d_forces,
    const uint32_t num_nodes,
    const cudaStream_t stream,
    const int tid
) {
    std::vector<float> forces_tmp(num_nodes * T::neighborsCount());
    CUDA_CHECK(cudaMemcpyAsync(forces_tmp.data(), d_forces, num_nodes * T::neighborsCount() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout TID(tid) << "Forces:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            std::cout TID(tid) << "  node " << i << " -> " << forces_tmp[T::neighborsCount() * i];
            for (uint32_t neigh_idx = 1; neigh_idx < T::neighborsCount(); neigh_idx++)
                std::cout << ", " << forces_tmp[T::neighborsCount() * i + neigh_idx];
            std::cout << "\n";
        }
    }
}

template<Topology T>
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
                else if (target >= UINT32_MAX - T::neighborsCount()) std::cout << " (" << j << " target=NEIGH(" << UINT32_MAX - target - 1 << ") score=" << score << ")";
                else std::cout << " (" << j << " target=" << target << " score=" << score << ")";
            }
            std::cout << "\n";
        }
    }
}

template<Topology T>
void logSwapPairs(
    const slot *d_swap_slots,
    const uint32_t num_nodes,
    const cudaStream_t stream,
    const int tid
) {
    std::vector<slot> slots_tmp(num_nodes);
    CUDA_CHECK(cudaMemcpyAsync(slots_tmp.data(), d_swap_slots, num_nodes * sizeof(slot), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout TID(tid) << "Swap pairs:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            slot node_slot = slots_tmp[i];
            if (node_slot.id == UINT32_MAX) std::cout TID(tid) << "  node " << i << " -> target=none score=" << node_slot.score << "\n";
            else if (node_slot.id >= UINT32_MAX - T::neighborsCount()) std::cout TID(tid) << "  node " << i << " -> target=NEIGH(" << UINT32_MAX - node_slot.id - 1 << ") score=" << node_slot.score << "\n";
            else std::cout TID(tid) << "  node " << i << " -> target=" << node_slot.id << " score=" << node_slot.score << "\n";
        }
    }
}

void logEvents(
    const swap *d_ev_swaps,
    const float *d_ev_scores,
    const uint32_t num_nodes,
    const std::string flare,
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

// TEMPLATE INSTANTIATIONS

#define INSTANTIATE_PLACEMENT_FUNCTIONS(T) \
    template void forceDirectedRefinement<T>( \
        const runconfig&, cudaDeviceProp, \
        const uint32_t*, const dim_t*, \
        const uint32_t*, const dim_t*, \
        const float*, uint32_t, uint32_t, uint32_t, \
        Coord_t<T>*, uint32_t*, \
        cudaStream_t, int); \
 \
    template void getLocalityMetrics<T>( \
        const runconfig&, const Coord_t<T>*, \
        const uint32_t*, const dim_t*, \
        const uint32_t*, const float*, \
        uint32_t, uint32_t, uint32_t, float*, float*, \
        cudaStream_t, int);

INSTANTIATE_PLACEMENT_FUNCTIONS(Lattice2D)
INSTANTIATE_PLACEMENT_FUNCTIONS(Torus6D)
INSTANTIATE_PLACEMENT_FUNCTIONS(ArbitraryGraph)

#undef INSTANTIATE_PLACEMENT_FUNCTIONS