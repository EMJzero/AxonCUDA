#include <tuple>

#include "thruster.cuh"

#include "runconfig.hpp"

#include "refinement.cuh"

#include "utils.cuh"
#include "defines.cuh"
#include "chaining.cuh"

using namespace config;

void refinementRepeats(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_srcs_count,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t *d_inbound_count,
    const float *d_hedge_weights,
    const uint32_t *d_nodes_sizes,
    const uint32_t level_idx,
    const uint32_t curr_num_nodes,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    const dim_t touching_size,
    uint32_t *d_pairs,
    float *d_f_scores,
    uint32_t *d_partitions,
    uint32_t *d_partitions_sizes,
    uint32_t *d_pins_per_partitions,
    uint32_t *d_partitions_inbound_sizes
) {
    // settings for refinement
    bool chainup = false; // true -> chain moves by size, then sort chains into a sequence, false -> directly sort moves into a sequence by gain
    bool encourage = cfg.mode == Mode::INCC; // true -> give a gain to moves that don't fully disconnect an hedge, doing so proportionally to how few pins the leave behind

    for (uint32_t fm_repeat = 0u; fm_repeat < cfg.refine_repeats; fm_repeat++) {
        std::cout << "Refining level " << level_idx << " repeat " << fm_repeat << ", remaining nodes=" << curr_num_nodes << " number of partitions=" << num_partitions << "\n";

        // initially relax partition size, then rely on later refinement to bring it back [TERRIBLE MISTAKE!]
        //const uint32_t h_varying_max_nodes_per_part = (int32_t)((float)h_max_nodes_per_part * (1.4f - 0.4f * (float)std::min(2*fm_repeat, cfg.refine_repeats) / cfg.refine_repeats));
        //CUDA_CHECK(cudaMemcpyToSymbol(max_nodes_per_part, &h_varying_max_nodes_per_part, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

        // by how much of a node's size to allow an invalid move to be proposed (but filtered later by events - if still invalid)
        uint32_t discount = fm_repeat < cfg.refine_repeats / 3 ? 1u : (fm_repeat < 2 * cfg.refine_repeats / 3 ? 2u : UINT32_MAX);

        // prepare this level's pins per partition
        const size_t pins_per_partitions_bytes = static_cast<size_t>(num_hedges) * num_partitions * sizeof(uint32_t);
        CUDA_CHECK(cudaMemset(d_pins_per_partitions, 0x00, pins_per_partitions_bytes));
        // while computing pins per partition also compute the distinct inbound counts per partition (number of pins with a count > 0)
        CUDA_CHECK(cudaMemset(d_partitions_inbound_sizes, 0x00, num_partitions * sizeof(uint32_t)));
        
        // TODO: if you are struggling for memory, store both pins per partition in a compact form
        // TODO: compute both pins per partition (inbound only and not) via a highly optimized map+histogram pattern
        //       => map from edges of nodes to hedges of partitions, then histogram for each hedge in // (by key)

        {
            // launch configuration - pins per partition kernel
            // TODO: could update this in-place instead of recomputing it each time by going over 'touching' for moved nodes when applying the refinement!
            // HARDER: if we change pins per partition to represent only destination (inbound) pins after we computed gains, also need to revert it to represent all pins...
            // => If we do this, uncomment "pins_per_partitions" in "fm_refinement_apply_kernel"
            // TODO: maybe it would be faster to build pins per partition with 'touching', by going one block per partition, 256 threads digesting touching hedge with
            //       an hash-map in shared memory, then dumped to global with one streak of atomics?
            // => call something like "apply_coarsening_touching_count" at the innermost level using partitions as groups to compute the initial pins per partition?
            int threads_per_block = 256;
            int num_threads_needed = num_hedges; // 1 thread per hedge
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - pins per partition kernel
            std::cout << "Running pins per partition kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            // NOTE: having this available during FM refinement makes its complexity linear in the connectivity, instead of quadratic!
            pins_per_partition_kernel<<<blocks, threads_per_block>>>(
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
        }
        
        // zero-out fm-ref gains kernel's outputs
        CUDA_CHECK(cudaMemset(d_pairs, 0xFF, curr_num_nodes * sizeof(uint32_t))); // 0xFF -> UINT32_MAX
        // NOTE: no need to init. "d_f_scores" if we use "d_pairs" to see which locations are valid

        {
            // launch configuration - fm-ref gains kernel
            // NOTE: choose threads_per_block multiple of WARP_SIZE
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = curr_num_nodes; // 1 warp per node
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - fm-ref gains kernel
            std::cout << "Running fm-ref gains kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            fm_refinement_gains_kernel<<<blocks, threads_per_block>>>(
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
                fm_repeat,
                discount,
                encourage, // encourage all moves only when not doing k-way partitioning
                // NOTE: repurposing those from the candidates kernel!
                d_pairs, // -> moves: pairs[node] -> partition the node wants to join
                d_f_scores
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // =============================
        // print some temporary results
        #if VERBOSE
        logMoves(
            d_pairs,
            d_f_scores,
            d_partitions,
            curr_num_nodes
        );
        #endif
        // =============================
        
        uint32_t *d_ranks = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ranks, curr_num_nodes * sizeof(uint32_t))); // node -> number of touching hedges seen as of now
        thrust::device_ptr<uint32_t> t_ranks(d_ranks);
        thrust::device_ptr<float> t_scores(d_f_scores);

        // alternate between sequence ordering techniques
        if (chainup) {
            // sort scores and build an array of ranks (node id -> his move's idx in sorted scores)
            thrust::device_vector<uint32_t> t_indices(curr_num_nodes);
            thrust::sequence(t_indices.begin(), t_indices.end());
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
        } else {
            // build move-chains to approximate high-gain swaps, then sort by chain total gain
            chaining(
                d_partitions,
                d_pairs,
                d_nodes_sizes,
                d_f_scores,
                curr_num_nodes,
                d_ranks
            );
        }

        {
            // launch configuration - fm-ref cascade kernel
            // NOTE: choose threads_per_block multiple of WARP_SIZE
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = curr_num_nodes ; // 1 warp per node
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - fm-ref cascade kernel
            std::cout << "Running fm-ref cascade kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            fm_refinement_cascade_kernel<<<blocks, threads_per_block>>>(
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
                encourage,
                d_f_scores
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

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
        {
            // launch configuration - build size events kernel
            int threads_per_block = 128;
            int num_threads_needed = curr_num_nodes; // 1 thread per move
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
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
        }

        // sort events by (partition, rank) [in lexicographical order for the tuple] and carry size_events_delta along
        auto size_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_size_events_partition, t_size_events_index));
        auto size_events_key_end = size_events_key_begin + num_size_events;
        thrust::sort_by_key(size_events_key_begin, size_events_key_end, t_size_events_delta);
        // inclusive scan inside each key (= partition) on the event deltas => for each event we get the cumulative size delta for that partition at that point in the sequence
        thrust::inclusive_scan_by_key(t_size_events_partition, t_size_events_partition + num_size_events, t_size_events_delta, t_size_events_delta);
        // now mark moves that would violate size constraint if the sequence were to end on them
        int32_t *d_valid_moves = nullptr;
        CUDA_CHECK(cudaMalloc(&d_valid_moves, curr_num_nodes * sizeof(int32_t))); // valid_move[rank idx] -> 1 if applying all moves up to the idx one in the ordered sequence gives a valid state
        CUDA_CHECK(cudaMemset(d_valid_moves, 0, curr_num_nodes * sizeof(int32_t)));
        thrust::device_ptr<int32_t> t_valid_moves(d_valid_moves);
        
        {
            // launch configuration - flag size events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_size_events; // 1 thread per event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
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
        }
        CUDA_CHECK(cudaFree(d_size_events_partition));
        CUDA_CHECK(cudaFree(d_size_events_index));
        CUDA_CHECK(cudaFree(d_size_events_delta));
        // compute, as of each event, the cumulative number of partitions that are invalid by summing the count of those made/unmade invalid at each event
        thrust::inclusive_scan(t_valid_moves, t_valid_moves + curr_num_nodes, t_valid_moves);
        
        // ======================================
        // preparatory step: update pins per partition into inbound (only) pins partition
        // simultaneously, also correct the calculation for partitions_inbound_sizes by removing outbounds
        {
            // launch configuration - inbound pins per partition kernel
            int threads_per_block = 256;
            int num_threads_needed = num_hedges; // 1 thread per hedge
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - inbound pins per partition kernel
            std::cout << "Running inbound pins per partition kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            // NOTE: inbound-only version of the above used for constraints checks...
            inbound_pins_per_partition_kernel<<<blocks, threads_per_block>>>(
                d_hedges,
                d_hedges_offsets,
                d_srcs_count,
                d_partitions,
                num_hedges,
                num_partitions,
                d_pins_per_partitions, // from now it represents inbound sets only
                d_partitions_inbound_sizes
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        // ======================================
        // extra step: compute moves validity by inbound set cardinality (same HP as the kernel above: all previous higher-gain moves will be applied)
        // explode each move into two events for every inbound hedge of the moved node, one decrementing and one incrementing the hedge's
        // occurrencies in the src partition's inbound set and dst partition's inbound set respectively
        // => results in n*h events (better than the n*h*p volume of conditions/counters we need to check)
        uint32_t *d_inbound_count_events_partition = nullptr;
        uint32_t *d_inbound_count_events_index = nullptr;
        uint32_t *d_inbound_count_events_hedge = nullptr;
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
        {
            // launch configuration - build hedge events kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = curr_num_nodes ; // 1 warp per move
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
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
        }
        
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
        {
            // launch configuration - count inbound events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_inbound_count_events; // 1 thread per hedge event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
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
        }

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

        {
            // launch configuration - build inbound events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_inbound_count_events; // 1 thread per hedge event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
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
        }
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
        int32_t *d_inbound_valid_moves = nullptr;
        CUDA_CHECK(cudaMalloc(&d_inbound_valid_moves, curr_num_nodes * sizeof(int32_t))); // valid_move[rank idx] -> 1 if applying all moves up to the idx one in the ordered sequence gives a valid state
        CUDA_CHECK(cudaMemset(d_inbound_valid_moves, 0u, curr_num_nodes * sizeof(int32_t)));
        thrust::device_ptr<int32_t> t_inbound_valid_moves(d_inbound_valid_moves);
        {
            // launch configuration - flag inbound events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_inbound_size_events; // 1 thread per event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
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
        }
        CUDA_CHECK(cudaFree(d_inbound_size_events_partition));
        CUDA_CHECK(cudaFree(d_inbound_size_events_index));
        CUDA_CHECK(cudaFree(d_inbound_size_events_delta));
        
        // compute, as of each event, the cumulative number of partitions that are invalid by summing the count of those made/unmade invalid at each event
        thrust::inclusive_scan(t_inbound_valid_moves, t_inbound_valid_moves + curr_num_nodes, t_inbound_valid_moves);
        // ======================================
        // find the move in the sequence that yields both the highest gain and a valid state (when all moves before it are applied)
        // index space 0..curr_num_nodes - 1
        auto idx_begin = thrust::make_counting_iterator<uint32_t>(0);
        // functor comparing sequence entries, skipping invalid ones by inbound size (only 0 allowed), prioritizing size events (zero or negative), and then picking the highest score
        best_move_functor best_scores { thrust::raw_pointer_cast(t_scores), thrust::raw_pointer_cast(t_valid_moves), thrust::raw_pointer_cast(t_inbound_valid_moves) };
        // max over valid endpoints only, find the point in the sequence of moves where applying them further never nets a higher gain in a valid state
        auto best_iterator_entry = thrust::max_element(idx_begin, idx_begin + curr_num_nodes, best_scores);
        const uint32_t best_rank = *best_iterator_entry;
        const uint32_t num_good_moves = best_rank + 1; // "+1" to make this the improving moves count, rather than the last improving move's idx
        // validity double-check
        int32_t size_validity, inbounds_validity;
        float acquired_gain;
        CUDA_CHECK(cudaMemcpy(&size_validity, d_valid_moves + best_rank, sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&inbounds_validity, d_inbound_valid_moves + best_rank, sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&acquired_gain, d_f_scores + best_rank, sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Best fm-ref move:\n  Move rank: " << best_rank << ", Acquired gain: " << acquired_gain << "\n";
        std::cout << "  Size constraint violations variation amount (in nodes above the limit): " << size_validity << ", Inbound constraint violations variation (in invalid partitions): " << inbounds_validity << "\n";
        if (size_validity <= 0 && inbounds_validity <= 0 && acquired_gain >= 0) {
            // launch configuration - fm-ref apply kernel
            int threads_per_block = 128;
            int num_threads_needed = curr_num_nodes; // 1 thread per move to apply
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
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
            //if (acquired_gain < 0) std::cerr << "WARNING: applied a refinement move with negative gain on level " << level_idx << " !!\n";
        } else {
            std::cout << "No valid refinement move found on level " << level_idx << " (reason: " << (size_validity > 0 ? (inbounds_validity > 0 ? "both size and inbounds validities" : "size validity") : (inbounds_validity > 0 ? "inbounds validity" : "negative gain")) << ") ...\n";
            if (size_validity > 0 && !chainup) chainup = true; // enable chaining when no moves are available via greedy sorting because of size constraints
            else if (fm_repeat < cfg.refine_repeats / 3) fm_repeat = cfg.refine_repeats / 2;
            else if (fm_repeat < 2 * cfg.refine_repeats / 3) fm_repeat = 2 * cfg.refine_repeats / 3;
            else fm_repeat = cfg.refine_repeats; // aka break!
        }
        CUDA_CHECK(cudaFree(d_ranks));
        CUDA_CHECK(cudaFree(d_valid_moves));
        CUDA_CHECK(cudaFree(d_inbound_valid_moves));
    }
}

void logPartitions(
    const uint32_t *d_partitions,
    const uint32_t *d_partitions_sizes,
    const uint32_t curr_num_nodes,
    const uint32_t num_partitions,
    const uint32_t h_max_nodes_per_part
) {
    std::vector<uint32_t> partitions_tmp(curr_num_nodes);
    CUDA_CHECK(cudaMemcpy(partitions_tmp.data(), d_partitions, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> partitions_sizes_tmp(num_partitions);
    CUDA_CHECK(cudaMemcpy(partitions_sizes_tmp.data(), d_partitions_sizes, num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::unordered_map<uint32_t, int> part_count;
    std::cout << "Partitioning results:\n";
    for (uint32_t i = 0; i < curr_num_nodes; ++i) {
        uint32_t part = partitions_tmp[i];
        part_count[part]++;
        if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH)) {
            if (part == UINT32_MAX) std::cout << "node " << i << " -> part=none";
            else std::cout << "  node " << i << " -> " << part;
            std::cout << ((i + 1) % 4 == 0 ? "\n" : "\t");
        }
    }
    for (uint32_t i = 0; i < num_partitions; ++i) {
        uint32_t part_size = partitions_sizes_tmp[i];
        if (part_size > h_max_nodes_per_part)
            std::cerr << "  WARNING, max partition size constraint (" << h_max_nodes_per_part << ") violated by part=" << i << " with part_size=" << part_size << " !!\n";
    }
    int max_ps = part_count.empty() ? 0 : std::max_element(part_count.begin(), part_count.end(), [](auto &a, auto &b){ return a.second < b.second; })->second;
    std::cout << "Non-empty partitions count: " << part_count.size() << ", Max partition size: " << max_ps << "\n";
}

void logMoves(
    const uint32_t *d_pairs,
    const float *d_f_scores,
    const uint32_t *d_partitions,
    const uint32_t curr_num_nodes
) {
    std::vector<uint32_t> moves_tmp(curr_num_nodes);
    std::vector<float> gains_tmp(curr_num_nodes);
    std::vector<uint32_t> src_partitions_tmp(curr_num_nodes);
    CUDA_CHECK(cudaMemcpy(moves_tmp.data(), d_pairs, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gains_tmp.data(), d_f_scores, curr_num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(src_partitions_tmp.data(), d_partitions, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Proposed moves:\n";
    for (uint32_t i = 0; i < curr_num_nodes; ++i) {
        if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH)) {
            std::cout << "  node " << i << " : ";
            uint32_t move = moves_tmp[i];
            if (move == UINT32_MAX) std::cout << "stay " << src_partitions_tmp[i];
            else std::cout << src_partitions_tmp[i] << " -> " << move;
            std::cout << " gain=" << std::fixed << std::setprecision(3) << gains_tmp[i];
            std::cout << ((i + 1) % 2 == 0 ? "\n" : "\t");
        }
    }
}
