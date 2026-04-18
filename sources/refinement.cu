#include <bit>
#include <tuple>

#include "thruster.cuh"

#include <cub/cub.cuh>

#include "runconfig.hpp"

#include "refinement.cuh"

#include "utils.cuh"
#include "defines.cuh"
#include "chaining.cuh"

using namespace config;

void refinementRepeats(
    const runconfig &cfg,
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
    const bool update_final_inbound_counts,
    uint32_t *d_pairs,
    float *d_f_scores,
    uint32_t *d_partitions,
    uint32_t *d_partitions_sizes,
    uint32_t *d_partitions_inbound_sizes
) {
    // prepare this level's pins per partition
    // NOTE: the inbound counters per partition are just the transposed of pins per partition! No need to compute them separately!
    uint32_t *d_pins_per_partitions = nullptr; // matrix<num_hedges x num_partitions>, pins_per_partitions[hedge idx * num_partitions + partition idx] -> count of pins of "hedge" in that "partition"
    const size_t pins_per_partitions_bytes = static_cast<size_t>(num_hedges) * num_partitions * sizeof(uint32_t);
    // |
    // if we are short on memory, go for the sparse pins-per-partition representation
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    // NOTE: "14" is the number of mallocs in this routine that COULD allocate up to "curr_num_nodes" entries
    if (free_bytes < pins_per_partitions_bytes + (6 * curr_num_nodes + 8 * 2 * touching_size) * sizeof(uint32_t)) {
        INFO(cfg) std::cout << "Not enough memory to allocate the dense pins per partition matrix: switching to the sparse version\n";
        return refinementSparseRepeats(
            cfg,
            d_hedges,
            d_hedges_offsets,
            d_srcs_count,
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            d_hedge_weights,
            d_nodes_sizes,
            level_idx,
            curr_num_nodes,
            num_hedges,
            num_partitions,
            touching_size,
            update_final_inbound_counts,
            d_pairs,
            d_f_scores,
            d_partitions,
            d_partitions_sizes,
            d_partitions_inbound_sizes
        );
    }
    // |
    // allocate pins per partition
    CUDA_CHECK(cudaMalloc(&d_pins_per_partitions, pins_per_partitions_bytes));

    // settings for refinement
    bool chainup = false; // true -> chain moves by size, then sort chains into a sequence, false -> directly sort moves into a sequence by gain
    bool encourage = cfg.mode == Mode::INCC; // true -> give a gain to moves that don't fully disconnect an hedge, doing so proportionally to how few pins the leave behind

    for (uint32_t fm_repeat = 0u; fm_repeat < cfg.refine_repeats; fm_repeat++) {
        INFO(cfg) std::cout << "Refining level " << level_idx << " repeat " << fm_repeat << ", remaining nodes=" << curr_num_nodes << " number of partitions=" << num_partitions << "\n";

        // by how much of a node's size to allow an invalid move to be proposed (but filtered later by events - if still invalid)
        uint32_t discount = fm_repeat < cfg.refine_repeats / 3 ? 1u : (fm_repeat < 2 * cfg.refine_repeats / 3 ? 2u : UINT32_MAX);

        // compute all pins per partition entries
        CUDA_CHECK(cudaMemset(d_pins_per_partitions, 0x00, pins_per_partitions_bytes));
        // while computing pins per partition also compute the distinct incident counts per partition (number of pins with a count > 0)
        CUDA_CHECK(cudaMemset(d_partitions_inbound_sizes, 0x00, num_partitions * sizeof(uint32_t)));
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
            LAUNCH(cfg) RUN << "pins per partition kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
            LAUNCH(cfg) RUN << "fm-ref gains kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
        LOG(cfg) {
            logMoves(
                d_pairs,
                d_f_scores,
                d_partitions,
                curr_num_nodes
            );
        }
        // =============================
        
        uint32_t *d_ranks = nullptr; // rank[node idx] -> position of node's move in the sorted sequence
        CUDA_CHECK(cudaMalloc(&d_ranks, curr_num_nodes * sizeof(uint32_t)));
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
                cfg,
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
            LAUNCH(cfg) RUN << "fm-ref cascade kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
        uint32_t *d_size_events_partition = nullptr; // size_events_partition[ev] -> partition affected by the event
        uint32_t *d_size_events_index = nullptr; // size_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        int32_t *d_size_events_delta = nullptr; // size_events_delta[ev] -> size variation brought by the event
        const uint32_t num_size_events = 2 * curr_num_nodes;
        CUDA_CHECK(cudaMalloc(&d_size_events_partition, num_size_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_size_events_index, num_size_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_size_events_delta, num_size_events * sizeof(int32_t)));
        thrust::device_ptr<uint32_t> t_size_events_partition(d_size_events_partition);
        thrust::device_ptr<uint32_t> t_size_events_index(d_size_events_index);
        thrust::device_ptr<int32_t> t_size_events_delta(d_size_events_delta);
        {
            // launch configuration - build size events kernel
            int threads_per_block = 128;
            int num_threads_needed = curr_num_nodes; // 1 thread per move
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - build size events kernel
            LAUNCH(cfg) RUN << "build size events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
        CUDA_CHECK(cudaMemset(d_valid_moves, 0x00, curr_num_nodes * sizeof(int32_t)));
        thrust::device_ptr<int32_t> t_valid_moves(d_valid_moves);
        
        {
            // launch configuration - flag size events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_size_events; // 1 thread per event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - flag size events kernel
            LAUNCH(cfg) RUN << "flag size events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
            LAUNCH(cfg) RUN << "inbound pins per partition kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
        uint32_t *d_inbound_count_events_partition = nullptr; // inbound_count_events_partition[ev] -> partition affected by the event
        uint32_t *d_inbound_count_events_index = nullptr; // inbound_count_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        uint32_t *d_inbound_count_events_hedge = nullptr; // d_inbound_count_events_hedge[ev] -> hedge involved in the event
        int32_t *d_inbound_count_events_delta = nullptr; // inbound_count_events_delta[ev] -> inbound_count variation brought by the event
        const uint32_t num_inbound_count_events = 2 * touching_size; // TODO: this is slightly larger than truly needed, because we just use inbound hedges...
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_partition, num_inbound_count_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_index, num_inbound_count_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_hedge, num_inbound_count_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_delta, num_inbound_count_events * sizeof(int32_t)));
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
            LAUNCH(cfg) RUN << "build hedge events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
        dim_t *d_inbound_size_events_offsets = nullptr; // inbound_size_events_offsets[event idx] -> initially a flag of whether each event will produce an increase/decrese in inbound counts, after the scan it becomes the offset of each new event
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_offsets, (num_inbound_count_events + 1) * sizeof(dim_t)));
        CUDA_CHECK(cudaMemset(d_inbound_size_events_offsets, 0x00, (num_inbound_count_events + 1) * sizeof(dim_t)));
        {
            // launch configuration - count inbound events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_inbound_count_events; // 1 thread per hedge event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - count inbound events kernel
            LAUNCH(cfg) RUN << "count inbound events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
        thrust::device_ptr<dim_t> t_inbound_size_events_offsets(d_inbound_size_events_offsets);
        thrust::inclusive_scan(t_inbound_size_events_offsets, t_inbound_size_events_offsets + num_inbound_count_events + 1, t_inbound_size_events_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
        dim_t num_inbound_size_events = 0; // last value in the inclusive scan = full reduce = total number of touching hedges among all sets
        CUDA_CHECK(cudaMemcpy(&num_inbound_size_events, d_inbound_size_events_offsets + num_inbound_count_events, sizeof(dim_t), cudaMemcpyDeviceToHost));
        uint32_t *d_inbound_size_events_partition = nullptr; // inbound_size_events_partition[ev] -> partition affected by the event
        uint32_t *d_inbound_size_events_index = nullptr; // inbound_size_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        int32_t *d_inbound_size_events_delta = nullptr; // inbound_size_events_delta[ev] -> inbound set size variation brought by the event
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_partition, num_inbound_size_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_index, num_inbound_size_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_delta, num_inbound_size_events * sizeof(int32_t)));
        thrust::device_ptr<uint32_t> t_inbound_size_events_partition(d_inbound_size_events_partition);
        thrust::device_ptr<uint32_t> t_inbound_size_events_index(d_inbound_size_events_index);
        thrust::device_ptr<int32_t> t_inbound_size_events_delta(d_inbound_size_events_delta);

        {
            // launch configuration - build inbound events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_inbound_count_events; // 1 thread per hedge event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - build inbound events kernel
            LAUNCH(cfg) RUN << "build inbound events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
        CUDA_CHECK(cudaMemset(d_inbound_valid_moves, 0x00, curr_num_nodes * sizeof(int32_t)));
        thrust::device_ptr<int32_t> t_inbound_valid_moves(d_inbound_valid_moves);
        if(num_inbound_size_events > 0) {
            // launch configuration - flag inbound events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_inbound_size_events; // 1 thread per event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - flag inbound events kernel
            LAUNCH(cfg) RUN << "flag inbound events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
        INFO(cfg) std::cout << "Best fm-ref move:\n  Move rank: " << best_rank << ", Acquired gain: " << acquired_gain << "\n";
        if (size_validity <= 0 && inbounds_validity <= 0 && acquired_gain >= 0) {
            // launch configuration - fm-ref apply kernel
            int threads_per_block = 128;
            int num_threads_needed = curr_num_nodes; // 1 thread per move to apply
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - fm-ref apply kernel
            LAUNCH(cfg) RUN << "fm-ref apply (" << num_good_moves << " good moves) kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
            INFO(cfg) {
                std::cout << "No valid refinement move found on level " << level_idx << " - reason: "
                    << (size_validity > 0 ? (inbounds_validity > 0 ? "both size and inbounds validities" : "size validity") : (inbounds_validity > 0 ? "inbounds validity" : "negative gain")) << "\n";
                if (size_validity > 0) std::cout << "  Size constraint violations variation amount (in nodes above the limit): " << size_validity << "\n";
                if (inbounds_validity > 0) std::cout << "  Inbound constraint violations variation (in invalid partitions): " << inbounds_validity << "\n";
            }
            if (size_validity > 0 && !chainup) chainup = true; // enable chaining when no moves are available via greedy sorting because of size constraints
            else if (fm_repeat < cfg.refine_repeats / 3) fm_repeat = cfg.refine_repeats / 2;
            else if (fm_repeat < 2 * cfg.refine_repeats / 3) fm_repeat = 2 * cfg.refine_repeats / 3;
            else fm_repeat = cfg.refine_repeats; // aka break!
        }
        CUDA_CHECK(cudaFree(d_ranks));
        CUDA_CHECK(cudaFree(d_valid_moves));
        CUDA_CHECK(cudaFree(d_inbound_valid_moves));
    }

    // recompute inbound set sizes
    if (update_final_inbound_counts) {
        uint32_t* pp_map = nullptr; // pp_map[(e * num_partitions + p)/32] -> the bit is set if "e" was already seen incident to "p"
        const dim_t pp_per_hedge = (static_cast<dim_t>(num_partitions) + 31u) / 32u;
        const dim_t pp_map_size = static_cast<dim_t>(num_hedges) * pp_per_hedge;
        CUDA_CHECK(cudaMalloc(&pp_map, pp_map_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(pp_map, 0x00, pp_map_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_partitions_inbound_sizes, 0x00, num_partitions * sizeof(uint32_t)));
        {
            // launch configuration - inbound sets size kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_hedges; // 1 warp per hedge
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - inbound sets size kernel
            LAUNCH(cfg) RUN << "inbound sets size kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            inbound_sets_size_kernel<<<blocks, threads_per_block>>>(
                d_hedges,
                d_hedges_offsets,
                d_srcs_count,
                d_partitions,
                num_hedges,
                num_partitions,
                pp_map,
                d_partitions_inbound_sizes
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaFree(pp_map));
    }

    CUDA_CHECK(cudaFree(d_pins_per_partitions));
}


// SPARSE PINS-PER-PARTITION VARIANT

void refinementSparseRepeats(
    const runconfig &cfg,
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
    const bool update_final_inbound_counts,
    uint32_t *d_pairs,
    float *d_f_scores,
    uint32_t *d_partitions,
    uint32_t *d_partitions_sizes,
    uint32_t *d_partitions_inbound_sizes
) {
    /*
    * IDEA for the pins-per-partition sparse data structure:
    * - define a struct (uint64_t count, uint64_t flag) -> bitmap
    * - allocate a matrix of R x (C / 64) entries
    * - iterate hedges and write, for each row, for each bitmap:
    *     - a 1 or a 0 in the bit of the flag corresponding to the column index % 64
    *     - simultaneously increment by 1 the counter in the entry in column "column / 64"
    * - then do an exclusive scan of each bitmap, over only the counters
    * - hence, each counter tell essentially how many bits were set to 1 before its bitmap
    *     - that is, the offset where each bitmap's actual data will commence in the segmented array
    * - use the full total to allocate a compressed array of segments, one segment per bitmap
    * - re-do the iteration over other hedges to fill the new array of segments
    * - to access a cell in the array, access first the matrix, and read the corresponding bitmap (one every 64 actual data entries),
    *     - count how many bits are set to 1 before your bit among the flags
    *     - add the counter to that count, and that is the offset to read in the segmented array
    * 
    * UPGRADE for inbound counts:
    * => allocate TWO compressed arrays of segments, indexed by the SAME offsets
    * => the first array, same as above, counts every pin
    * => the second array counts only inbound pins, that will be less,
    */

    // sparse pins-per-partition data structure
    bitmap* d_ppp_offsets = nullptr; // ppp_offsets[hedge-idx * ceil(num_partitions / 64) + part-idx / 64] -> bitmap to access the pin count for all (hedge, part / 64), ... (hedge, part / 64 + 63) pairs
    uint32_t* d_ppp = nullptr; // ppp[ppp_offsets[e*num_partitions+p/64].cnt + bits-at-one-before-the(p%64)th-in(ppp_offsets[e*num_partitions+p/64].flg)] -> pins count held by hedge e in partition p

    const uint32_t ppp_per_hedge = (num_partitions + BITMAP_CAPACITY - 1) / BITMAP_CAPACITY; // aka: ceil(num_partitions / 64)
    dim_t ppp_offsets_size = num_hedges * ppp_per_hedge;
    if (ppp_offsets_size * sizeof(bitmap) > (1ull << 32))
        INFO(cfg) std::cout
            << "Allocating " << std::fixed << std::setprecision(1) << (float)(ppp_offsets_size * sizeof(bitmap)) / (1 << 30)
            << " GB for pins-per-partition bitmaps ...\n";
    CUDA_CHECK(cudaMalloc(&d_ppp_offsets, ppp_offsets_size * sizeof(bitmap)));

    // size of the last allocation for the ppp segmented array
    // => re-allocate IFF a larger one is needed
    dim_t ppp_size = 0u;

    // settings for refinement
    bool chainup = false; // true -> chain moves by size, then sort chains into a sequence, false -> directly sort moves into a sequence by gain
    bool encourage = cfg.mode == Mode::INCC; // true -> give a gain to moves that don't fully disconnect an hedge, doing so proportionally to how few pins the leave behind

    for (uint32_t fm_repeat = 0u; fm_repeat < cfg.refine_repeats; fm_repeat++) {
        INFO(cfg) std::cout << "Refining level " << level_idx << " repeat " << fm_repeat << ", remaining nodes=" << curr_num_nodes << " number of partitions=" << num_partitions << " (mode: sparse-ppp)\n";

        // by how much of a node's size to allow an invalid move to be proposed (but filtered later by events - if still invalid)
        uint32_t discount = fm_repeat < cfg.refine_repeats / 3 ? 1u : (fm_repeat < 2 * cfg.refine_repeats / 3 ? 2u : UINT32_MAX);

        // build of offsets bitmaps
        CUDA_CHECK(cudaMemset(d_ppp_offsets, 0x00, ppp_offsets_size * sizeof(bitmap)));
        {
            // launch configuration - sparse pins per partition count kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_hedges; // 1 warp per hedge
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - sparse pins per partition count kernel
            LAUNCH(cfg) RUN << "sparse pins per partition count kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            // NOTE: having this available during FM refinement makes its complexity linear in the connectivity, instead of quadratic!
            sparse_pins_per_partition_count_kernel<<<blocks, threads_per_block>>>(
                d_hedges,
                d_hedges_offsets,
                d_partitions,
                num_hedges,
                ppp_per_hedge,
                d_ppp_offsets
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // exclusive scan of each bitmap counter
        uint64_t* raw_offsets = reinterpret_cast<uint64_t*>(d_ppp_offsets);
        /* -> better version, if "cuda/iterator" can be included!
        // cnt fields are one-every-two, at raw_offsets[0], raw_offsets[2], raw_offsets[4], ...
        auto it_first_offset = cuda::strided_iterator<uint64_t*, int>(raw_offsets, 2);
        auto it_last_offset = t_first_offset + ppp_offsets_size;
        thrust::exclusive_scan(thrust::device, it_first_offset, it_last_offset, it_first_offset);
        */
        auto cnt_first = thrust::make_permutation_iterator(
            raw_offsets, thrust::make_transform_iterator(
                thrust::make_counting_iterator<uint64_t>(0),
                [] __host__ __device__ (uint64_t i) { return i << 1; }
            )
        );
        thrust::exclusive_scan(thrust::device, cnt_first, cnt_first + ppp_offsets_size, cnt_first);

        // compute non-zero entries count
        bitmap last_bitmap; // last entry in the exclusive scan -> add to its cnt the number of bits set to 1 in flg to have the non-zero entries count
        CUDA_CHECK(cudaMemcpy(&last_bitmap, d_ppp_offsets + ppp_offsets_size - 1, sizeof(bitmap), cudaMemcpyDeviceToHost));
        const dim_t new_ppp_size = last_bitmap.cnt + std::popcount(last_bitmap.flg);
        if (new_ppp_size > ppp_size) {
            if (ppp_size * sizeof(uint32_t) > (1ull << 32))
                INFO(cfg) std::cout
                    << "Allocating " << std::fixed << std::setprecision(1) << (float)(ppp_size * sizeof(uint32_t)) / (1 << 30)
                    << " GB for sparse pins-per-partition ...\n";
            if (d_ppp != nullptr) CUDA_CHECK(cudaFree(d_ppp));
            CUDA_CHECK(cudaMalloc(&d_ppp, new_ppp_size * sizeof(uint32_t)));
            ppp_size = new_ppp_size;
        }

        // fill in the segmented array
        CUDA_CHECK(cudaMemset(d_ppp, 0x00, ppp_size * sizeof(uint32_t)));
        // compute incident hedges counts per partition while you are at it
        CUDA_CHECK(cudaMemset(d_partitions_inbound_sizes, 0x00, num_partitions * sizeof(uint32_t)));
        {
            // launch configuration - sparse pins per partition write kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_hedges; // 1 warp per hedge
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - sparse pins per partition write kernel
            LAUNCH(cfg) RUN << "sparse pins per partition write kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            // NOTE: having this available during FM refinement makes its complexity linear in the connectivity, instead of quadratic!
            sparse_pins_per_partition_write_kernel<<<blocks, threads_per_block>>>(
                d_hedges,
                d_hedges_offsets,
                d_partitions,
                d_ppp_offsets,
                num_hedges,
                ppp_per_hedge,
                d_ppp,
                d_partitions_inbound_sizes // NOTE: here filled with outbounds too
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // zero-out fm-ref gains kernel's outputs
        CUDA_CHECK(cudaMemset(d_pairs, 0xFF, curr_num_nodes * sizeof(uint32_t))); // 0xFF -> UINT32_MAX
        // NOTE: no need to init. "d_f_scores" if we use "d_pairs" to see which locations are valid

        {
            // launch configuration - fm-ref gains sparse kernel
            // NOTE: choose threads_per_block multiple of WARP_SIZE
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = curr_num_nodes; // 1 warp per node
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - fm-ref gains sparse kernel
            LAUNCH(cfg) RUN << "fm-ref gains sparse kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            fm_refinement_gains_sparse_ppp_kernel<<<blocks, threads_per_block>>>(
                d_touching,
                d_touching_offsets,
                d_hedge_weights,
                d_partitions,
                d_ppp_offsets,
                d_ppp,
                d_nodes_sizes,
                d_partitions_sizes,
                num_hedges,
                curr_num_nodes,
                num_partitions,
                ppp_per_hedge,
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
        LOG(cfg) {
            logMoves(
                d_pairs,
                d_f_scores,
                d_partitions,
                curr_num_nodes
            );
        }
        // =============================
        
        uint32_t *d_ranks = nullptr; // rank[node idx] -> position of node's move in the sorted sequence
        CUDA_CHECK(cudaMalloc(&d_ranks, curr_num_nodes * sizeof(uint32_t)));
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
                cfg,
                d_partitions,
                d_pairs,
                d_nodes_sizes,
                d_f_scores,
                curr_num_nodes,
                d_ranks
            );
        }

        // ======================================
        // low memory emergency measure: drop a certain fraction of moves
        size_t free_bytes, total_bytes;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        const size_t upper_bound = 2 * 9 * touching_size * sizeof(uint32_t) + 2 * (curr_num_nodes + 1) * sizeof(dim_t);
        if (free_bytes < upper_bound) {
            // set to UINT32_MAX (invalid) the last faction of moves by rank
            const uint32_t discard_offset = curr_num_nodes * ((float)free_bytes / upper_bound); // tailing fraction of moves by rank to discard
            ERR(cfg) std::cout << "Emergency discard of " << discard_offset << " moves to prevent OOM during sparse refinement !!\n";
            thrust::device_ptr<uint32_t> t_moves(d_pairs);
            thrust::transform(
                t_moves, t_moves + curr_num_nodes, t_ranks, t_moves,
                [discard_offset] __host__ __device__ (const uint32_t move, const uint32_t rank) {
                    return rank >= discard_offset ? UINT32_MAX : move;
                }
            );
        }
        // ======================================

        // flag events that propose a valid move
        // => for size events, store "1" as the flag -> use the offsets with a "*2"
        // => for inbound events, store the node's "inbound set size" as the flag -> use the offsets with a "*2"
        dim_t *d_size_events_offsets = nullptr; // size_events_offsets[node idx] -> size event index for the node (where to write it - if it has a valid move)
        dim_t *d_inbound_events_offsets = nullptr; // inbound_events_offsets[node idx] -> initial inbound event index for the node (where to start writing them - if it has a valid move)
        CUDA_CHECK(cudaMalloc(&d_size_events_offsets, (curr_num_nodes + 1) * sizeof(dim_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_events_offsets, (curr_num_nodes + 1) * sizeof(dim_t)));
        CUDA_CHECK(cudaMemset(d_size_events_offsets, 0x00, (curr_num_nodes + 1) * sizeof(dim_t)));
        CUDA_CHECK(cudaMemset(d_inbound_events_offsets, 0x00, (curr_num_nodes + 1) * sizeof(dim_t)));
        thrust::device_ptr<dim_t> t_size_events_offsets(d_size_events_offsets);
        thrust::device_ptr<dim_t> t_inbound_events_offsets(d_inbound_events_offsets);
        {
            // launch configuration - fm-ref cascade sparse kernel
            // NOTE: choose threads_per_block multiple of WARP_SIZE
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = curr_num_nodes ; // 1 warp per node
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - fm-ref cascade sparse kernel
            LAUNCH(cfg) RUN << "fm-ref cascade sparse kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            fm_refinement_cascade_sparse_ppp_kernel<<<blocks, threads_per_block>>>(
                d_hedges,
                d_hedges_offsets,
                d_touching,
                d_touching_offsets,
                d_inbound_count,
                d_hedge_weights,
                d_ranks,
                d_pairs,
                d_partitions,
                d_ppp_offsets,
                d_ppp,
                num_hedges,
                curr_num_nodes,
                num_partitions,
                ppp_per_hedge,
                encourage,
                d_f_scores,
                d_size_events_offsets,
                d_inbound_events_offsets
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // not re-sorting the scores array means you have the array ordered as per the initial scores,
        // but now, this scan updates the scores "as if all previous moves were applied"!
        thrust::inclusive_scan(t_scores, t_scores + curr_num_nodes, t_scores); // in-place (we don't need scores anymore anyway)
        // REMEMBER: moves never get re-ranked (re-sorted) after the first time with in-isolation gains. Keep them like that and just find the valid sequence of maximum gain! This is an heuristics!

        // flag valid moves and compute their offsets for both event types
        thrust::exclusive_scan(t_size_events_offsets, t_size_events_offsets + curr_num_nodes + 1, t_size_events_offsets);
        thrust::exclusive_scan(t_inbound_events_offsets, t_inbound_events_offsets + curr_num_nodes + 1, t_inbound_events_offsets);
        // |
        dim_t num_size_events = 0; // last value in the inclusive scan = full reduce = total number of moving nodes
        CUDA_CHECK(cudaMemcpy(&num_size_events, d_size_events_offsets + curr_num_nodes, sizeof(dim_t), cudaMemcpyDeviceToHost));
        num_size_events *= 2; // => 2 events per flagged move
        // |
        dim_t num_inbound_events = 0; // last value in the inclusive scan = full reduce = total number of inbound hedges among all moving nodes
        CUDA_CHECK(cudaMemcpy(&num_inbound_events, d_inbound_events_offsets + curr_num_nodes, sizeof(dim_t), cudaMemcpyDeviceToHost));
        num_inbound_events *= 2; // => 2 events per flagged pin
        INFO(cfg) std::cout << "Refinement sparse events construction: " << num_size_events << " size events, " << num_inbound_events << " inbound events\n";

        // ======================================
        // extra step: compute moves validity by size (same HP as the kernel above: all previous higher-gain moves will be applied)
        // explode each move into two events, one decrementing and incrementing the size of the src and dst partition respectively
        // => seeing each move as two distinct events makes us able to identify sequences of useful events first, then moves
        uint32_t *d_size_events_partition = nullptr; // size_events_partition[ev] -> partition affected by the event
        uint32_t *d_size_events_index = nullptr; // size_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        int32_t *d_size_events_delta = nullptr; // size_events_delta[ev] -> size variation brought by the event
        CUDA_CHECK(cudaMalloc(&d_size_events_partition, num_size_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_size_events_index, num_size_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_size_events_delta, num_size_events * sizeof(int32_t)));
        thrust::device_ptr<uint32_t> t_size_events_partition(d_size_events_partition);
        thrust::device_ptr<uint32_t> t_size_events_index(d_size_events_index);
        thrust::device_ptr<int32_t> t_size_events_delta(d_size_events_delta);
        {
            // launch configuration - build size events sparse kernel
            int threads_per_block = 128;
            int num_threads_needed = curr_num_nodes; // 1 thread per move
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - build size events sparse kernel
            LAUNCH(cfg) RUN << "build size events sparse kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            // TODO: could filter out null moves (target = -1)?
            build_size_events_sparse_kernel<<<blocks, threads_per_block>>>(
                d_pairs,
                d_ranks,
                d_partitions,
                d_nodes_sizes,
                d_size_events_offsets,
                curr_num_nodes,
                d_size_events_partition,
                d_size_events_index,
                d_size_events_delta
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaFree(d_size_events_offsets));

        // sort events by (partition, rank) [in lexicographical order for the tuple] and carry size_events_delta along
        auto size_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_size_events_partition, t_size_events_index));
        auto size_events_key_end = size_events_key_begin + num_size_events;
        thrust::sort_by_key(size_events_key_begin, size_events_key_end, t_size_events_delta);
        // inclusive scan inside each key (= partition) on the event deltas => for each event we get the cumulative size delta for that partition at that point in the sequence
        thrust::inclusive_scan_by_key(t_size_events_partition, t_size_events_partition + num_size_events, t_size_events_delta, t_size_events_delta);
        // now mark moves that would violate size constraint if the sequence were to end on them
        int32_t *d_valid_moves = nullptr;
        CUDA_CHECK(cudaMalloc(&d_valid_moves, curr_num_nodes * sizeof(int32_t))); // valid_move[rank idx] -> 1 if applying all moves up to the idx one in the ordered sequence gives a valid state
        CUDA_CHECK(cudaMemset(d_valid_moves, 0x00, curr_num_nodes * sizeof(int32_t)));
        thrust::device_ptr<int32_t> t_valid_moves(d_valid_moves);
        
        {
            // launch configuration - flag size events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_size_events; // 1 thread per event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - flag size events kernel
            LAUNCH(cfg) RUN << "flag size events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
            // launch configuration - inbound sparse pins per partition update kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_hedges; // 1 warp per hedge
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - inbound sparse pins per partition update kernel
            LAUNCH(cfg) RUN << "inbound sparse pins per partition update kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            // NOTE: inbound-only version of the above used for constraints checks...
            sparse_inbound_pins_per_partition_update_kernel<<<blocks, threads_per_block>>>(
                d_hedges,
                d_hedges_offsets,
                d_srcs_count,
                d_partitions,
                d_ppp_offsets,
                num_hedges,
                ppp_per_hedge,
                d_ppp, // from now it represents inbound sets only
                d_partitions_inbound_sizes
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ======================================
        // if memory is reeeeallly tight, spill non-coarse data structures to host
        cudaMemGetInfo(&free_bytes, &total_bytes);
        std::vector<bitmap> h_ppp_offsets;
        std::vector<uint32_t> h_ppp;
        if (free_bytes < 9 * num_inbound_events * sizeof(uint32_t)) {
            // TODO: make these async
            h_ppp_offsets.resize(ppp_offsets_size);
            h_ppp.resize(ppp_size);
            INFO(cfg) std::cout << "Emergency sparse spill of " << std::fixed << std::setprecision(3)
                << (float)(ppp_offsets_size * sizeof(bitmap) + ppp_size * sizeof(uint32_t)) / (1 << 30)
                << " GB from device to host ...\n";
            CUDA_CHECK(cudaMemcpy(h_ppp_offsets.data(), d_ppp_offsets, ppp_offsets_size * sizeof(bitmap), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ppp.data(), d_ppp, ppp_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_ppp_offsets));
            CUDA_CHECK(cudaFree(d_ppp));
        }
        // ======================================

        // ======================================
        // extra step: compute moves validity by inbound set cardinality (same HP as the kernel above: all previous higher-gain moves will be applied)
        // explode each move into two events for every inbound hedge of the moved node, one decrementing and one incrementing the hedge's
        // occurrencies in the src partition's inbound set and dst partition's inbound set respectively
        // => results in n*h events (better than the n*h*p volume of conditions/counters we need to check)
        uint32_t *d_inbound_count_events_partition = nullptr; // inbound_count_events_partition[ev] -> partition affected by the event
        uint32_t *d_inbound_count_events_index = nullptr; // inbound_count_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        uint32_t *d_inbound_count_events_hedge = nullptr; // d_inbound_count_events_hedge[ev] -> hedge involved in the event
        int32_t *d_inbound_count_events_delta = nullptr; // inbound_count_events_delta[ev] -> inbound_count variation brought by the event
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_partition, num_inbound_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_index, num_inbound_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_hedge, num_inbound_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_count_events_delta, num_inbound_events * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_inbound_count_events_partition, 0xFF, num_inbound_events * sizeof(uint32_t))); // => use inbound_count_events_partition being UINT32_MAX to spot invalid events
        thrust::device_ptr<uint32_t> t_inbound_count_events_partition(d_inbound_count_events_partition);
        thrust::device_ptr<uint32_t> t_inbound_count_events_index(d_inbound_count_events_index);
        thrust::device_ptr<uint32_t> t_inbound_count_events_hedge(d_inbound_count_events_hedge);
        thrust::device_ptr<int32_t> t_inbound_count_events_delta(d_inbound_count_events_delta);
        {
            // launch configuration - build hedge events sparse kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = curr_num_nodes ; // 1 warp per move
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - build hedge events sparse kernel
            LAUNCH(cfg) RUN << "build hedge events sparse kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            // TODO: could filter out null moves (target = -1)?
            build_hedge_events_sparse_kernel<<<blocks, threads_per_block>>>(
                d_pairs,
                d_ranks,
                d_partitions,
                d_touching,
                d_touching_offsets,
                d_inbound_count,
                d_inbound_events_offsets,
                curr_num_nodes,
                d_inbound_count_events_partition,
                d_inbound_count_events_index,
                d_inbound_count_events_hedge,
                d_inbound_count_events_delta
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaFree(d_inbound_events_offsets));
        
        // sort events by (partition, hedge, rank) [in lexicographical order for the tuple] and carry events_delta along
        // the resulting array will have events sorted by partition, and inside each partition sorted by hedge, and inside each hedge sorted by rank!
        if (num_inbound_events > 0) {
            uint32_t *d_count_events_sort_keys = nullptr;
            uint32_t *d_count_events_sort_keys_buffer = nullptr;
            uint32_t *d_count_events_sort_permutation = nullptr;
            uint32_t *d_count_events_sort_permutation_buffer = nullptr;
            uint32_t *d_count_events_sort_reorder_buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&d_count_events_sort_keys, num_inbound_events * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_count_events_sort_keys_buffer, num_inbound_events * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_count_events_sort_permutation, num_inbound_events * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_count_events_sort_permutation_buffer, num_inbound_events * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_count_events_sort_reorder_buffer, num_inbound_events * sizeof(uint32_t)));

            cub::DoubleBuffer<uint32_t> c_count_events_keys_double_buffer(d_count_events_sort_keys, d_count_events_sort_keys_buffer);
            cub::DoubleBuffer<uint32_t> c_count_events_permutation_double_buffer(d_count_events_sort_permutation, d_count_events_sort_permutation_buffer);
            void* c_count_events_sort_storage = nullptr;
            size_t c_count_events_sort_storage_bytes = 0;

            thrust::device_ptr<uint32_t> t_count_events_sort_permutation(d_count_events_sort_permutation);
            thrust::device_ptr<uint32_t> t_count_events_sort_reorder_buffer(d_count_events_sort_reorder_buffer);
            thrust::sequence(t_count_events_sort_permutation, t_count_events_sort_permutation + num_inbound_events);

            auto write_count_events_sort_keys = [num_inbound_events, &c_count_events_keys_double_buffer, &c_count_events_permutation_double_buffer] (const uint32_t* d_field) {
                thrust::device_ptr<uint32_t> t_sort_keys(c_count_events_keys_double_buffer.Current());
                thrust::device_ptr<uint32_t> t_sort_permutation(c_count_events_permutation_double_buffer.Current());
                thrust::transform(
                    t_sort_permutation, t_sort_permutation + num_inbound_events, t_sort_keys,
                    [d_field] __host__ __device__ (const uint32_t idx) {
                        return d_field[idx];
                    }
                );
            };

            write_count_events_sort_keys(d_inbound_count_events_index);
            cub::DeviceRadixSort::SortPairs(
                c_count_events_sort_storage, c_count_events_sort_storage_bytes,
                c_count_events_keys_double_buffer, c_count_events_permutation_double_buffer,
                num_inbound_events,
                /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream=*/0
            );
            const float count_events_sort_aux_gib = static_cast<float>(5ull * num_inbound_events * sizeof(uint32_t)) / (1 << 30);
            CUB(cfg) std::cout << "CUB radix sort requiring " << std::fixed << std::setprecision(3) << count_events_sort_aux_gib
                << " GB of auxiliary buffers and " << std::fixed << std::setprecision(3) << static_cast<float>(c_count_events_sort_storage_bytes) / (1 << 20)
                << " MB of temporary storage ...\n";
            CUDA_CHECK(cudaMalloc(&c_count_events_sort_storage, c_count_events_sort_storage_bytes));
            cub::DeviceRadixSort::SortPairs(
                c_count_events_sort_storage, c_count_events_sort_storage_bytes,
                c_count_events_keys_double_buffer, c_count_events_permutation_double_buffer,
                num_inbound_events,
                /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream=*/0
            );

            write_count_events_sort_keys(d_inbound_count_events_hedge);
            cub::DeviceRadixSort::SortPairs(
                c_count_events_sort_storage, c_count_events_sort_storage_bytes,
                c_count_events_keys_double_buffer, c_count_events_permutation_double_buffer,
                num_inbound_events,
                /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream=*/0
            );

            write_count_events_sort_keys(d_inbound_count_events_partition);
            cub::DeviceRadixSort::SortPairs(
                c_count_events_sort_storage, c_count_events_sort_storage_bytes,
                c_count_events_keys_double_buffer, c_count_events_permutation_double_buffer,
                num_inbound_events,
                /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream=*/0
            );

            thrust::device_ptr<uint32_t> t_count_events_sorted_permutation(c_count_events_permutation_double_buffer.Current());
            thrust::gather(t_count_events_sorted_permutation, t_count_events_sorted_permutation + num_inbound_events, t_inbound_count_events_partition, t_count_events_sort_reorder_buffer);
            CUDA_CHECK(cudaMemcpy(d_inbound_count_events_partition, d_count_events_sort_reorder_buffer, num_inbound_events * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            thrust::gather(t_count_events_sorted_permutation, t_count_events_sorted_permutation + num_inbound_events, t_inbound_count_events_hedge, t_count_events_sort_reorder_buffer);
            CUDA_CHECK(cudaMemcpy(d_inbound_count_events_hedge, d_count_events_sort_reorder_buffer, num_inbound_events * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            thrust::gather(t_count_events_sorted_permutation, t_count_events_sorted_permutation + num_inbound_events, t_inbound_count_events_index, t_count_events_sort_reorder_buffer);
            CUDA_CHECK(cudaMemcpy(d_inbound_count_events_index, d_count_events_sort_reorder_buffer, num_inbound_events * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            thrust::device_ptr<int32_t> t_count_events_sort_reorder_buffer_i32(reinterpret_cast<int32_t*>(d_count_events_sort_reorder_buffer));
            thrust::gather(t_count_events_sorted_permutation, t_count_events_sorted_permutation + num_inbound_events, t_inbound_count_events_delta, t_count_events_sort_reorder_buffer_i32);
            CUDA_CHECK(cudaMemcpy(d_inbound_count_events_delta, d_count_events_sort_reorder_buffer, num_inbound_events * sizeof(int32_t), cudaMemcpyDeviceToDevice));

            CUDA_CHECK(cudaFree(d_count_events_sort_keys));
            CUDA_CHECK(cudaFree(d_count_events_sort_keys_buffer));
            CUDA_CHECK(cudaFree(d_count_events_sort_permutation));
            CUDA_CHECK(cudaFree(d_count_events_sort_permutation_buffer));
            CUDA_CHECK(cudaFree(d_count_events_sort_reorder_buffer));
            CUDA_CHECK(cudaFree(c_count_events_sort_storage));
        }
        // inclusive scan by key of the deltas, the key being (partition, hedge) -> we now have the total number of times each hedge appears in the inbound set as of each move (in order of rank)
        auto count_events_scan_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_inbound_count_events_partition, t_inbound_count_events_hedge));
        auto count_events_scan_key_end = count_events_scan_key_begin + num_inbound_events;
        thrust::inclusive_scan_by_key(count_events_scan_key_begin, count_events_scan_key_end, t_inbound_count_events_delta, t_inbound_count_events_delta);
        
        // ======================================
        // un-spill, crossing fingers that memory will be enough now
        if (free_bytes < 9 * num_inbound_events * sizeof(uint32_t)) {
            INFO(cfg) std::cout << "Emergency sparse unspill of " << std::fixed << std::setprecision(3)
                << (float)(ppp_offsets_size * sizeof(bitmap) + ppp_size * sizeof(uint32_t)) / (1 << 30)
                << " GB from host to device ...\n";
            CUDA_CHECK(cudaMalloc(&d_ppp_offsets, ppp_offsets_size * sizeof(bitmap)));
            CUDA_CHECK(cudaMalloc(&d_ppp, ppp_size * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_ppp_offsets, h_ppp_offsets.data(), ppp_offsets_size * sizeof(bitmap), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ppp, h_ppp.data(), ppp_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
            std::vector<bitmap>().swap(h_ppp_offsets);
            std::vector<uint32_t>().swap(h_ppp);
        }
        // ======================================

        // new array of events, one event for each time the counter of an hedge in the inbound set (+ the overall inbounds per partition counter) goes from 0 to >0,
        // the event carrying a +1 to the inbound set size, one event for each time the counter of an hedge goes from >0 to 0 carrying a -1 to the inbound set size for that partition
        dim_t *d_inbound_size_events_offsets = nullptr; // inbound_size_events_offsets[event idx] -> initially a flag of whether each event will produce an increase/decrese in inbound counts, after the scan it becomes the offset of each new event
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_offsets, (num_inbound_events + 1) * sizeof(dim_t)));
        CUDA_CHECK(cudaMemset(d_inbound_size_events_offsets, 0x00, (num_inbound_events + 1) * sizeof(dim_t)));
        {
            // launch configuration - count inbound events sparse kernel
            int threads_per_block = 128;
            int num_threads_needed = num_inbound_events; // 1 thread per hedge event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - count inbound events sparse kernel
            LAUNCH(cfg) RUN << "count inbound events sparse kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            count_inbound_size_events_sparse_ppp_kernel<<<blocks, threads_per_block>>>(
                d_ppp_offsets,
                d_ppp,
                d_inbound_count_events_partition,
                d_inbound_count_events_index,
                d_inbound_count_events_hedge,
                d_inbound_count_events_delta,
                num_inbound_events,
                num_partitions,
                ppp_per_hedge,
                d_inbound_size_events_offsets
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // transform the counts in offsets with a scan and find the total count of new size events
        thrust::device_ptr<dim_t> t_inbound_size_events_offsets(d_inbound_size_events_offsets);
        thrust::inclusive_scan(t_inbound_size_events_offsets, t_inbound_size_events_offsets + num_inbound_events + 1, t_inbound_size_events_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
        dim_t num_inbound_size_events = 0; // last value in the inclusive scan = full reduce = total number of touching hedges among all sets
        CUDA_CHECK(cudaMemcpy(&num_inbound_size_events, d_inbound_size_events_offsets + num_inbound_events, sizeof(dim_t), cudaMemcpyDeviceToHost));
        uint32_t *d_inbound_size_events_partition = nullptr; // inbound_size_events_partition[ev] -> partition affected by the event
        uint32_t *d_inbound_size_events_index = nullptr; // inbound_size_events_index[ev] -> sequence position / idx of the move (w.r.t. d_ranks) that originated the event
        int32_t *d_inbound_size_events_delta = nullptr; // inbound_size_events_delta[ev] -> inbound set size variation brought by the event
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_partition, num_inbound_size_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_index, num_inbound_size_events * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_inbound_size_events_delta, num_inbound_size_events * sizeof(int32_t)));
        thrust::device_ptr<uint32_t> t_inbound_size_events_partition(d_inbound_size_events_partition);
        thrust::device_ptr<uint32_t> t_inbound_size_events_index(d_inbound_size_events_index);
        thrust::device_ptr<int32_t> t_inbound_size_events_delta(d_inbound_size_events_delta);

        {
            // launch configuration - build inbound events sparse kernel
            int threads_per_block = 128;
            int num_threads_needed = num_inbound_events; // 1 thread per hedge event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - build inbound events sparse kernel
            LAUNCH(cfg) RUN << "build inbound events sparse kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            build_inbound_size_events_sparse_ppp_kernel<<<blocks, threads_per_block>>>(
                d_ppp_offsets,
                d_ppp,
                d_inbound_count_events_partition,
                d_inbound_count_events_index,
                d_inbound_count_events_hedge,
                d_inbound_count_events_delta,
                d_inbound_size_events_offsets,
                num_inbound_events,
                num_partitions,
                ppp_per_hedge,
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
        CUDA_CHECK(cudaMemset(d_inbound_valid_moves, 0x00, curr_num_nodes * sizeof(int32_t)));
        thrust::device_ptr<int32_t> t_inbound_valid_moves(d_inbound_valid_moves);
        if (num_inbound_size_events > 0) {
            // launch configuration - flag inbound events kernel
            int threads_per_block = 128;
            int num_threads_needed = num_inbound_size_events; // 1 thread per event
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - flag inbound events kernel
            LAUNCH(cfg) RUN << "flag inbound events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
        INFO(cfg) std::cout << "Best fm-ref move:\n  Move rank: " << best_rank << ", Acquired gain: " << acquired_gain << "\n";
        if (size_validity <= 0 && inbounds_validity <= 0 && acquired_gain >= 0) {
            // launch configuration - fm-ref apply kernel
            int threads_per_block = 128;
            int num_threads_needed = curr_num_nodes; // 1 thread per move to apply
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - fm-ref apply kernel
            LAUNCH(cfg) RUN << "fm-ref apply (" << num_good_moves << " good moves) kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
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
            INFO(cfg) {
                std::cout << "No valid refinement move found on level " << level_idx << " - reason: "
                    << (size_validity > 0 ? (inbounds_validity > 0 ? "both size and inbounds validities" : "size validity") : (inbounds_validity > 0 ? "inbounds validity" : "negative gain")) << "\n";
                if (size_validity > 0) std::cout << "  Size constraint violations variation amount (in nodes above the limit): " << size_validity << "\n";
                if (inbounds_validity > 0) std::cout << "  Inbound constraint violations variation (in invalid partitions): " << inbounds_validity << "\n";
            }
            if (size_validity > 0 && !chainup) chainup = true; // enable chaining when no moves are available via greedy sorting because of size constraints
            else if (fm_repeat < cfg.refine_repeats / 3) fm_repeat = cfg.refine_repeats / 2;
            else if (fm_repeat < 2 * cfg.refine_repeats / 3) fm_repeat = 2 * cfg.refine_repeats / 3;
            else fm_repeat = cfg.refine_repeats; // aka break!
        }
        CUDA_CHECK(cudaFree(d_ranks));
        CUDA_CHECK(cudaFree(d_valid_moves));
        CUDA_CHECK(cudaFree(d_inbound_valid_moves));
    }

    CUDA_CHECK(cudaFree(d_ppp_offsets));
    CUDA_CHECK(cudaFree(d_ppp));

    // recompute inbound set sizes
    if (update_final_inbound_counts) {
        uint32_t* pp_map = nullptr; // pp_map[(e * num_partitions + p)/32] -> the bit is set if "e" was already seen incident to "p"
        const dim_t pp_per_hedge = (static_cast<dim_t>(num_partitions) + 31u) / 32u;
        const dim_t pp_map_size = static_cast<dim_t>(num_hedges) * pp_per_hedge;
        CUDA_CHECK(cudaMalloc(&pp_map, pp_map_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(pp_map, 0x00, pp_map_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_partitions_inbound_sizes, 0x00, num_partitions * sizeof(uint32_t)));
        {
            // launch configuration - inbound sets size kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_hedges; // 1 warp per hedge
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - inbound sets size kernel
            LAUNCH(cfg) RUN << "inbound sets size kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            inbound_sets_size_kernel<<<blocks, threads_per_block>>>(
                d_hedges,
                d_hedges_offsets,
                d_srcs_count,
                d_partitions,
                num_hedges,
                num_partitions,
                pp_map,
                d_partitions_inbound_sizes
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaFree(pp_map));
    }
}


// LOGGING

void logPartitions(
    const uint32_t *d_partitions,
    const uint32_t *d_partitions_sizes,
    const uint32_t *d_partitions_inbound_sizes,
    const uint32_t curr_num_nodes,
    const uint32_t num_partitions,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part
) {
    std::vector<uint32_t> partitions_tmp(curr_num_nodes);
    CUDA_CHECK(cudaMemcpy(partitions_tmp.data(), d_partitions, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> partitions_sizes_tmp(num_partitions);
    CUDA_CHECK(cudaMemcpy(partitions_sizes_tmp.data(), d_partitions_sizes, num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> partitions_inbound_sizes_tmp(num_partitions);
    CUDA_CHECK(cudaMemcpy(partitions_inbound_sizes_tmp.data(), d_partitions_inbound_sizes, num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost));
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
        uint32_t part_inbound_size = partitions_inbound_sizes_tmp[i];
        if (part_size > h_max_nodes_per_part)
           std::cerr << "  WARNING, max partition size constraint (" << h_max_nodes_per_part << ") violated by part=" << i << " with part_size=" << part_size << " !!\n";
        if (part_inbound_size > h_max_inbound_per_part)
            std::cerr << "  WARNING, max partition inbound size constraint (" << h_max_inbound_per_part << ") violated by part=" << i << " with part_inbound_size=" << part_inbound_size << " !!\n";
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