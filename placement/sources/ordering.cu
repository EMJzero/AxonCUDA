#include <stdint.h>
#include <algorithm>

#include <cub/cub.cuh>

#include "thruster.cuh"

#include "runconfig_plc.hpp"

#include "utils.cuh"
#include "utils_plc.cuh"
#include "ordering.cuh"

// for each partition, randomly bisect it, mapping every partition id "p" to either "p*2" or "p*2+1"
void split_partitions_rand(
    const runconfig &cfg,
    uint32_t* d_partitions,
    uint32_t num_nodes,
    uint32_t num_parts,
    curandGenerator_t gen,
    const cudaStream_t stream,
    const int tid
) {
    auto thrust_exec = thrust::cuda::par.on(stream);
    thrust::device_ptr<uint32_t> t_partitions(d_partitions);

    thrust::device_vector<uint32_t> t_original_idx(num_nodes); // original_idx[i] -> idx of node currently in partition partitions[i]
    thrust::sequence(thrust_exec, t_original_idx.begin(), t_original_idx.end());

    uint32_t* d_partitions_cpy = nullptr; // auxiliary copy of current partitions for sorting and scattering
    CUDA_CHECK(cudaMallocAsync(&d_partitions_cpy, num_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_partitions_cpy, d_partitions, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    thrust::device_ptr<uint32_t> t_partitions_cpy(d_partitions_cpy);

    // generate one random uint32 per element, seeded
    thrust::device_vector<uint32_t> t_rand_keys(num_nodes);
    CURAND_CHECK(curandSetStream(gen, stream));
    CURAND_CHECK(curandGenerate(gen, thrust::raw_pointer_cast(t_rand_keys.data()), num_nodes));

    // sort by (partition, random), carrying along the original indices
    // => now partitions_cpy is grouped by "p", with random order inside each group
    auto part_rand_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_partitions_cpy, t_rand_keys.begin()));
    auto part_rand_key_end = part_rand_key_begin + num_nodes;
    thrust::sort_by_key(thrust_exec, part_rand_key_begin, part_rand_key_end, t_original_idx.begin());

    // build offset indices over reordered partitions
    thrust::device_vector<uint32_t> t_part_offsets(num_parts + 1); // part_offsets[p] -> first index of partition p in partitions_cpy
    thrust::counting_iterator<uint32_t> search_begin(0);
    thrust::lower_bound(
        thrust_exec,
        t_partitions_cpy, t_partitions_cpy + num_nodes,
        search_begin, search_begin + num_parts,
        t_part_offsets.begin()
    );
    t_part_offsets[num_parts] = num_nodes;

    // split each partition in half:
    // - inside each partition, original node indices are not randomly ordered
    // - take the lower half of those indices and map it to p*2, take the upper half and map it to p*2+1
    {
        // launch configuration - split partitions kernel
        int threads_per_block = 256;
        int num_threads_needed = num_nodes; // 1 thread per node
        int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - split partitions kernel
        LAUNCH(cfg) TID(tid) RUN << "split partitions kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        split_partitions_kernel<<<blocks, threads_per_block, 0, stream>>>(
            thrust::raw_pointer_cast(t_part_offsets.data()),
            num_nodes,
            d_partitions_cpy
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // undo the sort - scatter back to updated partitions to their original idxs
    thrust::scatter(
        thrust_exec,
        t_partitions_cpy,
        t_partitions_cpy + num_nodes,
        t_original_idx.begin(),
        t_partitions
    );

    CUDA_CHECK(cudaFreeAsync(d_partitions_cpy, stream));
}

void compute_partitions_cutnet(
    const runconfig &cfg,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const float* d_hedge_weights,
    const uint32_t* d_partitions,
    const uint32_t num_hedges,
    const uint32_t num_parts,
    const dim_t hedges_size,
    float* d_cutnet,
    const cudaStream_t stream,
    const int tid
) {
    auto thrust_exec = thrust::cuda::par.on(stream);

    /*
    * IDEA:
    * - prepare a copy of the segmented hedge buffer
    * - map operation to replace each pin with its partition
    * - segmented sort inside each hedge
    * - filter operation to keep only (within each segmente) the even numbers that are followed by their value +1 (their odd partition in the pair)
    *   - not need exactly to remove the elements, but to spot relevant ones
    * - flag surviving elements and prefix sum the flags, this gives you a unique offset per element
    * - for each surviving element create an event containing the tuple (hedge weight, partition id / 2), divide by 2 to get the parent partition's id
    *   - could put in the event the hedge's id, and recover the weight later, but little would changes
    *   - could sort immediately after filtering, but work with a larger buffer...
    * - sort events by parent partition id, and do a segmented reduce within each parent id, that's each parent partition's cutnet cost
    *
    * TODO: switch from the segmented sort to deduplicating part_pins in shared-memory for each hedge, and directly yielding the count of unique partitions per hedge
    *       from the count and offsets (aka, flags) you then allocate the buffer and repeat the deduplication to write final cuts
    */

    uint32_t* d_part_pins = nullptr; // part_pins[hedges_offsets[hedge idx] + pin idx] -> partition the pin is in
    CUDA_CHECK(cudaMallocAsync(&d_part_pins, hedges_size * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_part_pins, d_hedges, hedges_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));

    thrust::device_ptr<uint32_t> t_part_pins(d_part_pins);
    thrust::device_ptr<const uint32_t> t_partitions(d_partitions);
    thrust::device_ptr<const dim_t> t_hedges_offsets(d_hedges_offsets);
    thrust::device_ptr<float> t_cutnet(d_cutnet);

    // map pins to their partition -> map each entry part_pins[i] to partitions[t_part_pins[i]]
    thrust::gather(thrust_exec, t_part_pins, t_part_pins + hedges_size, t_partitions, t_part_pins);

    // segmented sort of part_pins (using the segments from hedges)
    uint32_t* d_part_pins_buffer = nullptr; // CUB segmented sort buffer
    CUDA_CHECK(cudaMallocAsync(&d_part_pins_buffer, hedges_size * sizeof(uint32_t), stream));
    cub::DoubleBuffer<uint32_t> c_part_pins_double_buffer(d_part_pins, d_part_pins_buffer);
    void* c_part_pins_storage = nullptr;
    size_t c_part_pins_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(
        c_part_pins_storage, c_part_pins_storage_bytes, c_part_pins_double_buffer,
        hedges_size, num_hedges, d_hedges_offsets, d_hedges_offsets + 1,
        /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, stream
    );
    CUB(cfg) std::cout TID(tid) << "CUB segmented sort requiring " << std::fixed << std::setprecision(3) << (float)(hedges_size * sizeof(uint32_t)) / (1 << 30)
        << " GB of pong-buffer and " << std::fixed << std::setprecision(3) << ((float)c_part_pins_storage_bytes) / (1 << 20)
        << " MB of temporary storage ...\n";
    CUDA_CHECK(cudaMallocAsync(&c_part_pins_storage, c_part_pins_storage_bytes, stream));
    cub::DeviceSegmentedRadixSort::SortKeys(
        c_part_pins_storage, c_part_pins_storage_bytes, c_part_pins_double_buffer,
        hedges_size, num_hedges, d_hedges_offsets, d_hedges_offsets + 1,
        /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, stream
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (c_part_pins_double_buffer.Current() != d_part_pins) {
        uint32_t* tmp = d_part_pins_buffer;
        d_part_pins_buffer = d_part_pins;
        d_part_pins = tmp;
    }
    CUDA_CHECK(cudaFreeAsync(d_part_pins_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(c_part_pins_storage, stream));

    dim_t* d_flags = nullptr; // event_weight[idx] -> weight of the hedge being cut in event idx
    CUDA_CHECK(cudaMallocAsync(&d_flags, (hedges_size + 1) * sizeof(dim_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_flags, 0x00, (hedges_size + 1) * sizeof(dim_t), stream));
    thrust::device_ptr<dim_t> t_flags(d_flags);
    {
        // launch configuration - flag cutnet events kernel
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = num_hedges; // 1 warp per hedge
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - flag cutnet events kernel
        LAUNCH(cfg) TID(tid) RUN << "flag cutnet events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        flag_cutnet_events_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_part_pins,
            d_hedges_offsets,
            num_hedges,
            d_flags
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // exclusive prefix sum of flags, then extract the last value (total sum) as the events count
    thrust::exclusive_scan(thrust_exec, t_flags, t_flags + hedges_size + 1, t_flags);
    dim_t events_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&events_count, d_flags + hedges_size, sizeof(dim_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float* d_event_weight = nullptr; // event_weight[idx] -> weight of the hedge being cut in event idx
    uint32_t* d_event_part = nullptr; // event_part[idx] -> partition/2 affected by event idx

    if (events_count > 0) {
        CUDA_CHECK(cudaMallocAsync(&d_event_weight, events_count * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&d_event_part, events_count * sizeof(uint32_t), stream));

        thrust::device_ptr<float> t_event_weight(d_event_weight);
        thrust::device_ptr<uint32_t> t_event_part(d_event_part);

        // for each part_pins entry that previously generated a flag, use the new prefix-summed flags as the index in event_weight and event_part
        // where to let that part_pins entry write its content (in event_part) and its hedge's weight (in event_weight)
        {
            // launch configuration - cutnet event generation kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_hedges; // 1 warp per hedge
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - cutnet event generation kernel
            LAUNCH(cfg) TID(tid) RUN << "cutnet event generation kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            cutnet_event_generation_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_part_pins,
                d_hedges_offsets,
                d_hedge_weights,
                d_flags,
                num_hedges,
                d_event_weight,
                d_event_part
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // sort event_weight and event_part both according to event_part
        thrust::sort_by_key(thrust_exec, t_event_part, t_event_part + events_count, t_event_weight);

        // reduce-sum each segment of event_weight with the same event_part value and store the result in cutnet[t_event_part[.]]
        uint32_t* d_unique_event_part = nullptr; // unique_event_part[idx] -> idx-th partition/2 that generated at least one cutnet event
        float* d_unique_event_weight = nullptr; // unique_event_weight[idx] -> reduced cutnet contribution for unique_event_part[idx]
        CUDA_CHECK(cudaMallocAsync(&d_unique_event_part, events_count * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMallocAsync(&d_unique_event_weight, events_count * sizeof(float), stream));
        thrust::device_ptr<uint32_t> t_unique_event_part(d_unique_event_part);
        thrust::device_ptr<float> t_unique_event_weight(d_unique_event_weight);
        auto reduced_end = thrust::reduce_by_key(
            thrust_exec,
            t_event_part, t_event_part + events_count, t_event_weight,
            t_unique_event_part, t_unique_event_weight
        );
        dim_t unique_events_count = thrust::get<0>(reduced_end) - t_unique_event_part;
        thrust::scatter(thrust_exec, t_unique_event_weight, t_unique_event_weight + unique_events_count, t_unique_event_part, t_cutnet);

        CUDA_CHECK(cudaFreeAsync(d_unique_event_part, stream));
        CUDA_CHECK(cudaFreeAsync(d_unique_event_weight, stream));
        CUDA_CHECK(cudaFreeAsync(d_event_weight, stream));
        CUDA_CHECK(cudaFreeAsync(d_event_part, stream));
    } else {
        thrust::fill(thrust_exec, t_cutnet, t_cutnet + num_parts / 2, FLT_MAX);
    }

    CUDA_CHECK(cudaFreeAsync(d_part_pins, stream));
    CUDA_CHECK(cudaFreeAsync(d_flags, stream));
}

// return a high-locality, seeded 1D ordering of nodes
uint32_t* locality_ordering(
    const runconfig &cfg,
    const uint32_t num_nodes,
    const uint32_t num_hedges,
    const dim_t hedges_size,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const float* d_hedge_weights,
    const uint32_t* d_touching,
    const dim_t* d_touching_offsets,
    const uint64_t seed,
    const cudaStream_t stream,
    const int tid
) {
    auto thrust_exec = thrust::cuda::par.on(stream);
    /*
    * IDEA:
    * - recursive bisection
    * - divide nodes in two halves, and each half in half again, and so creating a binary tree where each leaf is a single node
    *  - done with a random balanced bipartitioning and repeated label propagation
    * - then go back up the binary tree
    * - at each branch, decide how to order the two halves ("which goes first")
    *   - check whether you are on the left or right side of your grandparent
    *   - compute how strongly each half is connected with the other side of your grandparent
    *   - if the grandparent has you on the left, and your left side is more strongly connected to grandpa's right,
    *       reverse the order of every leaf under yourself, trapping good connections inside, while favoring links with grandpa's second child
    *   - else, everything stays as it is  - mirrored idea if you are on grandpa's right side
    * - recurse up until the root, that gives you a strong 1D ordering that trapped connection locality as much as possible
    * 
    * Label propagation:
    * - 60% of the logic is the one from your failed initial partition implementation
    * - you need to compute pins-per-partition on the fly, but they are just for a bi-partitioning, so it’s cheap (just two counters per hedge as seen from each super-partition)
    * - use the same gain-even-on-no-disconnect logic to encourage node moves at the beginning, but stabilize to no gain unless you disconnect on later iterations
    * - seed the random initial partitioning, before
    * - to select moves to apply, do a segmented sort by gain, where segments are the moves proposed in each super-partition
    * - to keep the strictest balance:
    *     - initialize as balanced
    *     - sort moves by gain AND such that every move is followed by one in the inverse direct, then count such that you keep a number that keeps balance (be wary of partitions differing by one in size
    * - yes, you need two partition arrays, one for super-partitions (the layer about you in the tree), and one for newly built ones
    */

    /*
    * How to generate partition ids:
    * - given a partition with id "p", currently being bisected, its two child partitions will have ids:
    *   - p*2   - p*2 + 1
    * - this works because the number of partitions doubles at every bisection of every partition
    * - uneven partitions will lead to some non-existing id, be wary
    * - when re-merging partitions, fuse all even "p"s with their odd successor and give both the "p/2" id
    */

    assert(num_nodes > 0); // how, why, what?!

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandSetStream(gen, stream));

    uint32_t* d_partitions = nullptr; // partitions[node idx] -> current partition (of bypartitions) the node is in

    CUDA_CHECK(cudaMallocAsync(&d_partitions, num_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_partitions, 0x00, num_nodes * sizeof(uint32_t), stream)); // everyone starts in the same partition
    thrust::device_ptr<uint32_t> t_partitions(d_partitions);

    uint32_t num_parts = 1u;

    bool* d_moves = nullptr; // move[node idx] -> false if the node doesn't want to move, true if the node would like to switch partition p*2->p*2+1 or p*2+1->p*2
    float* d_scores = nullptr; // score[node idx] -> connectivity gain for the above move (even not moving is done with a "gain")
    uint32_t* d_even_event_idx = nullptr; // event_idx[node idx] -> event idx for the node, in case the node decided to move (the node was in an even partition)
    uint32_t* d_odd_event_idx = nullptr; // event_idx[node idx] -> ... (the node was in an odd partition)

    CUDA_CHECK(cudaMallocAsync(&d_moves, num_nodes * sizeof(bool), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scores, num_nodes * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_even_event_idx, (num_nodes + 1) * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_odd_event_idx, (num_nodes + 1) * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_even_event_idx(d_even_event_idx);
    thrust::device_ptr<uint32_t> t_odd_event_idx(d_odd_event_idx);

    // IDEA:
    // - initialize this on each level from current partitions
    // - after label prop, compute the new cutnet (=connectivity) for each pair of partitions
    // - iff a partitions pair's cutnet improved, copy over here the new partition ids for the nodes of that pair of partitions
    // - before going to the next level, make this the actual partitioning
    uint32_t* d_last_best_partitions = nullptr; // last_best_partitions [node idx] -> last best partition (of bypartitions) the node was in
    CUDA_CHECK(cudaMallocAsync(&d_last_best_partitions, num_nodes * sizeof(uint32_t), stream));

    uint32_t level_idx = 0u;
    while (num_parts < (num_nodes + 1) / 2) { // as long as partitions do not strictly contain 1 or 2 nodes...
        INFO(cfg) std::cout TID(tid) << "Bisection level " << level_idx << " number of partitions=" << num_parts << "\n";
        level_idx++;

        // random bisection of every partition
        split_partitions_rand(
            cfg,
            d_partitions,
            num_nodes,
            num_parts,
            gen,
            stream,
            tid
        );

        num_parts *= 2;

        // initialize best partitions
        CUDA_CHECK(cudaMemcpyAsync(d_last_best_partitions, d_partitions, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));

        float* d_cutnet = nullptr; // cutnet [part idx / 2 | (part idx + 1) / 2] -> current cutnet for the pair of partitions
        float* d_last_best_cutnet = nullptr; // last_best_cutnet [part idx / 2 | (part idx + 1) / 2] -> last best cutnet for that pair of partitions
        CUDA_CHECK(cudaMallocAsync(&d_cutnet, (num_parts / 2) * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&d_last_best_cutnet, (num_parts / 2) * sizeof(float), stream));
        thrust::device_ptr<float> t_cutnet(d_cutnet);
        thrust::device_ptr<float> t_last_best_cutnet(d_last_best_cutnet);

        // compute initial partitions cutnet
        compute_partitions_cutnet(
            cfg,
            d_hedges,
            d_hedges_offsets,
            d_hedge_weights,
            d_partitions,
            num_hedges,
            num_parts,
            hedges_size,
            d_last_best_cutnet,
            stream,
            tid
        );

        for (uint32_t lp_repeat = 0u; lp_repeat < cfg.labelprop_repeats; lp_repeat++) {
            // compute gains (and moves) in-isolation
            // NOTE: no need to init. "d_moves" and "d_scores", they are overwritten anyway
            {
                // launch configuration - label propagation kernel
                int threads_per_block = 128; // 128/32 -> 4 warps per block
                int warps_per_block = threads_per_block / WARP_SIZE;
                int num_warps_needed = num_nodes; // 1 warp per node
                int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
                // launch - label propagation kernel
                LAUNCH(cfg) TID(tid) RUN << "label propagation kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                label_propagation_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_hedges,
                    d_hedges_offsets,
                    d_touching,
                    d_touching_offsets,
                    d_hedge_weights,
                    d_partitions,
                    num_nodes,
                    d_moves,
                    d_even_event_idx,
                    d_odd_event_idx,
                    d_scores
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            
            // build move events (partition, score, node)
            CUDA_CHECK(cudaMemsetAsync(d_even_event_idx + num_nodes, 0u, sizeof(uint32_t), stream));
            CUDA_CHECK(cudaMemsetAsync(d_odd_event_idx + num_nodes, 0u, sizeof(uint32_t), stream));
            thrust::exclusive_scan(thrust_exec, t_even_event_idx, t_even_event_idx + num_nodes + 1, t_even_event_idx); // in-place
            thrust::exclusive_scan(thrust_exec, t_odd_event_idx, t_odd_event_idx + num_nodes + 1, t_odd_event_idx); // in-place
            uint32_t even_events_count;
            uint32_t odd_events_count;
            CUDA_CHECK(cudaMemcpyAsync(&even_events_count, d_even_event_idx + num_nodes, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&odd_events_count, d_odd_event_idx + num_nodes, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (even_events_count == 0 || odd_events_count == 0) {
                INFO(cfg) std::cout TID(tid) << "No valid label propagation move found on level " << level_idx << " repeat " << lp_repeat << " (even events count=" << even_events_count << " odd events count=" << odd_events_count << ") !!\n";
                break;
            }
            INFO(cfg) std::cout TID(tid) << "Label propagation on level " << level_idx << " repeat " << lp_repeat << " (even events count=" << even_events_count << " odd events count=" << odd_events_count << ")\n";

            // NOTE: partitions inside events are stored as p/2 and (p-1)/2 !!
            uint32_t* d_even_event_part = nullptr; // part[idx] -> src partition / 2 for the idx-th move (partition being even)
            float* d_even_event_score = nullptr; // score[idx] -> gain for the idx-th move
            uint32_t* d_even_event_node = nullptr; // node[idx] -> node moved in the idx-th move
            uint32_t* d_odd_event_part = nullptr; // part[idx] -> (src partition - 1) / 2 ... (partition being odd)
            float* d_odd_event_score = nullptr; // score[idx] -> ...
            uint32_t* d_odd_event_node = nullptr; // node[idx] -> ...
            CUDA_CHECK(cudaMallocAsync(&d_even_event_part, even_events_count * sizeof(uint32_t), stream));
            CUDA_CHECK(cudaMallocAsync(&d_even_event_score, even_events_count * sizeof(float), stream));
            CUDA_CHECK(cudaMallocAsync(&d_even_event_node, even_events_count * sizeof(uint32_t), stream));
            CUDA_CHECK(cudaMallocAsync(&d_odd_event_part, odd_events_count * sizeof(uint32_t), stream));
            CUDA_CHECK(cudaMallocAsync(&d_odd_event_score, odd_events_count * sizeof(float), stream));
            CUDA_CHECK(cudaMallocAsync(&d_odd_event_node, odd_events_count * sizeof(uint32_t), stream));
            thrust::device_ptr<uint32_t> t_even_event_part(d_even_event_part);
            thrust::device_ptr<float> t_even_event_score(d_even_event_score);
            thrust::device_ptr<uint32_t> t_even_event_node(d_even_event_node);
            thrust::device_ptr<uint32_t> t_odd_event_part(d_odd_event_part);
            thrust::device_ptr<float> t_odd_event_score(d_odd_event_score);
            thrust::device_ptr<uint32_t> t_odd_event_node(d_odd_event_node);
            {
                // launch configuration - label move events kernel
                int threads_per_block = 256;
                int num_threads_needed = num_nodes; // 1 thread per node
                int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
                // launch - label move events kernel
                LAUNCH(cfg) TID(tid) RUN << "label move events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                label_move_events_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_moves,
                    d_scores,
                    d_even_event_idx,
                    d_odd_event_idx,
                    d_partitions,
                    num_nodes,
                    d_even_event_part,
                    d_even_event_score,
                    d_even_event_node,
                    d_odd_event_part,
                    d_odd_event_score,
                    d_odd_event_node
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // sort events by (partition, score, node)
            auto move_even_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_even_event_part, t_even_event_score, t_even_event_node));
            auto move_even_events_key_end = move_even_events_key_begin + even_events_count;
            thrust::sort(thrust_exec, move_even_events_key_begin, move_even_events_key_end);
            // |
            auto move_odd_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_odd_event_part, t_odd_event_score, t_odd_event_node));
            auto move_odd_events_key_end = move_odd_events_key_begin + odd_events_count;
            thrust::sort(thrust_exec, move_odd_events_key_begin, move_odd_events_key_end);

            // build offset indices over reordered events per partition
            uint32_t* d_part_even_event_offsets = nullptr; // part_even_event_offsets[p] -> first index of partition p*2 in even_event_part
            CUDA_CHECK(cudaMallocAsync(&d_part_even_event_offsets, (num_parts/2 + 1) * sizeof(uint32_t), stream));
            thrust::device_ptr<uint32_t> t_part_even_event_offsets(d_part_even_event_offsets);
            thrust::counting_iterator<uint32_t> even_search_begin(0);
            // NOTE: the search was "made to work" by storing p/2 inside event_part-s, hence it is enough to search from 0 to num_parts/2
            thrust::lower_bound(
                thrust_exec,
                t_even_event_part, t_even_event_part + even_events_count,
                even_search_begin, even_search_begin + num_parts/2,
                t_part_even_event_offsets
            );
            CUDA_CHECK(cudaMemcpyAsync(d_part_even_event_offsets + num_parts/2, &even_events_count, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            // |
            uint32_t* d_part_odd_event_offsets = nullptr; // part_odd_event_offsets[p] -> first index of partition p*2+1 in odd_event_part
            CUDA_CHECK(cudaMallocAsync(&d_part_odd_event_offsets, (num_parts/2 + 1) * sizeof(uint32_t), stream));
            thrust::device_ptr<uint32_t> t_part_odd_event_offsets(d_part_odd_event_offsets);
            thrust::counting_iterator<uint32_t> odd_search_begin(0);
            thrust::lower_bound(
                thrust_exec,
                t_odd_event_part, t_odd_event_part + odd_events_count,
                odd_search_begin, odd_search_begin + num_parts/2,
                t_part_odd_event_offsets
            );
            CUDA_CHECK(cudaMemcpyAsync(d_part_odd_event_offsets + num_parts/2, &odd_events_count, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

            // repurpose "event_idx" to store the reverse map: event_idx[node] -> event-idx (if any) of node - in other words this scatter does "event_idx[event_node[i]] = i"
            CUDA_CHECK(cudaMemsetAsync(d_even_event_idx, 0xFF, num_nodes * sizeof(uint32_t), stream));
            thrust::scatter(thrust_exec, thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(even_events_count), t_even_event_node, t_even_event_idx);
            CUDA_CHECK(cudaMemsetAsync(d_odd_event_idx, 0xFF, num_nodes * sizeof(uint32_t), stream));
            thrust::scatter(thrust_exec, thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(odd_events_count), t_odd_event_node, t_odd_event_idx);

            // update gains in-sequence
            // assume moves are done in pairs => re-compute the pair's gain in-sequence, assuming all prior pairs already swapped
            // => already accumulate the two scores on the "even" segment's event (only up to the length of the smallest events segment between even and odd)
            CUDA_CHECK(cudaMemsetAsync(d_even_event_score, 0x00, even_events_count * sizeof(float), stream));
            {
                // launch configuration - label cascade kernel
                int threads_per_block = 128; // 128/32 -> 4 warps per block
                int warps_per_block = threads_per_block / WARP_SIZE;
                int num_warps_needed = even_events_count + odd_events_count; // 1 warp per event
                int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
                // launch - label cascade kernel
                LAUNCH(cfg) TID(tid) RUN << "label cascade kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                label_cascade_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_hedges,
                    d_hedges_offsets,
                    d_touching,
                    d_touching_offsets,
                    d_hedge_weights,
                    d_partitions,
                    d_part_even_event_offsets,
                    d_part_odd_event_offsets,
                    d_even_event_idx,
                    d_odd_event_idx,
                    d_even_event_node,
                    d_odd_event_node,
                    even_events_count,
                    odd_events_count,
                    d_even_event_score
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // inclusive scan inside each key (= partition) on the even event scores => for each event we get the cumulative gain up to that point in the partition's move sequence
            thrust::inclusive_scan_by_key(thrust_exec, t_even_event_part, t_even_event_part + even_events_count, t_even_event_score, t_even_event_score);
            // extract the maximum idx (relative to the start of the overall array) for every partition's pair
            // => repurpose d_even_event_idx as even_event_idx[idx] -> absolute idx of the last moves-pair to apply for partitions 2*idx and 2*idx+1
            auto event_score_pair = thrust::make_zip_iterator(thrust::make_tuple(t_even_event_score, thrust::counting_iterator<uint32_t>(0)));
            CUDA_CHECK(cudaMemsetAsync(d_even_event_idx, 0xFF, (num_parts / 2) * sizeof(uint32_t), stream));
            auto d_event_argmax = thrust::make_transform_output_iterator(
                t_even_event_idx, // discard the "max" part of the "argmax" return tuple
                [] __device__ (auto x) { return (thrust::get<0>(x) <= 0.0f) ? UINT32_MAX : thrust::get<1>(x); }
            );
            thrust::reduce_by_key(
                thrust_exec,
                t_even_event_part, t_even_event_part + even_events_count, event_score_pair,
                thrust::make_discard_iterator(), d_event_argmax,
                thrust::equal_to<uint32_t>{},
                [] __device__ (auto a, auto b) { return (thrust::get<0>(b) > thrust::get<0>(a)) ? b : a; }
            );

            // apply pairs of improving moves
            // add together the gain of equi-ranked moves between bisected partitions as the gain of the pair to swap
            {
                // launch configuration - apply move events kernel
                int threads_per_block = 256;
                int num_threads_needed = even_events_count; // 1 thread per event
                int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
                // launch - apply move events kernel
                LAUNCH(cfg) TID(tid) RUN << "apply move events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                apply_move_events_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_even_event_idx,
                    d_even_event_part,
                    d_even_event_node,
                    d_part_even_event_offsets,
                    d_part_odd_event_offsets,
                    d_odd_event_node,
                    even_events_count,
                    d_partitions
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // compute the new partitions cutnet
            compute_partitions_cutnet(
                cfg,
                d_hedges,
                d_hedges_offsets,
                d_hedge_weights,
                d_partitions,
                num_hedges,
                num_parts,
                hedges_size,
                d_cutnet,
                stream,
                tid
            );

            // track the best partitioning found so far at this level
            {
                // launch configuration - update best partitions kernel
                int threads_per_block = 256;
                int num_threads_needed = num_nodes; // 1 thread per node
                int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
                // launch - update best partitions kernel
                LAUNCH(cfg) TID(tid) RUN << "update best partitions kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                update_best_partitions_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_partitions,
                    d_cutnet,
                    d_last_best_cutnet,
                    num_nodes,
                    d_last_best_partitions

                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // update the best cutnets per partitions pair found so far at this level
            thrust::transform(
                thrust_exec, t_last_best_cutnet, t_last_best_cutnet + num_parts / 2,
                t_cutnet, t_last_best_cutnet, thrust::minimum<float>{}
            );

            CUDA_CHECK(cudaFreeAsync(d_even_event_part, stream));
            CUDA_CHECK(cudaFreeAsync(d_even_event_score, stream));
            CUDA_CHECK(cudaFreeAsync(d_even_event_node, stream));
            CUDA_CHECK(cudaFreeAsync(d_odd_event_part, stream));
            CUDA_CHECK(cudaFreeAsync(d_odd_event_score, stream));
            CUDA_CHECK(cudaFreeAsync(d_odd_event_node, stream));
            CUDA_CHECK(cudaFreeAsync(d_part_even_event_offsets, stream));
            CUDA_CHECK(cudaFreeAsync(d_part_odd_event_offsets, stream));
        }

        // recover best partitions
        uint32_t* d_temp_partitions = d_last_best_partitions;
        d_last_best_partitions = d_partitions;
        d_partitions = d_temp_partitions;
        t_partitions = thrust::device_ptr<uint32_t>(d_partitions);

        CUDA_CHECK(cudaFreeAsync(d_cutnet, stream));
        CUDA_CHECK(cudaFreeAsync(d_last_best_cutnet, stream));
    }

    CUDA_CHECK(cudaFreeAsync(d_moves, stream));
    CUDA_CHECK(cudaFreeAsync(d_scores, stream));
    CUDA_CHECK(cudaFreeAsync(d_even_event_idx, stream));
    CUDA_CHECK(cudaFreeAsync(d_odd_event_idx, stream));
    CUDA_CHECK(cudaFreeAsync(d_last_best_partitions, stream));

    // one final bisection to go down to 1-element partitions
    split_partitions_rand(
        cfg,
        d_partitions,
        num_nodes,
        num_parts,
        gen,
        stream,
        tid
    );

    num_parts *= 2;

    uint32_t* d_order = nullptr; // order[idx] -> node currently in position idx
    uint32_t* d_ord_part = nullptr; // ord_part[idx] -> partition of node in order[idx]

    CUDA_CHECK(cudaMallocAsync(&d_order, num_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ord_part, num_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ord_part, d_partitions, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    thrust::device_ptr<uint32_t> t_order(d_order);
    thrust::device_ptr<uint32_t> t_ord_part(d_ord_part);
    thrust::sequence(thrust_exec, t_order, t_order + num_nodes);
    thrust::sort_by_key(thrust_exec, t_ord_part, t_ord_part + num_nodes, t_order); // this also sorts copy(d_partitions) into d_ord_part

    // fuse back partitions while internally reversing them as needed to "trap" strong connections locally inside partition pairs
    while (num_parts > 1) { // go back up the bisection tree
        INFO(cfg) std::cout TID(tid) << "Tree reorientation level " << level_idx << " number of partitions=" << num_parts << "\n";
        level_idx--;
        
        float* d_sibling_score = nullptr; // sibling_score[p] -> total connection strength between partition p and the sibling subtree of floor(p/2)
        CUDA_CHECK(cudaMallocAsync(&d_sibling_score, num_parts * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_sibling_score, 0x00, num_parts * sizeof(float), stream));

        // compute connection strength of each partition with its parent's sibling subtree
        {
            // launch configuration - sibling tree connection strength kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = num_nodes; // 1 warp per event
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - sibling tree connection strength kernel
            LAUNCH(cfg) TID(tid) RUN << "sibling tree connection strength kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            sibling_tree_connection_strength_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_hedges,
                d_hedges_offsets,
                d_touching,
                d_touching_offsets,
                d_hedge_weights,
                d_order,
                d_ord_part,
                d_partitions,
                num_nodes,
                d_sibling_score
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // refold p*2 and p*2+1 back into p
        num_parts /= 2;
        thrust::transform(
            thrust_exec,
            t_partitions, t_partitions + num_nodes, t_partitions,
            [] __device__ (uint32_t x) { return x >> 1; }
        );
        thrust::transform(
            thrust_exec,
            t_ord_part, t_ord_part + num_nodes, t_ord_part,
            [] __device__ (uint32_t x) { return x >> 1; }
        );

        bool* d_reverse = nullptr; // reverse[p/2] -> true if the subtree of p/2 (well, ex-p/2, since we already folded it back in p) needs to have its leaves-order reversed
        CUDA_CHECK(cudaMallocAsync(&d_reverse, num_parts * sizeof(bool), stream));
        {
            // launch configuration - flag reversals kernel
            int threads_per_block = 256;
            int num_threads_needed = num_parts; // 1 thread per (half) partition
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - flag reversals kernel
            LAUNCH(cfg) TID(tid) RUN << "flag reversals kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            flag_reversals_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_sibling_score,
                num_parts,
                d_reverse
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // build offset indices over ord_part
        uint32_t* d_ord_part_offsets = nullptr; // ord_part_offsets[p] -> first index of partition p in ord_part
        CUDA_CHECK(cudaMallocAsync(&d_ord_part_offsets, (num_parts + 1) * sizeof(uint32_t), stream));
        thrust::device_ptr<uint32_t> t_ord_part_offsets(d_ord_part_offsets);
        thrust::counting_iterator<uint32_t> ord_search_begin(0);
        // NOTE: the search was "made to work" by storing p/2 inside event_part-s, hence it is enough to search from 0 to num_parts/2
        thrust::lower_bound(
            thrust_exec,
            t_ord_part, t_ord_part + num_nodes,
            ord_search_begin, ord_search_begin + num_parts,
            t_ord_part_offsets
        );
        CUDA_CHECK(cudaMemcpyAsync(d_ord_part_offsets + num_parts, &num_nodes, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

        // apply the reversal of leaves/nodes inside each flagged subtree
        {
            // launch configuration - apply reversals kernel
            int threads_per_block = 256;
            int num_threads_needed = num_nodes; // 1 thread per (half) partition
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - apply reversals kernel
            LAUNCH(cfg) TID(tid) RUN << "apply reversals kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            apply_reversals_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_ord_part,
                d_ord_part_offsets,
                d_reverse,
                num_nodes,
                d_order
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        CUDA_CHECK(cudaFreeAsync(d_sibling_score, stream));
        CUDA_CHECK(cudaFreeAsync(d_reverse, stream));
        CUDA_CHECK(cudaFreeAsync(d_ord_part_offsets, stream));
    }

    // write d_order_idx as the reverse map of order
    uint32_t* d_order_idx = nullptr; // order_idx[node] -> position in the ordering for node

    CUDA_CHECK(cudaMallocAsync(&d_order_idx, num_nodes * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_order_idx(d_order_idx);
    thrust::scatter(thrust_exec, thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(num_nodes), t_order, t_order_idx);

    CUDA_CHECK(cudaFreeAsync(d_order, stream));
    CUDA_CHECK(cudaFreeAsync(d_ord_part, stream));
    CUDA_CHECK(cudaFreeAsync(d_partitions, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CURAND_CHECK(curandDestroyGenerator(gen));

    // =============================
    // measure 1D order locality
    // metric: width spanned by each hedge (lowest pin idx - to - highest pin idx) times its weight
    LOG(cfg) {
        float* d_hedge_span = nullptr; // hedge_span[hedge idx] -> max-pin-idx - min-pin-idx times the hedge's weight
        CUDA_CHECK(cudaMallocAsync(&d_hedge_span, num_hedges * sizeof(float), stream));
        {
            // launch configuration - measure sequence locality kernel
            int threads_per_block = 256;
            int num_threads_needed = num_hedges; // 1 thread per (half) partition
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - measure sequence locality kernel
            LAUNCH(cfg) TID(tid) RUN << "measure sequence locality kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            measure_sequence_locality_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_hedges,
                d_hedges_offsets,
                d_hedge_weights,
                d_order_idx,
                num_hedges,
                d_hedge_span
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        thrust::device_ptr<float> t_hedge_span(d_hedge_span);
        float tot_span = thrust::reduce(thrust_exec, t_hedge_span, t_hedge_span + num_hedges);
        CUDA_CHECK(cudaFreeAsync(d_hedge_span, stream));
        std::cout TID(tid) << "Initial sequence (1D) weighted locality: " << std::fixed << std::setprecision(3) << tot_span << "\n";
    }
    // =============================

    return d_order_idx;
}
