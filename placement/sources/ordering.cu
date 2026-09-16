#include <vector>
#include <climits>
#include <stdint.h>
#include <algorithm>

#include <cub/cub.cuh>

#include "thruster.cuh"

#include "runconfig_plc.hpp"

#include "utils.cuh"
#include "utils_plc.cuh"
#include "ordering.cuh"

// map every pin of every multi-start's view of the hypergraph to the partition that multi-start put it in
struct batch_pin_to_partition {
    const uint32_t* hedges;
    const uint32_t* partitions;
    const dim_t hedges_size;
    const uint32_t num_nodes;
    __host__ __device__ uint32_t operator()(const dim_t idx) const {
        const dim_t my_start = idx / hedges_size;
        return partitions[my_start * num_nodes + hedges[idx - my_start * hedges_size]];
    }
};

// segment bounds of the batched segmented sort: multi-start "b"'s hedge "h" starts at b*hedges_size + hedges_offsets[h]
struct batch_hedge_offset {
    const dim_t* hedges_offsets;
    const uint32_t num_hedges;
    const dim_t hedges_size;
    __host__ __device__ dim_t operator()(const uint32_t seg) const {
        const uint32_t my_start = seg / num_hedges;
        return my_start * hedges_size + hedges_offsets[seg - my_start * num_hedges];
    }
};

// the multi-start a batch-flat node belongs to, which at num_parts == 1 is also its composite partition
struct batch_start_of_node {
    const uint32_t num_nodes;
    __host__ __device__ uint32_t operator()(const uint32_t node) const {
        return node / num_nodes;
    }
};

// a batch-flat ordering position, reduced to a position local to its own multi-start
struct batch_local_position {
    const uint32_t num_nodes;
    __host__ __device__ uint32_t operator()(const uint32_t position) const {
        return position % num_nodes;
    }
};

// drop the trailing reduce_by_key group formed by the empty event slots, whose composite key is UINT32_MAX
struct batch_valid_part_pair {
    const uint32_t batch_part_pairs;
    __host__ __device__ bool operator()(const uint32_t part_pair) const {
        return part_pair < batch_part_pairs;
    }
};

// for each partition, randomly bisect it, mapping every partition id "p" to either "p*2" or "p*2+1"
void split_partitions_rand(
    const runconfig &cfg,
    uint32_t* d_partitions,
    uint32_t num_nodes,
    uint32_t num_parts,
    uint32_t batch_size,
    const std::vector<curandGenerator_t> &gens,
    const cudaStream_t stream,
    const int tid
) {
    auto thrust_exec = thrust::cuda::par.on(stream);
    thrust::device_ptr<uint32_t> t_partitions(d_partitions);

    const uint32_t batch_nodes = batch_size * num_nodes;
    const uint32_t batch_parts = batch_size * num_parts;

    uint32_t* d_original_idx = nullptr; // original_idx[i] -> idx of node currently in partition partitions[i]
    CUDA_CHECK(cudaMallocAsync(&d_original_idx, batch_nodes * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_original_idx(d_original_idx);
    thrust::sequence(thrust_exec, t_original_idx, t_original_idx + batch_nodes);

    uint32_t* d_partitions_cpy = nullptr; // auxiliary copy of current partitions for sorting and scattering
    CUDA_CHECK(cudaMallocAsync(&d_partitions_cpy, batch_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_partitions_cpy, d_partitions, batch_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    thrust::device_ptr<uint32_t> t_partitions_cpy(d_partitions_cpy);

    // generate one random uint32 per element, seeded
    // NOTE: one generator per multi-start, so a multi-start draws the very same sequence whatever it is batched with
    uint32_t* d_rand_keys = nullptr; // rand_keys[i] -> random tie-breaking key for the node currently in partition partitions[i]
    CUDA_CHECK(cudaMallocAsync(&d_rand_keys, batch_nodes * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_rand_keys(d_rand_keys);
    for (uint32_t start = 0; start < batch_size; start++) {
        CURAND_CHECK(curandSetStream(gens[start], stream));
        CURAND_CHECK(curandGenerate(gens[start], d_rand_keys + start * num_nodes, num_nodes));
    }

    // sort by (partition, random), carrying along the original indices
    // => now partitions_cpy is grouped by composite "p", with random order inside each group
    auto part_rand_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_partitions_cpy, t_rand_keys));
    auto part_rand_key_end = part_rand_key_begin + batch_nodes;
    thrust::sort_by_key(thrust_exec, part_rand_key_begin, part_rand_key_end, t_original_idx);

    // build offset indices over reordered partitions
    uint32_t* d_part_offsets = nullptr; // part_offsets[p] -> first index of partition p in partitions_cpy
    CUDA_CHECK(cudaMallocAsync(&d_part_offsets, (batch_parts + 1) * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_part_offsets(d_part_offsets);
    thrust::counting_iterator<uint32_t> search_begin(0);
    thrust::lower_bound(
        thrust_exec,
        t_partitions_cpy, t_partitions_cpy + batch_nodes,
        search_begin, search_begin + batch_parts,
        t_part_offsets
    );
    CUDA_CHECK(cudaMemcpyAsync(d_part_offsets + batch_parts, &batch_nodes, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    // split each partition in half:
    // - inside each partition, original node indices are not randomly ordered
    // - take the lower half of those indices and map it to p*2, take the upper half and map it to p*2+1
    {
        // launch configuration - split partitions kernel
        int threads_per_block = 256;
        int num_threads_needed = batch_nodes; // 1 thread per node
        int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - split partitions kernel
        LAUNCH(cfg) TID(tid) RUN << "split partitions kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        split_partitions_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_part_offsets,
            num_nodes,
            batch_size,
            d_partitions_cpy
        );
        DBG(cfg) CUDA_CHECK(cudaGetLastError());
        DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // undo the sort - scatter back to updated partitions to their original idxs
    thrust::scatter(
        thrust_exec,
        t_partitions_cpy,
        t_partitions_cpy + batch_nodes,
        t_original_idx,
        t_partitions
    );

    CUDA_CHECK(cudaFreeAsync(d_original_idx, stream));
    CUDA_CHECK(cudaFreeAsync(d_partitions_cpy, stream));
    CUDA_CHECK(cudaFreeAsync(d_rand_keys, stream));
    CUDA_CHECK(cudaFreeAsync(d_part_offsets, stream));
}

void compute_partitions_cutnet(
    const runconfig &cfg,
    const uint32_t* d_hedges,
    const dim_t* d_hedges_offsets,
    const float* d_hedge_weights,
    const uint32_t* d_partitions,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_parts,
    const uint32_t batch_size,
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
    * - sort events by parent partition id, and do a segmented reduce within each parent id
    *   => that yields each parent partition's weighted minority pin-cut across its bisection
    *
    * TODO: switch from the segmented sort to deduplicating part_pins in shared-memory for each hedge, and directly yielding the count of unique partitions per hedge
    *       from the count and offsets (aka, flags) you then allocate the buffer and repeat the deduplication to write final split costs
    */

    const dim_t batch_pins = static_cast<dim_t>(batch_size) * hedges_size;
    const uint32_t batch_hedges = batch_size * num_hedges;
    const uint32_t batch_part_pairs = batch_size * (num_parts / 2);

    // CUB's segmented sort counts items with an int
    if (batch_pins > static_cast<dim_t>(INT_MAX)) {
        ERR(cfg) std::cerr TID(tid) << "ABORTING: cutnet segmented sort over " << batch_pins << " pins exceeds INT_MAX, lower the batch size !!\n";
        abort();
    }

    uint32_t* d_part_pins = nullptr; // part_pins[start*hedges_size + hedges_offsets[hedge idx] + pin idx] -> partition the pin is in
    CUDA_CHECK(cudaMallocAsync(&d_part_pins, batch_pins * sizeof(uint32_t), stream));

    thrust::device_ptr<uint32_t> t_part_pins(d_part_pins);
    thrust::device_ptr<float> t_cutnet(d_cutnet);

    // initialize every partition pair as "fully trapped"; pairs that generate events will overwrite their true split cost
    thrust::fill(thrust_exec, t_cutnet, t_cutnet + batch_part_pairs, 0.0f);

    // map pins to their partition -> each multi-start maps the shared pin sequence through its own partitioning
    thrust::transform(
        thrust_exec,
        thrust::counting_iterator<dim_t>(0), thrust::counting_iterator<dim_t>(batch_pins),
        t_part_pins,
        batch_pin_to_partition{d_hedges, d_partitions, hedges_size, num_nodes}
    );

    // segmented sort of part_pins (using the segments from hedges, one set of them per multi-start)
    auto batch_hedges_offsets = thrust::make_transform_iterator(
        thrust::counting_iterator<uint32_t>(0),
        batch_hedge_offset{d_hedges_offsets, num_hedges, hedges_size}
    );
    uint32_t* d_part_pins_buffer = nullptr; // CUB segmented sort buffer
    CUDA_CHECK(cudaMallocAsync(&d_part_pins_buffer, batch_pins * sizeof(uint32_t), stream));
    cub::DoubleBuffer<uint32_t> c_part_pins_double_buffer(d_part_pins, d_part_pins_buffer);
    void* c_part_pins_storage = nullptr;
    size_t c_part_pins_storage_bytes = 0;
    // NOTE: 'DeviceSegmentedSort', and not 'DeviceSegmentedRadixSort', because hedges are many and short
    // => it bins segments by size and picks a per-bin algorithm, instead of paying a full radix pass for every one of them
    cub::DeviceSegmentedSort::SortKeys(
        c_part_pins_storage, c_part_pins_storage_bytes, c_part_pins_double_buffer,
        batch_pins, batch_hedges, batch_hedges_offsets, batch_hedges_offsets + 1,
        stream
    );
    CUB(cfg) std::cout TID(tid) << "CUB segmented sort requiring " << std::fixed << std::setprecision(3) << (float)(batch_pins * sizeof(uint32_t)) / (1 << 30)
        << " GB of pong-buffer and " << std::fixed << std::setprecision(3) << ((float)c_part_pins_storage_bytes) / (1 << 20)
        << " MB of temporary storage ...\n";
    CUDA_CHECK(cudaMallocAsync(&c_part_pins_storage, c_part_pins_storage_bytes, stream));
    cub::DeviceSegmentedSort::SortKeys(
        c_part_pins_storage, c_part_pins_storage_bytes, c_part_pins_double_buffer,
        batch_pins, batch_hedges, batch_hedges_offsets, batch_hedges_offsets + 1,
        stream
    );
    DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
    if (c_part_pins_double_buffer.Current() != d_part_pins) {
        uint32_t* tmp = d_part_pins_buffer;
        d_part_pins_buffer = d_part_pins;
        d_part_pins = tmp;
    }
    CUDA_CHECK(cudaFreeAsync(d_part_pins_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(c_part_pins_storage, stream));

    // NOTE: 32 bits are enough, these end up holding event offsets, and events never outnumber pins, which the check above caps at INT_MAX
    uint32_t* d_flags = nullptr; // event_weight[idx] -> weight of the hedge being cut in event idx
    CUDA_CHECK(cudaMallocAsync(&d_flags, (batch_pins + 1) * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_flags, 0x00, (batch_pins + 1) * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_flags(d_flags);
    {
        // launch configuration - flag cutnet events kernel
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = batch_hedges; // 1 warp per hedge
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - flag cutnet events kernel
        LAUNCH(cfg) TID(tid) RUN << "flag cutnet events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        flag_cutnet_events_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_part_pins,
            d_hedges_offsets,
            num_hedges,
            batch_size,
            hedges_size,
            d_flags
        );
        DBG(cfg) CUDA_CHECK(cudaGetLastError());
        DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // exclusive prefix sum of flags, then extract the last value (total sum) as the events count
    // NOTE: the scan is over integers, so batching it whole leaves every multi-start's offsets unchanged
    thrust::exclusive_scan(thrust_exec, t_flags, t_flags + batch_pins + 1, t_flags);
    uint32_t events_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&events_count, d_flags + batch_pins, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float* d_event_weight = nullptr; // event_weight[idx] -> weight of the hedge being cut in event idx
    uint32_t* d_event_part = nullptr; // event_part[idx] -> composite partition/2 affected by event idx

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
            int num_warps_needed = batch_hedges; // 1 warp per hedge
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - cutnet event generation kernel
            LAUNCH(cfg) TID(tid) RUN << "cutnet event generation kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            cutnet_event_generation_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_part_pins,
                d_hedges_offsets,
                d_hedge_weights,
                d_flags,
                num_hedges,
                batch_size,
                hedges_size,
                d_event_weight,
                d_event_part
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // sort event_weight and event_part both according to event_part
        // => partitions are composite, so this single sort already groups every multi-start's events apart
        thrust::sort_by_key(thrust_exec, t_event_part, t_event_part + events_count, t_event_weight);

        // reduce-sum each segment of event_weight with the same event_part value and store the result in cutnet[t_event_part[.]]
        // => although the buffer is still called "cutnet", it now stores weighted minority pin-cut
        uint32_t* d_unique_event_part = nullptr; // unique_event_part[idx] -> idx-th composite partition/2 that generated at least one cutnet event
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
    }

    CUDA_CHECK(cudaFreeAsync(d_part_pins, stream));
    CUDA_CHECK(cudaFreeAsync(d_flags, stream));
    DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
}

// return a high-locality, seeded 1D ordering of nodes
uint32_t* locality_ordering(
    const runconfig &cfg,
    const uint32_t num_nodes,
    const uint32_t batch_size,
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

    const uint32_t batch_nodes = batch_size * num_nodes;

    // one generator per multi-start, so a multi-start draws the very same sequence whatever it is batched with
    std::vector<curandGenerator_t> gens(batch_size);
    for (uint32_t start = 0; start < batch_size; start++) {
        CURAND_CHECK(curandCreateGenerator(&gens[start], CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gens[start], seed + start));
        CURAND_CHECK(curandSetStream(gens[start], stream));
    }

    uint32_t* d_partitions = nullptr; // partitions[node idx] -> current composite partition (of bypartitions) the node is in

    CUDA_CHECK(cudaMallocAsync(&d_partitions, batch_nodes * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_partitions(d_partitions);
    // everyone starts in the same partition, which at num_parts == 1 makes the composite id coincide with the multi-start's own idx
    thrust::transform(
        thrust_exec,
        thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(batch_nodes),
        t_partitions,
        batch_start_of_node{num_nodes}
    );

    uint32_t num_parts = 1u;

    bool* d_moves = nullptr; // move[node idx] -> false if the node doesn't want to move, true if the node would like to switch partition p*2->p*2+1 or p*2+1->p*2
    float* d_scores = nullptr; // score[node idx] -> connectivity gain for the above move (even not moving is done with a "gain")
    uint8_t* d_active = nullptr; // active[start] -> 0 once that multi-start stopped improving at the current level

    CUDA_CHECK(cudaMallocAsync(&d_moves, batch_nodes * sizeof(bool), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scores, batch_nodes * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_active, batch_size * sizeof(uint8_t), stream));

    // NOTE: move events are not compacted, node idx "i" owns event slot "i" in one of the two lists
    // => the slots it does not own carry UINT32_MAX as partition, so they sort behind every real event of their own multi-start
    // => no device-to-host readback is ever needed to size these, nor the launches consuming them
    uint32_t* d_even_event_part = nullptr; // part[idx] -> composite src partition / 2 for the idx-th move (partition being even)
    float* d_even_event_score = nullptr; // score[idx] -> gain for the idx-th move
    uint32_t* d_even_event_node = nullptr; // node[idx] -> node moved in the idx-th move
    uint32_t* d_odd_event_part = nullptr; // part[idx] -> composite (src partition - 1) / 2 ... (partition being odd)
    float* d_odd_event_score = nullptr; // score[idx] -> ...
    uint32_t* d_odd_event_node = nullptr; // node[idx] -> ...
    CUDA_CHECK(cudaMallocAsync(&d_even_event_part, batch_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_even_event_score, batch_nodes * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_even_event_node, batch_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_odd_event_part, batch_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_odd_event_score, batch_nodes * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_odd_event_node, batch_nodes * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_even_event_part(d_even_event_part);
    thrust::device_ptr<float> t_even_event_score(d_even_event_score);
    thrust::device_ptr<uint32_t> t_even_event_node(d_even_event_node);
    thrust::device_ptr<uint32_t> t_odd_event_part(d_odd_event_part);
    thrust::device_ptr<float> t_odd_event_score(d_odd_event_score);
    thrust::device_ptr<uint32_t> t_odd_event_node(d_odd_event_node);

    // one slot past the nodes is the scratch bin the empty event slots scatter their rank into
    uint32_t* d_even_ranks = nullptr; // ranks[node idx] -> even event index for node idx (UINT32_MAX if no event)
    uint32_t* d_odd_ranks = nullptr; // ranks[node idx] -> ...
    CUDA_CHECK(cudaMallocAsync(&d_even_ranks, (batch_nodes + 1) * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_odd_ranks, (batch_nodes + 1) * sizeof(uint32_t), stream));

    // num_parts/2 never exceeds num_nodes, so one slot per node per multi-start is always enough
    uint32_t* d_apply_up_to = nullptr; // apply_up_to[p/2] -> last absolute event idx to apply for composite partitions p and p+1
    CUDA_CHECK(cudaMallocAsync(&d_apply_up_to, batch_nodes * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_apply_up_to(d_apply_up_to);

    // IDEA:
    // - initialize this on each level from current partitions
    // - after label prop, compute the new split cost for each pair of partitions
    // - iff a partitions pair's split cost improved, copy over here the new partition ids for the nodes of that pair of partitions
    // - before going to the next level, make this the actual partitioning
    uint32_t* d_last_best_partitions = nullptr; // last_best_partitions [node idx] -> last best composite partition the node was in
    CUDA_CHECK(cudaMallocAsync(&d_last_best_partitions, batch_nodes * sizeof(uint32_t), stream));

    // pinned, so the periodic activity check does not stall the host on a pageable copy
    uint8_t* h_active = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_active, batch_size * sizeof(uint8_t)));

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
            batch_size,
            gens,
            stream,
            tid
        );
        num_parts *= 2;
        const uint32_t batch_part_pairs = batch_size * (num_parts / 2);

        CUDA_CHECK(cudaMemcpyAsync(d_last_best_partitions, d_partitions, batch_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));

        float* d_cutnet = nullptr; // cutnet[p/2] -> weighted minority pin-cut of the bisection of composite partition p/2
        float* d_last_best_cutnet = nullptr; // last_best_cutnet[p/2] -> best such cost seen so far at this level
        CUDA_CHECK(cudaMallocAsync(&d_cutnet, batch_part_pairs * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&d_last_best_cutnet, batch_part_pairs * sizeof(float), stream));
        thrust::device_ptr<float> t_cutnet(d_cutnet);
        thrust::device_ptr<float> t_last_best_cutnet(d_last_best_cutnet);

        // baseline split cost of the freshly randomized bisection
        compute_partitions_cutnet(
            cfg,
            d_hedges,
            d_hedges_offsets,
            d_hedge_weights,
            d_partitions,
            num_hedges,
            num_nodes,
            num_parts,
            batch_size,
            hedges_size,
            d_last_best_cutnet,
            stream,
            tid
        );

        // build offset indices over reordered events per partition
        uint32_t* d_part_even_event_offsets = nullptr; // part_even_event_offsets[p] -> first index of composite partition p*2 in even_event_part
        uint32_t* d_part_odd_event_offsets = nullptr; // part_odd_event_offsets[p] -> first index of composite partition p*2+1 in odd_event_part
        CUDA_CHECK(cudaMallocAsync(&d_part_even_event_offsets, (batch_part_pairs + 1) * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMallocAsync(&d_part_odd_event_offsets, (batch_part_pairs + 1) * sizeof(uint32_t), stream));
        thrust::device_ptr<uint32_t> t_part_even_event_offsets(d_part_even_event_offsets);
        thrust::device_ptr<uint32_t> t_part_odd_event_offsets(d_part_odd_event_offsets);

        // every multi-start re-enters each level active, the label propagation retires them as they run out of improving moves
        CUDA_CHECK(cudaMemsetAsync(d_active, 0x01, batch_size * sizeof(uint8_t), stream));

        for (uint32_t lp_repeat = 0u; lp_repeat < cfg.labelprop_repeats; lp_repeat++) {
            // compute gains (and moves) in-isolation
            // NOTE: no need to init. "d_moves" and "d_scores", they are overwritten anyway
            {
                // launch configuration - label propagation kernel
                int threads_per_block = 128; // 128/32 -> 4 warps per block
                int warps_per_block = threads_per_block / WARP_SIZE;
                int num_warps_needed = batch_nodes; // 1 warp per node
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
                    batch_size,
                    d_active,
                    d_moves,
                    d_scores
                );
                DBG(cfg) CUDA_CHECK(cudaGetLastError());
                DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // build move events (partition, score, node)
            {
                // launch configuration - label move events kernel
                int threads_per_block = 256;
                int num_threads_needed = batch_nodes; // 1 thread per node
                int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
                // launch - label move events kernel
                LAUNCH(cfg) TID(tid) RUN << "label move events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                label_move_events_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_moves,
                    d_scores,
                    d_partitions,
                    num_nodes,
                    batch_size,
                    d_active,
                    d_even_event_part,
                    d_even_event_score,
                    d_even_event_node,
                    d_odd_event_part,
                    d_odd_event_score,
                    d_odd_event_node
                );
                DBG(cfg) CUDA_CHECK(cudaGetLastError());
                DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // sort events by (partition, score, node)
            // => partitions are composite, so a single sort already groups every multi-start's events apart, empty slots trailing behind
            auto move_even_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_even_event_part, t_even_event_score, t_even_event_node));
            thrust::sort(thrust_exec, move_even_events_key_begin, move_even_events_key_begin + batch_nodes);
            // |
            auto move_odd_events_key_begin = thrust::make_zip_iterator(thrust::make_tuple(t_odd_event_part, t_odd_event_score, t_odd_event_node));
            thrust::sort(thrust_exec, move_odd_events_key_begin, move_odd_events_key_begin + batch_nodes);

            // NOTE: the search was "made to work" by storing p/2 inside event_part-s, hence it is enough to search from 0 to batch_part_pairs
            // => searching one past the last pair yields the total count of real events, so no boundary needs writing by hand
            thrust::counting_iterator<uint32_t> even_search_begin(0);
            thrust::lower_bound(
                thrust_exec,
                t_even_event_part, t_even_event_part + batch_nodes,
                even_search_begin, even_search_begin + batch_part_pairs + 1,
                t_part_even_event_offsets
            );
            // |
            thrust::counting_iterator<uint32_t> odd_search_begin(0);
            thrust::lower_bound(
                thrust_exec,
                t_odd_event_part, t_odd_event_part + batch_nodes,
                odd_search_begin, odd_search_begin + batch_part_pairs + 1,
                t_part_odd_event_offsets
            );

            // build the reverse map: ranks[node] -> event-idx (if any) of node - in other words this scatter does "ranks[event_node[i]] = i"
            // => empty slots all point at the scratch bin one past the nodes, so they write there and are ignored
            CUDA_CHECK(cudaMemsetAsync(d_even_ranks, 0xFF, (batch_nodes + 1) * sizeof(uint32_t), stream));
            thrust::scatter(thrust_exec, thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(batch_nodes), t_even_event_node, thrust::device_ptr<uint32_t>(d_even_ranks));
            CUDA_CHECK(cudaMemsetAsync(d_odd_ranks, 0xFF, (batch_nodes + 1) * sizeof(uint32_t), stream));
            thrust::scatter(thrust_exec, thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(batch_nodes), t_odd_event_node, thrust::device_ptr<uint32_t>(d_odd_ranks));

            // update gains in-sequence
            // assume moves are done in pairs => re-compute the pair's gain in-sequence, assuming all prior pairs already swapped
            // => already accumulate the two scores on the "even" segment's event (only up to the length of the smallest events segment between even and odd)
            CUDA_CHECK(cudaMemsetAsync(d_even_event_score, 0x00, batch_nodes * sizeof(float), stream));
            {
                // launch configuration - label cascade kernel
                int threads_per_block = 128; // 128/32 -> 4 warps per block
                int warps_per_block = threads_per_block / WARP_SIZE;
                int num_warps_needed = 2u * batch_nodes; // 1 warp per event slot, even ones then odd ones
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
                    d_even_ranks,
                    d_odd_ranks,
                    d_even_event_node,
                    d_odd_event_node,
                    num_nodes,
                    batch_size,
                    d_even_event_score
                );
                DBG(cfg) CUDA_CHECK(cudaGetLastError());
                DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // inclusive scan inside each key (= composite partition) on the even event scores => for each event we get the cumulative gain up to that point in the partition's move sequence
            thrust::inclusive_scan_by_key(thrust_exec, t_even_event_part, t_even_event_part + batch_nodes, t_even_event_score, t_even_event_score);
            // extract the maximum idx (relative to the start of the overall array) for every partition's pair
            auto event_score_pair = thrust::make_zip_iterator(thrust::make_tuple(t_even_event_score, thrust::counting_iterator<uint32_t>(0)));
            CUDA_CHECK(cudaMemsetAsync(d_apply_up_to, 0xFF, batch_part_pairs * sizeof(uint32_t), stream));
            uint32_t* d_argmax_event_part = nullptr;
            uint32_t* d_argmax_event_idx = nullptr;
            CUDA_CHECK(cudaMallocAsync(&d_argmax_event_part, (batch_part_pairs + 1) * sizeof(uint32_t), stream));
            CUDA_CHECK(cudaMallocAsync(&d_argmax_event_idx, (batch_part_pairs + 1) * sizeof(uint32_t), stream));
            thrust::device_ptr<uint32_t> t_argmax_event_part(d_argmax_event_part);
            thrust::device_ptr<uint32_t> t_argmax_event_idx(d_argmax_event_idx);
            auto d_event_argmax = thrust::make_transform_output_iterator(
                t_argmax_event_idx, // discard the "max" part of the "argmax" return tuple
                [] __device__ (auto x) { return (thrust::get<0>(x) <= 0.0f) ? UINT32_MAX : thrust::get<1>(x); }
            );
            auto argmax_end = thrust::reduce_by_key(
                thrust_exec, t_even_event_part, t_even_event_part + batch_nodes, event_score_pair,
                t_argmax_event_part, d_event_argmax, thrust::equal_to<uint32_t>{},
                [] __device__ (auto a, auto b) { return (thrust::get<0>(b) > thrust::get<0>(a)) ? b : a; }
            );
            const uint32_t argmax_count = thrust::get<0>(argmax_end) - t_argmax_event_part;
            // reduce_by_key emits compact groups; scatter by composite partition id to build the apply_up_to map expected by the kernel
            // => the trailing group of empty slots carries UINT32_MAX as key, so it scatters past the real pairs and is dropped
            thrust::scatter_if(
                thrust_exec, t_argmax_event_idx, t_argmax_event_idx + argmax_count, t_argmax_event_part,
                thrust::make_transform_iterator(t_argmax_event_part, batch_valid_part_pair{batch_part_pairs}),
                t_apply_up_to
            );
            CUDA_CHECK(cudaFreeAsync(d_argmax_event_part, stream));
            CUDA_CHECK(cudaFreeAsync(d_argmax_event_idx, stream));

            // retire the multi-starts with no strictly improving balanced prefix left
            {
                // launch configuration - labelprop activity kernel
                int threads_per_block = 256;
                int blocks = batch_size; // 1 block per multi-start
                // launch - labelprop activity kernel
                LAUNCH(cfg) TID(tid) RUN << "labelprop activity kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                labelprop_activity_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_apply_up_to,
                    num_parts / 2,
                    d_active
                );
                DBG(cfg) CUDA_CHECK(cudaGetLastError());
                DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // apply pairs of improving moves
            // add together the gain of equi-ranked moves between bisected partitions as the gain of the pair to swap
            {
                // launch configuration - apply move events kernel
                int threads_per_block = 256;
                int num_threads_needed = batch_nodes; // 1 thread per event slot
                int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
                // launch - apply move events kernel
                LAUNCH(cfg) TID(tid) RUN << "apply move events kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                apply_move_events_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_apply_up_to,
                    d_even_event_part,
                    d_even_event_node,
                    d_part_even_event_offsets,
                    d_part_odd_event_offsets,
                    d_odd_event_node,
                    num_nodes,
                    batch_size,
                    d_partitions
                );
                DBG(cfg) CUDA_CHECK(cudaGetLastError());
                DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // compute the new partitions split cost
            compute_partitions_cutnet(
                cfg,
                d_hedges,
                d_hedges_offsets,
                d_hedge_weights,
                d_partitions,
                num_hedges,
                num_nodes,
                num_parts,
                batch_size,
                hedges_size,
                d_cutnet,
                stream,
                tid
            );

            // track the best partitioning found so far at this level
            {
                // launch configuration - update best partitions kernel
                int threads_per_block = 256;
                int num_threads_needed = batch_nodes; // 1 thread per node
                int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
                // launch - update best partitions kernel
                LAUNCH(cfg) TID(tid) RUN << "update best partitions kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
                update_best_partitions_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    d_partitions,
                    d_cutnet,
                    d_last_best_cutnet,
                    num_nodes,
                    batch_size,
                    d_last_best_partitions
                );
                DBG(cfg) CUDA_CHECK(cudaGetLastError());
                DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // update the best split costs per partitions pair found so far at this level
            thrust::transform(
                thrust_exec, t_last_best_cutnet, t_last_best_cutnet + batch_part_pairs,
                t_cutnet, t_last_best_cutnet, thrust::minimum<float>{}
            );

            // multi-starts retire themselves on the device, so only check on the host once per repeat
            CUDA_CHECK(cudaMemcpyAsync(h_active, d_active, batch_size * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint32_t active_count = 0u;
            for (uint32_t start = 0; start < batch_size; start++) active_count += h_active[start];
            INFO(cfg) std::cout TID(tid) << "Label propagation on level " << level_idx << " repeat " << lp_repeat << " (multi-starts still improving=" << active_count << "/" << batch_size << ")\n";
            if (active_count == 0u) {
                INFO(cfg) std::cout TID(tid) << "Stopping label propagation on level " << level_idx << " repeat " << lp_repeat << " with no strictly improving balanced prefix left\n";
                break;
            }
        }

        // recover best partitions
        uint32_t* d_temp_partitions = d_last_best_partitions;
        d_last_best_partitions = d_partitions;
        d_partitions = d_temp_partitions;
        t_partitions = thrust::device_ptr<uint32_t>(d_partitions);

        CUDA_CHECK(cudaFreeAsync(d_cutnet, stream));
        CUDA_CHECK(cudaFreeAsync(d_last_best_cutnet, stream));
        CUDA_CHECK(cudaFreeAsync(d_part_even_event_offsets, stream));
        CUDA_CHECK(cudaFreeAsync(d_part_odd_event_offsets, stream));
    }

    CUDA_CHECK(cudaFreeHost(h_active));
    CUDA_CHECK(cudaFreeAsync(d_moves, stream));
    CUDA_CHECK(cudaFreeAsync(d_scores, stream));
    CUDA_CHECK(cudaFreeAsync(d_active, stream));
    CUDA_CHECK(cudaFreeAsync(d_even_event_part, stream));
    CUDA_CHECK(cudaFreeAsync(d_even_event_score, stream));
    CUDA_CHECK(cudaFreeAsync(d_even_event_node, stream));
    CUDA_CHECK(cudaFreeAsync(d_odd_event_part, stream));
    CUDA_CHECK(cudaFreeAsync(d_odd_event_score, stream));
    CUDA_CHECK(cudaFreeAsync(d_odd_event_node, stream));
    CUDA_CHECK(cudaFreeAsync(d_even_ranks, stream));
    CUDA_CHECK(cudaFreeAsync(d_odd_ranks, stream));
    CUDA_CHECK(cudaFreeAsync(d_apply_up_to, stream));
    CUDA_CHECK(cudaFreeAsync(d_last_best_partitions, stream));

    // one final bisection to go down to 1-element partitions
    split_partitions_rand(
        cfg,
        d_partitions,
        num_nodes,
        num_parts,
        batch_size,
        gens,
        stream,
        tid
    );
    num_parts *= 2;

    uint32_t* d_order = nullptr; // order[idx] -> batch-flat node currently in position idx
    uint32_t* d_ord_part = nullptr; // ord_part[idx] -> composite partition of node in order[idx]

    CUDA_CHECK(cudaMallocAsync(&d_order, batch_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ord_part, batch_nodes * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ord_part, d_partitions, batch_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    thrust::device_ptr<uint32_t> t_order(d_order);
    thrust::device_ptr<uint32_t> t_ord_part(d_ord_part);
    thrust::sequence(thrust_exec, t_order, t_order + batch_nodes);
    thrust::sort_by_key(thrust_exec, t_ord_part, t_ord_part + batch_nodes, t_order); // this also sorts copy(d_partitions) into d_ord_part

    // fuse back partitions while internally reversing them as needed to "trap" strong connections locally inside partition pairs
    while (num_parts > 2) { // go back up the bisection tree
        INFO(cfg) std::cout TID(tid) << "Tree reorientation level " << level_idx << " number of partitions=" << num_parts << "\n";
        level_idx--;

        float* d_sibling_score = nullptr; // sibling_score[p] -> total connection strength between composite partition p and the sibling subtree of floor(p/2)
        CUDA_CHECK(cudaMallocAsync(&d_sibling_score, batch_size * num_parts * sizeof(float), stream));
        float* d_slot_score = nullptr; // slot_score[idx] -> strength contributed by the node in ordering slot idx, towards its partition
        CUDA_CHECK(cudaMallocAsync(&d_slot_score, batch_nodes * sizeof(float), stream));

        // build offset indices over ord_part before the fold, so every partition's slots form one contiguous segment
        uint32_t* d_slot_part_offsets = nullptr; // slot_part_offsets[p] -> first ordering slot of composite partition p
        CUDA_CHECK(cudaMallocAsync(&d_slot_part_offsets, (batch_size * num_parts + 1) * sizeof(uint32_t), stream));
        thrust::device_ptr<uint32_t> t_slot_part_offsets(d_slot_part_offsets);
        thrust::counting_iterator<uint32_t> slot_search_begin(0);
        thrust::lower_bound(
            thrust_exec,
            t_ord_part, t_ord_part + batch_nodes,
            slot_search_begin, slot_search_begin + batch_size * num_parts + 1,
            t_slot_part_offsets
        );

        // compute connection strength of each partition with its parent's sibling subtree
        {
            // launch configuration - sibling tree connection strength kernel
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = batch_nodes; // 1 warp per event
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
                batch_size,
                d_slot_score
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // sum each partition's slot scores
        // NOTE: a segmented sum, and not an atomicAdd inside the kernel, so the reduction order - and with it the float rounding - is fixed
        // => CUB decomposes every segment the same way whatever the segment count, so a multi-start's scores do not depend on its batch either
        void* c_sibling_storage = nullptr;
        size_t c_sibling_storage_bytes = 0;
        CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
            c_sibling_storage, c_sibling_storage_bytes,
            d_slot_score, d_sibling_score,
            batch_size * num_parts, d_slot_part_offsets, d_slot_part_offsets + 1,
            stream
        ));
        CUDA_CHECK(cudaMallocAsync(&c_sibling_storage, c_sibling_storage_bytes, stream));
        CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
            c_sibling_storage, c_sibling_storage_bytes,
            d_slot_score, d_sibling_score,
            batch_size * num_parts, d_slot_part_offsets, d_slot_part_offsets + 1,
            stream
        ));
        CUDA_CHECK(cudaFreeAsync(c_sibling_storage, stream));

        // refold p*2 and p*2+1 back into p
        // => the composite id folds along with it, since "b*num_parts + p >> 1" is "b*(num_parts/2) + p/2" for even num_parts
        num_parts /= 2;
        thrust::transform(
            thrust_exec,
            t_partitions, t_partitions + batch_nodes, t_partitions,
            [] __device__ (uint32_t x) { return x >> 1; }
        );
        thrust::transform(
            thrust_exec,
            t_ord_part, t_ord_part + batch_nodes, t_ord_part,
            [] __device__ (uint32_t x) { return x >> 1; }
        );

        bool* d_reverse = nullptr; // reverse[p/2] -> true if the subtree of p/2 (well, ex-p/2, since we already folded it back in p) needs to have its leaves-order reversed
        CUDA_CHECK(cudaMallocAsync(&d_reverse, batch_size * num_parts * sizeof(bool), stream));
        {
            // launch configuration - flag reversals kernel
            int threads_per_block = 256;
            int num_threads_needed = batch_size * num_parts; // 1 thread per (half) partition
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - flag reversals kernel
            LAUNCH(cfg) TID(tid) RUN << "flag reversals kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            flag_reversals_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_sibling_score,
                num_parts,
                batch_size,
                d_reverse
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // build offset indices over ord_part
        uint32_t* d_ord_part_offsets = nullptr; // ord_part_offsets[p] -> first index of composite partition p in ord_part
        CUDA_CHECK(cudaMallocAsync(&d_ord_part_offsets, (batch_size * num_parts + 1) * sizeof(uint32_t), stream));
        thrust::device_ptr<uint32_t> t_ord_part_offsets(d_ord_part_offsets);
        thrust::counting_iterator<uint32_t> ord_search_begin(0);
        // NOTE: searching one past the last composite partition yields the total node count, so no boundary needs writing by hand
        thrust::lower_bound(
            thrust_exec,
            t_ord_part, t_ord_part + batch_nodes,
            ord_search_begin, ord_search_begin + batch_size * num_parts + 1,
            t_ord_part_offsets
        );

        // apply the reversal of leaves/nodes inside each flagged subtree
        {
            // launch configuration - apply reversals kernel
            int threads_per_block = 256;
            int num_threads_needed = batch_nodes; // 1 thread per (half) partition
            int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
            // launch - apply reversals kernel
            LAUNCH(cfg) TID(tid) RUN << "apply reversals kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            apply_reversals_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_ord_part,
                d_ord_part_offsets,
                d_reverse,
                batch_nodes,
                d_order
            );
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        CUDA_CHECK(cudaFreeAsync(d_sibling_score, stream));
        CUDA_CHECK(cudaFreeAsync(d_slot_score, stream));
        CUDA_CHECK(cudaFreeAsync(d_slot_part_offsets, stream));
        CUDA_CHECK(cudaFreeAsync(d_reverse, stream));
        CUDA_CHECK(cudaFreeAsync(d_ord_part_offsets, stream));
    }

    // write d_order_idx as the reverse map of order
    // => positions are made local to their own multi-start, so they index the shared 1D-to-(N)D map directly
    uint32_t* d_order_idx = nullptr; // order_idx[node] -> position in its multi-start's ordering for node

    CUDA_CHECK(cudaMallocAsync(&d_order_idx, batch_nodes * sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> t_order_idx(d_order_idx);
    auto local_positions = thrust::make_transform_iterator(thrust::counting_iterator<uint32_t>(0), batch_local_position{num_nodes});
    thrust::scatter(thrust_exec, local_positions, local_positions + batch_nodes, t_order, t_order_idx);

    CUDA_CHECK(cudaFreeAsync(d_order, stream));
    CUDA_CHECK(cudaFreeAsync(d_ord_part, stream));
    CUDA_CHECK(cudaFreeAsync(d_partitions, stream));
    DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));

    for (uint32_t start = 0; start < batch_size; start++)
        CURAND_CHECK(curandDestroyGenerator(gens[start]));

    // =============================
    // measure 1D order locality
    // metric: width spanned by each hedge (lowest pin idx - to - highest pin idx) times its weight
    // NOTE: only the first multi-start of the batch is measured, its order_idx slice already holds local positions
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
            DBG(cfg) CUDA_CHECK(cudaGetLastError());
            DBG(cfg) CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        thrust::device_ptr<float> t_hedge_span(d_hedge_span);
        float tot_span = thrust::reduce(thrust_exec, t_hedge_span, t_hedge_span + num_hedges);
        CUDA_CHECK(cudaFreeAsync(d_hedge_span, stream));
        std::cout TID(tid) << "Initial sequence (1D) weighted locality: " << std::fixed << std::setprecision(3) << tot_span << "\n";
    }
    // =============================

    return d_order_idx;
}
