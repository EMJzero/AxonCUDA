#include "construction.cuh"
#include "utils.cuh"

// count how many hedges are inbound and how many touch each node
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// SHUFFLES OVER: d
__global__
void touching_count_kernel(
    const uint32_t* __restrict__ hedges, // stores srcs first, then dsts
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t num_hedges,
    dim_t* __restrict__ touching_offsets, // initialized at 0s
    uint32_t* __restrict__ inbound_count // initialized at 0s
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the hedge to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    /*
    * Idea:
    * - every warp visits an hedge
    * - for every pin, it atomically increments the pin's touching set size
    * - for every destination, it atomically increments the pin's inbound count
    */

    const uint32_t* hedge = hedges + hedges_offsets[warp_id];
    const uint32_t hedge_size = (uint32_t)(hedges_offsets[warp_id + 1] - hedges_offsets[warp_id]);
    const dim_t hedge_srcs_count = srcs_count[warp_id];

    for (uint32_t pin_idx = lane_id; pin_idx < hedge_size; pin_idx += WARP_SIZE) {
        const uint32_t pin = hedge[pin_idx]; // already a group id
        atomicAdd(&touching_offsets[pin + 1], 1); // leave the first entry to be 0 (offset of the first set)
        if (pin_idx >= hedge_srcs_count) // the pin is a dst
            atomicAdd(&inbound_count[pin], 1);
    }
}

// write inbound and outbound sets
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
// SHUFFLES OVER: h (touching)
__global__
void touching_build_kernel(
    const uint32_t* __restrict__ hedges, // already coarsened as of here, thus contain group ids!
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_hedges,
    uint32_t* __restrict__ touching,
    uint32_t* __restrict__ inserted_inbound, // initialized at 0s
    uint32_t* __restrict__ inserted_outbound // initialized from inbound_count
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the hedge to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    /*
    * Idea:
    * - every warp visits an hedge
    * - for every source, it claims a slot by atomically incrementing the pin's count of seen srcs, then inserts the src in the pin's inbound set
    * - for every destination, same mechanism, with a separate atomic counter
    */

    const uint32_t* hedge = hedges + hedges_offsets[warp_id];
    const uint32_t hedge_size = (uint32_t)(hedges_offsets[warp_id + 1] - hedges_offsets[warp_id]);
    const dim_t hedge_srcs_count = srcs_count[warp_id];

    for (uint32_t pin_idx = lane_id; pin_idx < hedge_size; pin_idx += WARP_SIZE) {
        const uint32_t pin = hedge[pin_idx]; // already a group id
        uint32_t *pin_touching = touching + touching_offsets[pin];
        uint32_t insert_idx;
        if (pin_idx >= hedge_srcs_count) // the pin is a dst
            insert_idx = atomicAdd(&inserted_inbound[pin], 1);
        else
            insert_idx = atomicAdd(&inserted_outbound[pin], 1);
        pin_touching[insert_idx] = warp_id;
    }
}

// count how many unique neighbors each sample (node) has
// SEQUENTIAL COMPLEXITY: #samples*d*h
// PARALLEL OVER: n
// SHUFFLES OVER: h (touching)
__global__
void neighbors_sample_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    const uint32_t num_samples,
    const uint32_t samples_per_repeat, // number of samples to explore per kernel launch
    const uint32_t curr_repeat, // offset w.r.t. the start of samples
    uint32_t* __restrict__ flags_bits, // bit-flags (a unique set per sample - concatenated), set a the i-th bit to 1 if the 1-th node has been seen
    dim_t* __restrict__ neighbors_count // neighbors_count[i] -> maximum number of neighbors seen by samples handled by the i-th warp across repeats
) {
    // STYLE: one sample (node) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the sample to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= samples_per_repeat) return;

    const uint32_t sample_idx = curr_repeat * samples_per_repeat + warp_id;
    if (sample_idx >= num_samples) return;

    uint32_t* my_flags = flags_bits + warp_id * ((num_nodes + 32 - 1) / 32);
    const uint32_t my_node = static_cast<uint32_t>((static_cast<uint64_t>(sample_idx) * static_cast<uint64_t>(num_nodes)) / static_cast<uint64_t>(num_samples));

    const uint32_t* my_touching = touching + touching_offsets[my_node];
    const uint32_t* not_my_touching = touching + touching_offsets[my_node + 1];

    dim_t unique_neigh = 0u;

    for (const uint32_t* hedge_ptr = my_touching; hedge_ptr < not_my_touching; hedge_ptr++) {
        const uint32_t hedge = *hedge_ptr;
        const uint32_t* my_hedge = hedges + hedges_offsets[hedge];
        const uint32_t* not_my_hedge = hedges + hedges_offsets[hedge + 1];
        for (const uint32_t* pin_ptr = my_hedge + lane_id; pin_ptr < not_my_hedge; pin_ptr += WARP_SIZE) {
            const uint32_t pin = *pin_ptr;
            // now all thread in the warp set the "pin-th bit" of my_flags to 1 in parallel
            // => since multiple threads might want to write the same 32bit word, make them agree on a single update per word
            const uint32_t word_idx = pin >> 5;
            const uint32_t bit_mask = 1u << (pin & 31);
            const unsigned active = __activemask();
            // lanes targeting the same 32-bit word cooperate
            const unsigned peers = __match_any_sync(active, word_idx);
            // OR-reduce all bit requests for the same word across peers
            const uint32_t combined_mask = __reduce_or_sync(peers, bit_mask);
            const int leader = __ffs(peers) - 1;
            if (static_cast<int>(lane_id) == leader) {
                // exactly one writer per word per iteration
                uint32_t* const word_ptr = my_flags + word_idx;
                const uint32_t old_bits = *word_ptr;
                const uint32_t updated_bits = old_bits | combined_mask;
                *word_ptr = updated_bits;
                unique_neigh += static_cast<dim_t>(__popc(updated_bits & ~old_bits));
            }
        }
    }

    unique_neigh = warpReduceSumLN0<dim_t>(unique_neigh);

    if (lane_id == 0)
        neighbors_count[warp_id] = max(unique_neigh, neighbors_count[warp_id]);
}

// compute the distinct neighbors count of each node, aided by a global hash-set
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
__global__
void neighborhoods_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    const dim_t max_neighbors,
    const bool discharge, // if true -> finish by dumping SM into GM too, making GM contain the whole (sparse) set
    uint32_t* __restrict__ neighbors,
    dim_t* __restrict__ neighbors_offsets // here filled as counters of "how many neighbors per node" -> then do a prefix sum for the offsets
) {
    // STYLE: one node per block, one touching hedge per warp, distinct neighbors in shared memory!
    const uint32_t node_id = blockIdx.x;
    // NOTE: the whole block returns, no need to sync
    if (node_id >= num_nodes) return;

    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // local per block - coincides with the touching hedge to handle
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const uint32_t* my_touching = touching + touching_offsets[node_id];
    //const uint32_t* not_my_touching = touching + touching_offsets[node_id + 1];
    const uint32_t my_touching_count = (uint32_t)(touching_offsets[node_id + 1] - touching_offsets[node_id]);

    uint32_t* my_neighbors = neighbors + node_id * max_neighbors;

    // hash-set for deduplication (allows false-negatives, back it up with true deduplication in global memory)
    __shared__ uint32_t dedupe[SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE];
    // HP: each node has less than UINT32_MAX neighbors
    __shared__ uint32_t seen_distinct_total;
    uint32_t seen_distinct = 0;
    
    // initialize shared memory
    blk_init<uint32_t>(dedupe, SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    blk_init<uint32_t>(my_neighbors, max_neighbors, HASH_EMPTY);
    if (threadIdx.x == 0)
        seen_distinct_total = 0;
    __syncthreads();
    
    // no touching hedges, return immediately
    // NOTE: the whole block returns, no need to sync
    // MUST: return only after helping initializing shared memory
    if (my_touching_count == 0) {
        if (threadIdx.x == 0) neighbors_offsets[node_id] = 0;
        return;
    }
    if (warp_id >= my_touching_count && !discharge) return;

    // TODO: could optimize by iterating directly on pointers
    for (uint32_t touching_hedge_idx = warp_id; touching_hedge_idx < my_touching_count; touching_hedge_idx += warps_per_block) { // the block loops over touching hedges
        if (touching_hedge_idx < my_touching_count) {
            const uint32_t my_hedge_idx = my_touching[touching_hedge_idx];
            const uint32_t* my_hedge = hedges + hedges_offsets[my_hedge_idx];
            //const uint32_t* not_my_hedge = hedges + hedges_offsets[my_hedge_idx + 1];
            const uint32_t my_hedge_size = (uint32_t)(hedges_offsets[my_hedge_idx + 1] - hedges_offsets[my_hedge_idx]);
            for (uint32_t node_idx = lane_id; node_idx < my_hedge_size; node_idx += WARP_SIZE) { // the warp loops over hedge pins
                uint32_t neighbor = my_hedge[node_idx];
                if (neighbor == node_id) continue;
                uint8_t inserted = sm_hashset_try_insert(dedupe, SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE, neighbor);
                // inserted == 1 -> no need to put into GM what already is in the SM hash-set
                // inserted == 2 -> SM full, check in GM
                if (inserted == 1 || inserted == 2 && gm_hashset_insert(my_neighbors, max_neighbors, neighbor)) // triggers an assert if 'max_neighbors' is exceeded
                    seen_distinct++;
            }
        }
    }

    // reduce distinct counts per-warp
    seen_distinct = warpReduceSum<uint32_t>(seen_distinct);

    // accumulate counts per-block
    if (lane_id == 0)
        atomicAdd(&seen_distinct_total, seen_distinct);

    __syncthreads();

    if (threadIdx.x == 0)
        neighbors_offsets[node_id] = (dim_t)seen_distinct_total;

    if (discharge) {
        // dump SM over to GM all at once
        for (uint32_t i = warp_id * WARP_SIZE + lane_id; i < SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE; i += warps_per_block * WARP_SIZE) {
            uint32_t neighbor = dedupe[i];
            if (neighbor != HASH_EMPTY)
                gm_hashset_insert(my_neighbors, max_neighbors, neighbor);
        }
    }
}

// compute the distinct neighbors of each node (again) and write them at the pre-computed offsets in global memory (handled as a hash-set)
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
__global__
void neighborhoods_scatter_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    const dim_t* __restrict__ neighbors_offsets,
    uint32_t* __restrict__ neighbors
) {
    // STYLE: one node per block, one touching hedge per warp, distinct neighbors in shared memory!
    const uint32_t node_id = blockIdx.x;
    if (node_id >= num_nodes) return;

    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // local per block - coincides with the touching hedge to handle
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const uint32_t* my_touching = touching + touching_offsets[node_id];
    //const uint32_t* not_my_touching = touching + touching_offsets[node_id + 1];
    const uint32_t my_touching_count = (uint32_t)(touching_offsets[node_id + 1] - touching_offsets[node_id]);

    uint32_t* my_neighbors = neighbors + neighbors_offsets[node_id];
    const uint32_t my_neighbors_count = (uint32_t)(neighbors_offsets[node_id + 1] - neighbors_offsets[node_id]);

    // hash-set for deduplication
    __shared__ uint32_t dedupe[SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE];
    
    // initialize shared memory
    blk_init<uint32_t>(dedupe, SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    blk_init<uint32_t>(my_neighbors, my_neighbors_count, HASH_EMPTY);
    __syncthreads();

    // can return only after helping initializing shared memory
    if (warp_id >= my_touching_count) return;

    // TODO: could optimize by iterating directly on pointers
    for (uint32_t touching_hedge_idx = warp_id; touching_hedge_idx < my_touching_count; touching_hedge_idx += warps_per_block) { // the block loops over touching hedges
        if (touching_hedge_idx < my_touching_count) {
            const uint32_t my_hedge_idx = my_touching[touching_hedge_idx];
            const uint32_t* my_hedge = hedges + hedges_offsets[my_hedge_idx];
            //const uint32_t* not_my_hedge = hedges + hedges_offsets[my_hedge_idx + 1];
            const uint32_t my_hedge_size = (uint32_t)(hedges_offsets[my_hedge_idx + 1] - hedges_offsets[my_hedge_idx]);
            for (uint32_t node_idx = lane_id; node_idx < my_hedge_size; node_idx += WARP_SIZE) { // the warp loops over hedge pins
                uint32_t neighbor = my_hedge[node_idx];
                if (neighbor != node_id && sm_hashset_try_insert(dedupe, SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE, neighbor))
                    gm_hashset_insert(my_neighbors, my_neighbors_count, neighbor); // should never happen, but could trigger an assert if 'my_neighbors_count' is exceeded
            }
        }
    }
}

// count how many distinct new pins are in each hedge
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// WARPS OVER: d
__global__
void apply_coarsening_hedges_count(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ groups, // groups[node idx] -> new group/node id
    const uint32_t num_hedges,
    const uint32_t max_hedge_size,
    uint32_t* __restrict__ coarse_oversized_hedges,
    dim_t* __restrict__ coarse_hedges_offsets, // coarse_hedges_offsets[hedge idx] -> count of distinct groups among its pins
    uint32_t* __restrict__ coarse_srcs_count
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the hedge to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    /*
    * Idea:
    * - first deduplicate inside hedges to count their new, distinct, nodes (count srcs and dsts all together)
    * - scan the counts per hedge to get the new offsets
    * - repeat the deduplication and write (scatter) each new hedge to its offset
    *
    * TODO, upgrade options:
    * - instead of three kernels to count distinct nodes, scan offsets, and scatter distinct nodes, can't we do all in one?
    * - one hedge per warp or block, use the shared memory hash-map to keep a "first-come-first-served" set of counters in shared memory,
    *   and reditect additional entries to atomically incremented global counters (issue: musre ensure the src is preserved!!)
    *
    * Must ensure that:
    * - there are no self-cycles (remove the >>src<< to break them)
    * - the same node never appears twice in the same hedge
    *   => exploited by the coarsening of touching sets
    */

    const dim_t hedge_start_idx = hedges_offsets[warp_id], hedge_end_idx = hedges_offsets[warp_id + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    const uint32_t hedge_srcs_count = srcs_count[warp_id];
    const uint32_t *hedge_srcs_end = hedges + hedge_start_idx + hedge_srcs_count;
    uint32_t *oversized_coarse_hedges_start = coarse_oversized_hedges + max_hedge_size * warp_id;
    uint32_t new_hedges_count = 0, new_srcs_count = 0;
    wrp_init<uint32_t>(oversized_coarse_hedges_start, max_hedge_size, HASH_EMPTY); // HP: HASH_EMPTY is > than any value that could be inserted

    // SM deduplication hash-set
    extern __shared__ uint32_t block_new_pins[];
    uint32_t *new_pins = block_new_pins + MAX_SM_WARP_DEDUPE_BUFFER_SIZE * (threadIdx.x / WARP_SIZE);
    wrp_init<uint32_t>(new_pins, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    __syncwarp();

    // go over the hedge's destinations
    for (const uint32_t* curr = hedge_start + hedge_srcs_count + lane_id; curr < hedge_end; curr += WARP_SIZE) {
        const uint32_t pin = groups[*curr]; // read and map pin to its new id
        const uint8_t inserted = sm_hashset_try_insert(new_pins, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, pin); // check in the SM cache
        // inserted == 1 -> no need to put into GM what already is in the SM hash-set
        // inserted == 2 -> SM full, check in GM
        if (inserted == 1 || inserted == 2 && gm_hashset_insert(oversized_coarse_hedges_start, max_hedge_size, pin)) // actual dedupe
            new_hedges_count++;
    }

    // go over the hedge's sources
    for (const uint32_t* curr = hedge_start + lane_id; curr < hedge_srcs_end; curr += WARP_SIZE) {
        const uint32_t pin = groups[*curr];
        const uint8_t inserted = sm_hashset_try_insert(new_pins, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, pin);
        if (inserted == 1 || inserted == 2 && gm_hashset_insert(oversized_coarse_hedges_start, max_hedge_size, pin))
            new_srcs_count++;
    }

    new_hedges_count += new_srcs_count;

    new_hedges_count = warpReduceSumLN0<uint32_t>(new_hedges_count);
    new_srcs_count = warpReduceSumLN0<uint32_t>(new_srcs_count);

    if (lane_id == 0) {
        coarse_hedges_offsets[warp_id + 1] = (dim_t)new_hedges_count; // leave the first entry to be 0 (offset of the first hedge)
        coarse_srcs_count[warp_id] = new_srcs_count;
    }
}

// write the new distinct destinations of hedges
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// WARPS OVER: d
__global__
void apply_coarsening_hedges_scatter_dsts(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ groups, // groups[node idx] -> new group/node id
    const uint32_t num_hedges,
    const dim_t* __restrict__ coarse_hedges_offsets, // coarse_hedges_offsets[hedge idx] -> count of distinct groups among its pins
    uint32_t* __restrict__ coarse_hedges
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the hedge to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    /* Idea: second part of "apply_coarsening_hedges_count" -> scatter
    * 
    * Must ensure that:
    * - the source remains the first nodes in each coarse hedge
    * - self-cycles are broken by removing the src, not the dst
    */

    const dim_t hedge_start_idx = hedges_offsets[warp_id], hedge_end_idx = hedges_offsets[warp_id + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    const uint32_t hedge_srcs_count = srcs_count[warp_id];
    const dim_t coarse_hedge_start_idx = coarse_hedges_offsets[warp_id];
    const uint32_t coarse_hedge_size = (uint32_t)(coarse_hedges_offsets[warp_id + 1] - coarse_hedges_offsets[warp_id]);
    uint32_t *coarse_hedge_start = coarse_hedges + coarse_hedge_start_idx;
    wrp_init<uint32_t>(coarse_hedge_start, coarse_hedge_size, HASH_EMPTY); // HP: HASH_EMPTY is > than any value that could be inserted

    // SM deduplication hash-set
    extern __shared__ uint32_t block_new_pins[];
    uint32_t *new_pins = block_new_pins + MAX_SM_WARP_DEDUPE_BUFFER_SIZE * (threadIdx.x / WARP_SIZE);
    wrp_init<uint32_t>(new_pins, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

    // go over the hedge's destinations
    for (const uint32_t* curr = hedge_start + hedge_srcs_count + lane_id; curr < hedge_end; curr += WARP_SIZE) {
        const uint32_t pin = groups[*curr]; // read and map pin to its new id
        if (sm_hashset_try_insert(new_pins, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, pin)) // check in the SM cache
            gm_hashset_insert(coarse_hedge_start, coarse_hedge_size, pin); // actual dedupe and final insertion
    }
}

// write the new distinct sources of hedges
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
// WARPS OVER: d
__global__
void apply_coarsening_hedges_scatter_srcs(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ groups, // groups[node idx] -> new group/node id
    const uint32_t num_hedges,
    const dim_t* __restrict__ coarse_hedges_offsets, // coarse_hedges_offsets[hedge idx] -> count of distinct groups among its pins
    const uint32_t* __restrict__ coarse_srcs_count,
    uint32_t* __restrict__ coarse_hedges
) {
    // STYLE: one hedge per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the hedge to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    /* 
    * Second half of 'apply_coarsening_hedges_scatter_srcs'
    * => assumes the destinations segments have been now sorted, and can be used for binary search
    */

    const dim_t hedge_start_idx = hedges_offsets[warp_id];
    const uint32_t *hedge_start = hedges + hedge_start_idx;
    const uint32_t hedge_srcs_count = srcs_count[warp_id];
    const uint32_t *hedge_srcs_end = hedges + hedge_start_idx + hedge_srcs_count;
    const dim_t coarse_hedge_start_idx = coarse_hedges_offsets[warp_id];
    const uint32_t coarse_hedge_size = (uint32_t)(coarse_hedges_offsets[warp_id + 1] - coarse_hedges_offsets[warp_id]);
    const uint32_t coarse_hedge_srcs_count = coarse_srcs_count[warp_id];
    const uint32_t coarse_hedge_dsts_count = coarse_hedge_size - coarse_hedge_srcs_count;
    uint32_t *coarse_hedge_srcs_start = coarse_hedges + coarse_hedge_start_idx;
    uint32_t *coarse_hedge_dsts_start = coarse_hedges + coarse_hedge_start_idx + coarse_hedge_srcs_count;

    // SM deduplication hash-set
    extern __shared__ uint32_t block_new_pins[];
    uint32_t *new_pins = block_new_pins + MAX_SM_WARP_DEDUPE_BUFFER_SIZE * (threadIdx.x / WARP_SIZE);
    wrp_init<uint32_t>(new_pins, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, HASH_EMPTY);

    // go over the hedge's destinations
    for (const uint32_t* curr = hedge_start + lane_id; curr < hedge_srcs_end; curr += WARP_SIZE) {
        const uint32_t pin = groups[*curr]; // read and map pin to its new id
        // insert == 0, 1 -> no need to put into GM what already is in the SM hash-set
        // insert == 2 -> SM full, check in GM
        if (sm_hashset_try_insert(new_pins, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, pin) == 2) { // check in the SM cache
            if(binary_search<uint32_t, false>(coarse_hedge_dsts_start, coarse_hedge_dsts_count, pin) == UINT32_MAX) // check among destinations (be wary: descending order!)
                gm_hashset_insert(coarse_hedge_srcs_start, coarse_hedge_srcs_count, pin); // actual dedupe and final insertion
        }
    }
    __syncwarp();

    // dump SM over to GM all at once => the deferred flush prevents catastrophic probe lengths in the main loop
    for (uint32_t i = lane_id; i < MAX_SM_WARP_DEDUPE_BUFFER_SIZE; i += WARP_SIZE) {
        uint32_t pin = new_pins[i];
        if (pin != HASH_EMPTY && binary_search<uint32_t, false>(coarse_hedge_dsts_start, coarse_hedge_dsts_count, pin) == UINT32_MAX)
            gm_hashset_insert(coarse_hedge_srcs_start, coarse_hedge_srcs_count, pin);
    }
}

// count how many distinct neighbors there are in each group
// SEQUENTIAL COMPLEXITY: n*d*h (in reality there are <<d*h neighbors per node)
// PARALLEL OVER: n
// WARPS OVER: d*h (neighbors)
__global__
void apply_coarsening_neighbors_count(
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ groups, // groups[node id] -> node's group id
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const dim_t max_neighbors,
    const bool discharge, // if true -> finish by dumping SM into GM too, making GM contain the whole (sparse) set
    uint32_t* __restrict__ oversized_coarse_neighbors, // deduplication buffer, you can use it between oversized_coarse_neighbors[max_neighbors * ungroups_offsets[my idx]] and oversized_coarse_neighbors[max_neighbors * ungroups_offsets[my idx + 1]]
    dim_t* __restrict__ coarse_neighbors_offsets // group id -> count of distinct neighbors
) {
    // STYLE: one group (new node) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the group to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_groups) return;

    /*
    * Idea:
    * - first translate to their new id and deduplicate neighbors inside each group to count their new, distinct, nodes
    * - scan the counts per group to get the new offsets
    * - repeat the deduplication and write (scatter) each new neighbor (neighboring group id) to its offset
    *
    * Must ensure that:
    * - a node itself NEVER appears among its own neighbors
    *
    * NOTE: by doing this, instead of rebuilding neighborhoods from scratch, we have fewer neighbors to deduplicate,
    *       since most were already handled at the level above! While this runs we keep allocated both the old and new sets...
    *
    * TODO: compare this with rebuilding neighborhoods from scratch, to see if it is faster! (it is worth the memory investment)
    *
    * TODO: is it better here to have one group per warp, or one per block, like the initial construction of neighbors?
    */

    const dim_t ungroups_start_idx = ungroups_offsets[warp_id], ungroups_end_idx = ungroups_offsets[warp_id + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    uint32_t *oversized_coarse_neighbors_start = oversized_coarse_neighbors + max_neighbors * ungroups_start_idx;
    const dim_t oversized_coarse_neighbors_size = max_neighbors * (ungroups_end_idx - ungroups_start_idx);
    dim_t new_neighbors_count = 0u;
    wrp_init<uint32_t>(oversized_coarse_neighbors_start, oversized_coarse_neighbors_size, HASH_EMPTY); // HP: HASH_EMPTY is > than any value that could be inserted

    // SM deduplication hash-set
    extern __shared__ uint32_t block_new_neighbors[];
    uint32_t *new_neighbors = block_new_neighbors + MAX_SM_WARP_DEDUPE_BUFFER_SIZE * (threadIdx.x / WARP_SIZE);
    wrp_init<uint32_t>(new_neighbors, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    __syncwarp();

    // for every original node in the group, go over its touching set
    // TODO: maybe let threads that finish a node move over to the next in the group as soon as possible
    for (const uint32_t* ungroup = ungroups_start; ungroup < ungroups_end; ungroup++) {
        const uint32_t node = *ungroup;
        const uint32_t* my_neighbors = neighbors + neighbors_offsets[node];
        const uint32_t my_neighbors_count = neighbors_offsets[node + 1] - neighbors_offsets[node];
        for (uint32_t i = lane_id; i < my_neighbors_count; i += WARP_SIZE) {
            const uint32_t new_neighbor = groups[my_neighbors[i]]; // translate to group id
            if (warp_id == new_neighbor) continue;
            const uint8_t inserted = sm_hashset_try_insert(new_neighbors, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, new_neighbor); // check in the SM cache
            // inserted == 1 -> no need to put into GM what already is in the SM hash-set
            // inserted == 2 -> SM full, check in GM
            if (inserted == 1 || inserted == 2 && gm_hashset_insert(oversized_coarse_neighbors_start, oversized_coarse_neighbors_size, new_neighbor)) // triggers an assert if 'max_neighbors' is exceeded
                new_neighbors_count++;
        }
    }

    new_neighbors_count = warpReduceSumLN0<dim_t>(new_neighbors_count);

    if (discharge) {
        // dump SM over to GM all at once; rely on the above warp-reduce as a sync
        for (uint32_t i = lane_id; i < MAX_SM_WARP_DEDUPE_BUFFER_SIZE; i += WARP_SIZE) {
            uint32_t neighbor = new_neighbors[i];
            if (neighbor != HASH_EMPTY)
                gm_hashset_insert(oversized_coarse_neighbors_start, oversized_coarse_neighbors_size, neighbor);
        }
    }

    if (lane_id == 0)
        coarse_neighbors_offsets[warp_id + 1] = new_neighbors_count; // leave the first entry to be 0 (offset of the first set)
}

// write distinct neighbors
// SEQUENTIAL COMPLEXITY: n*d*h (in reality there are <<d*h neighbors per node)
// PARALLEL OVER: n
// WARPS OVER: d*h (neighbors)
__global__
void apply_coarsening_neighbors_scatter(
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ groups, // groups[node id] -> node's group id
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const dim_t* __restrict__ coarse_neighbors_offsets, // group id -> count of distinct neighbors
    uint32_t* __restrict__ coarse_neighbors
) {
    // STYLE: one group (new node) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the group to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_groups) return;

    // Idea: second part of "apply_coarsening_neighbors_count" -> scatter

    const dim_t ungroups_start_idx = ungroups_offsets[warp_id], ungroups_end_idx = ungroups_offsets[warp_id + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    const dim_t coarse_neighbors_start_idx = coarse_neighbors_offsets[warp_id], coarse_neighbors_end_idx = coarse_neighbors_offsets[warp_id + 1];
    const uint32_t coarse_neighbors_size = (uint32_t)(coarse_neighbors_end_idx - coarse_neighbors_start_idx);
    uint32_t *coarse_neighbors_start = coarse_neighbors + coarse_neighbors_start_idx;
    wrp_init<uint32_t>(coarse_neighbors_start, coarse_neighbors_size, HASH_EMPTY); // HP: HASH_EMPTY is > than any value that could be inserted

    // SM deduplication hash-set
    extern __shared__ uint32_t block_new_neighbors[];
    uint32_t *new_neighbors = block_new_neighbors + MAX_SM_WARP_DEDUPE_BUFFER_SIZE * (threadIdx.x / WARP_SIZE);
    wrp_init<uint32_t>(new_neighbors, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    __syncwarp();

    // for every original node in the group, go over its touching set
    // TODO: maybe let threads that finish a node move over to the next in the group as soon as possible
    for (const uint32_t* ungroup = ungroups_start; ungroup < ungroups_end; ungroup++) {
        const uint32_t node = *ungroup;
        const uint32_t* my_neighbors = neighbors + neighbors_offsets[node];
        const uint32_t my_neighbors_count = neighbors_offsets[node + 1] - neighbors_offsets[node];
        for (uint32_t i = lane_id; i < my_neighbors_count; i += WARP_SIZE) {
            const uint32_t new_neighbor = groups[my_neighbors[i]]; // translate to group id
            if (warp_id == new_neighbor) continue;
            if (sm_hashset_try_insert(new_neighbors, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, new_neighbor)) // check in the SM cache
                gm_hashset_insert(coarse_neighbors_start, coarse_neighbors_size, new_neighbor); // actual dedupe and final insertion
        }
    }
}

// count how many hedges touch each node
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void apply_coarsening_touching_count(
    const uint32_t* __restrict__ hedges, // already coarsened as of here, thus contain group ids!
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    dim_t* __restrict__ coarse_touching_offsets // here filled as counters of "how many hedges per node" -> then do a prefix sum for the offsets
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /*
    * Idea:
    * - first coarsen hedge, then going over coarse hedge and counting the occurrencies of each group (new node) should be
    *   faster than just deduplicating touching hedges between nodes in the same group
    * - scan the touching counts per group (new node) to get the new offsets
    * - the scatter operates in a different way, assigning one thread per group, the thread goes over the nodes
    *   in the group via the "ungroups" and "ungroups_offsets" structures, deduplicates hedges and writes them in the touching set
    *
    * NOTE: this exploits the fact that the same pin never appears twice in the same hedge!
    *
    * TODO, upgrade options:
    * - for now we just do atomics towards global memory, because nodes are too many for shared memory, eventually:
    *   => use the shared memory hash-map to keep a "first-come-first-served" set of counters in shared memory,
    *     and reditect additional entries to global counters, then atomically increment global memory
    *
    * TODO: could partially skip this counting step, because we already have the new distinct inbound count evaluated during the candidates kernel!
    * 
    * TODO: could already count inbound and outbound separately, then use singler scatter kernel putting them already in the right place!
    */

    const dim_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;

    // TODO: initialize shared memory
    // in utils.cuh : #define SM_MAX_HASHMAP_SIZE 4096u ?
    //__shared__ hashmap_entry hashmap[SM_MAX_HASHMAP_SIZE];
    //blk_init<uint32_t>(hashmap, SM_MAX_HASHMAP_SIZE*2, HASH_EMPTY);

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t pin = *curr; // already a group id
        atomicAdd(&coarse_touching_offsets[pin + 1], 1); // leave the first entry to be 0 (offset of the first set)
    }
}

// write distinct inbound touching hedges
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
// SHUFFLES OVER: h (touching)
__global__
void apply_coarsening_touching_scatter_inbound(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const dim_t* __restrict__ coarse_touching_offsets, // group id -> count of distinct touching hedges
    uint32_t* __restrict__ coarse_touching,
    uint32_t* __restrict__ coarse_inbound_count
) {
    // STYLE: one group (new node) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the group to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_groups) return;

    /*
    * Idea: second part of "apply_coarsening_touching_scatter" -> scatter inbound
    * Alternative version: one thread per hedge and dedupe inside each touching set (by linearly going over it, with a CAS at the end)
    * 
    * Must ensure that:
    * - inbound hedges are at the start of the set
    * - inbound hedges are sorted by id
    *
    * TODO upgrade option:
    * - one warp per node is fine with many nodes, but when nodes are few switch to one node per block!
    */

    /*
    * Important: here hedges that are both inbound and outbound are lost on one side, the outbound one specifically.
    *            In other words, and hedge both entering and leaving a group will only be remembered as an inbound one,
    *            this is because the deduplication is done for all touching at once, while the inbound count is incremented
    *            only for the first occurrence, that is, the inbound one!
    */

    const dim_t ungroups_start_idx = ungroups_offsets[warp_id], ungroups_end_idx = ungroups_offsets[warp_id + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    const dim_t coarse_touching_start_idx = coarse_touching_offsets[warp_id];
    const uint32_t coarse_touching_size = (uint32_t)(coarse_touching_offsets[warp_id + 1] - coarse_touching_offsets[warp_id]);
    uint32_t *coarse_touching_start = coarse_touching + coarse_touching_start_idx;
    uint32_t new_inbound_count = 0u;
    wrp_init<uint32_t>(coarse_touching_start, coarse_touching_size, HASH_EMPTY); // HP: HASH_EMPTY is > than any value that could be inserted

    // SM deduplication hash-set
    extern __shared__ uint32_t block_new_touching[];
    uint32_t *new_touching = block_new_touching + MAX_SM_WARP_DEDUPE_BUFFER_SIZE * (threadIdx.x / WARP_SIZE);
    wrp_init<uint32_t>(new_touching, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    __syncwarp();

    // for every original node in the group, go over its inbound set
    for (const uint32_t* ungroup = ungroups_start; ungroup < ungroups_end; ungroup++) {
        const uint32_t node = *ungroup;
        const uint32_t* my_inbound = touching + touching_offsets[node];
        const uint32_t my_inbound_count = inbound_count[node];
        for (uint32_t i = lane_id; i < my_inbound_count; i += WARP_SIZE) {
            const uint32_t hedge_idx = my_inbound[i];
            if (sm_hashset_try_insert(new_touching, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, hedge_idx)) { // check in the SM cache
                if (gm_hashset_insert(coarse_touching_start, coarse_touching_size, hedge_idx)) // dedupe among inbound hedges
                    new_inbound_count++;
            }
        }
    }

    new_inbound_count = warpReduceSumLN0<uint32_t>(new_inbound_count);

    if (lane_id == 0)
        coarse_inbound_count[warp_id] = new_inbound_count;
}

// write distinct outbound touching hedges
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n
// SHUFFLES OVER: h (touching)
__global__
void apply_coarsening_touching_scatter_outbound(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const dim_t* __restrict__ coarse_touching_offsets, // group id -> count of distinct touching hedges
    const uint32_t* __restrict__ coarse_inbound_count,
    uint32_t* __restrict__ coarse_touching
) {
    // STYLE: one group (new node) per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the group to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_groups) return;

    /* 
    * Idea: third part of "apply_coarsening_touching_scatter" -> scatter outbound
    *  => assumes each inbound set has been now sorted, and can be used for binary search
    */

    const dim_t ungroups_start_idx = ungroups_offsets[warp_id], ungroups_end_idx = ungroups_offsets[warp_id + 1];
    const uint32_t *ungroups_start = ungroups + ungroups_start_idx, *ungroups_end = ungroups + ungroups_end_idx;
    const dim_t coarse_touching_start_idx = coarse_touching_offsets[warp_id];
    const uint32_t coarse_touching_size = (uint32_t)(coarse_touching_offsets[warp_id + 1] - coarse_touching_offsets[warp_id]);
    const uint32_t new_inbound_count = coarse_inbound_count[warp_id];
    const uint32_t coarse_outbound_size = coarse_touching_size - new_inbound_count;
    uint32_t *coarse_touching_start = coarse_touching + coarse_touching_start_idx;
    uint32_t *coarse_outbound_start = coarse_touching_start + new_inbound_count;

    // SM deduplication hash-set
    extern __shared__ uint32_t block_new_touching[];
    uint32_t *new_touching = block_new_touching + MAX_SM_WARP_DEDUPE_BUFFER_SIZE * (threadIdx.x / WARP_SIZE);
    wrp_init<uint32_t>(new_touching, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, HASH_EMPTY);
    __syncwarp();

    // for every original node in the group, go over its outbound set
    for (const uint32_t* ungroup = ungroups_start; ungroup < ungroups_end; ungroup++) {
        const uint32_t node = *ungroup;
        const uint32_t* my_outbound = touching + touching_offsets[node] + inbound_count[node];
        const uint32_t my_outbound_count = (uint32_t)(touching_offsets[node + 1] - touching_offsets[node]) - inbound_count[node];
        for (uint32_t i = lane_id; i < my_outbound_count; i += WARP_SIZE) {
            const uint32_t hedge_idx = my_outbound[i];
            // insert == 0, 1 -> no need to put into GM what already is in the SM hash-set
            // insert == 2 -> SM full, check in GM
            if (sm_hashset_try_insert(new_touching, MAX_SM_WARP_DEDUPE_BUFFER_SIZE, hedge_idx) == 2) { // check in the SM cache
                if(binary_search<uint32_t, true>(coarse_touching_start, new_inbound_count, hedge_idx) == UINT32_MAX) // check among inbound hedges
                    gm_hashset_insert(coarse_outbound_start, coarse_outbound_size, hedge_idx); // dedupe among outbound hedges
            }
        }
    }
    __syncwarp();

    // dump SM over to GM all at once => the deferred flush prevents catastrophic probe lengths in the main loop
    for (uint32_t i = lane_id; i < MAX_SM_WARP_DEDUPE_BUFFER_SIZE; i += WARP_SIZE) {
        uint32_t hedge_idx = new_touching[i];
        if (hedge_idx != HASH_EMPTY && binary_search<uint32_t, true>(coarse_touching_start, new_inbound_count, hedge_idx) == UINT32_MAX)
            gm_hashset_insert(coarse_outbound_start, coarse_outbound_size, hedge_idx);
    }
}

// write to each node the partition of its group
__global__
void apply_uncoarsening_partitions(
    const uint32_t* __restrict__ groups, // groups[node id] -> node's group
    const uint32_t* __restrict__ coarse_partitions, // coarse_partitions[group id] -> group's partition
    const uint32_t num_nodes,
    uint32_t* __restrict__ partitions // partitions[node id] -> group's partition
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    // Idea: gather, each node reads its group, and from it its partition

    const uint32_t my_group = groups[tid];
    const uint32_t my_partition = coarse_partitions[my_group];
    partitions[tid] = my_partition;
}

// for each hyperedge, count how many of its pins are in each partition
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
__global__
void pins_per_partition_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions, // partitions[idx] is the partition node idx is part of
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ pins_per_partitions, // pins_per_partitions[hedge_idx * num_partitions + partition_idx] is the number of pins of that partition owned by this hedge
    uint32_t* __restrict__ partitions_inbound_sizes // partitions_inbound_sizes[part] -> number of inbound_pins_per_partitions for 'part' that are not zero => will be incorrect as of here (also including outbounds)
) {
    // STYLE: one hedge per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hedges) return;

    /*
    * TODO upgrade:
    * - one hedge per warp -> go over the hedge in // in the warp
    * - shared memory histogram per hyperedge
    */

    /*
    * TODO alternative upgrade:
    * - one node per warp -> go over touching in // in the warp
    * - shared memory histogram per partition
    * => take the code from 'inbound_pins_per_partition_kernel' as of commit 'f803463'
    */

    const dim_t hedge_start_idx = hedges_offsets[tid], hedge_end_idx = hedges_offsets[tid + 1];
    const uint32_t *hedge_start = hedges + hedge_start_idx, *hedge_end = hedges + hedge_end_idx;
    uint32_t *my_pins_per_partitions = pins_per_partitions + tid * num_partitions;

    for (const uint32_t* curr = hedge_start; curr < hedge_end; curr++) {
        const uint32_t part = partitions[*curr];
        const uint32_t prev = atomicAdd(&my_pins_per_partitions[part], 1);
        if (prev == 0) atomicAdd(&partitions_inbound_sizes[part], 1);
    }
}

// generic kernel to pack oversized/sparse CSR-like subarrays, assumes blank entries to be UINT32_MAX
// SEQUENTIAL COMPLEXITY: n*sub_size
// PARALLEL OVER: n
// SHUFFLES OVER: sub_size
__global__
void pack_segments(
    const uint32_t* __restrict__ oversized, // an array of 'num_subs' subarrays, each of size 'sub_size'
    const dim_t* __restrict__ offsets, // 'offsets[i]' specifies the starting idx in 'out' where to write the packed i-th subarray of 'oversized'
    const uint32_t num_subs,
    const dim_t sub_size,
    uint32_t* __restrict__ out // packed array of subarrays, each of length 'offsets[i+1] - offsets[i]'
) {
    // STYLE: one subarray per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_subs) return;

    dim_t out_start_idx = offsets[warp_id];
    //const uint32_t out_len = (uint32_t)(offsets[warp_id + 1] - offsets[warp_id]);
    const dim_t in_start_idx = warp_id * sub_size;

    // Process WARP_SIZE items in rounds
    for (int i = lane_id; i < sub_size; i += WARP_SIZE) {
        uint32_t v = oversized[in_start_idx + i];
        int p = (v != UINT32_MAX);
        unsigned active = __activemask();
        unsigned mask = __ballot_sync(active, p); // bit set to 1 for every lane not seeing a null value

        int lane_rank = __popc(mask & ((1u << lane_id) - 1u)); // count how many bits were set to 1 before by lanes before me
        int p_count = __popc(mask); // total number of non-null value seen by the warp

        if (p) out[out_start_idx + lane_rank] = v;

        out_start_idx += p_count;
    }
}

// variant of 'pack_segments' that allows for segments of variable size:
// each segment is assumed to be of length multiple of 'sub_base_size', starting at offset 'oversized_offsets[i]*sub_base_size'
__global__
void pack_segments_varsize(
    const uint32_t* __restrict__ oversized, // an array of 'num_subs' subarrays, each of size 'sub_size'
    const dim_t* __restrict__ oversized_offsets, // 'oversized_offsets[i]*sub_base_size' specifies the starting idx in 'oversized' of the i-th sparse subarray
    const dim_t* __restrict__ offsets, // 'offsets[i]' specifies the starting idx in 'out' where to write the packed i-th subarray of 'oversized'
    const uint32_t num_subs,
    const dim_t sub_base_size,
    uint32_t* __restrict__ out // packed array of subarrays, each of length 'offsets[i+1] - offsets[i]'
) {
    // STYLE: one subarray per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_subs) return;

    dim_t out_start_idx = offsets[warp_id];
    //const uint32_t out_len = (uint32_t)(offsets[warp_id + 1] - offsets[warp_id]);
    const dim_t in_start_idx = sub_base_size * oversized_offsets[warp_id];
    // TODO: some warps get large subarrays and work more...not balanced!
    const dim_t sub_size = sub_base_size * (oversized_offsets[warp_id + 1] - oversized_offsets[warp_id]);

    // process WARP_SIZE items in rounds
    for (int i = lane_id; i < sub_size; i += WARP_SIZE) {
        uint32_t v = oversized[in_start_idx + i];
        int p = (v != UINT32_MAX);
        unsigned active = __activemask();
        unsigned mask = __ballot_sync(active, p); // bit set to 1 for every lane not seeing a null value

        int lane_rank = __popc(mask & ((1u << lane_id) - 1u)); // count how many bits were set to 1 before by lanes before me
        int p_count = __popc(mask); // total number of non-null value seen by the warp

        if (p) out[out_start_idx + lane_rank] = v;

        out_start_idx += p_count;
    }
}
