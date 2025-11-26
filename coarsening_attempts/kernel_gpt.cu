// kernel_candidates.cu
#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32u

// Tunables: increase HASH_SIZE for fewer collisions (costs shared memory).
// Make HASH_SIZE a power of two for simple masking.
#define HASH_SIZE 1024u

// How many probes at most during insertion/find in the hash table
#define PROBE_LIMIT 8u

// Helper: warp-wide "broadcast" using shuffle
static __inline__ __device__ uint32_t warp_bcast(uint32_t val, int srcLane) {
    return __shfl_sync(0xffffffffu, val, srcLane);
}

static __inline__ __device__ uint32_t warp_lane_id() {
    return threadIdx.x & (WARP_SIZE - 1u);
}

// Simple mix hash for 32-bit uint
static __inline__ __device__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

/*
 * Kernel: for each warp assigned a hyperedge (warp_id -> hyperedge index)
 * - loads that hyperedge's nodes into local storage (he_nodes)
 * - for each node u in he_nodes (warp-rotated):
 *     - initializes per-warp shared-hash (keys = UINT32_MAX empty, vals = 0)
 *     - scans ALL other hyperedges, in batches of WARP_SIZE:
 *         - each lane reads a value 'node' (or UINT32_MAX)
 *         - warp-shuffle to let all lanes see each node in the batch
 *         - for each seen node 'n', do: insert/add `weight` into hash[n]
 *     - after scan, lane 0 reduces hash to find best candidate (max val)
 *     - write best_candidate[node_u] = best_id (and best_score if desired)
 *
 * Shared memory layout per warp:
 *   keys:  HASH_SIZE uint32_t
 *   vals:  HASH_SIZE float
 *
 * We place the whole per-warp hash table in shared memory. Each block contains
 * multiple warps, so we slice shared memory per-warp.
 *
 * NOTE: HASH_SIZE must be chosen so that (WARPS_PER_BLOCK * HASH_SIZE *
 * (sizeof(uint32_t)+sizeof(float))) fits the GPU shared memory limit.
 */
extern
__global__
void hyperedge_candidate_kernel(
    const uint32_t* hedge_offsets, // length num_hedges+1
    const uint32_t* hedges_flat,   // concatenated nodes
    const float* hedge_weight,     // optional per-hyperedge weight (if used)
    uint32_t num_hedges,
    uint32_t num_nodes,            // total distinct node count (used for result array size)
    uint32_t* out_best,            // output: best candidate node id per source node (global)
    float* out_best_score          // output: best score per source node (optional)
) {
    // compute warp id across grid
    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = global_tid / WARP_SIZE;
    if (warp_id >= num_hedges) return;

    uint32_t lane = threadIdx.x & (WARP_SIZE - 1u);
    uint32_t warps_per_block = blockDim.x / WARP_SIZE;
    uint32_t warp_in_block = (threadIdx.x / WARP_SIZE); // 0..warps_per_block-1

    // Shared memory layout: each warp gets a slice
    // keys_base points to uint32_t block of keys, vals_base to float block of vals
    extern __shared__ uint8_t smem_bytes[]; // raw shared memory
    uint8_t* smem_base = smem_bytes;

    // total size per warp in bytes
    const size_t bytes_per_key = sizeof(uint32_t);
    const size_t bytes_per_val = sizeof(float);
    const size_t bytes_per_entry = bytes_per_key + bytes_per_val;
    const size_t warp_table_bytes = (size_t)HASH_SIZE * (bytes_per_key + bytes_per_val);

    // pointer to this warp's table slice
    uint8_t* my_warp_slice = smem_base + warp_in_block * warp_table_bytes;
    uint32_t* keys = reinterpret_cast<uint32_t*>(my_warp_slice);
    float* vals = reinterpret_cast<float*>(my_warp_slice + HASH_SIZE * sizeof(uint32_t));

    // === 1) Load assigned hyperedge nodes into local array (in registers/local mem)
    uint32_t he_idx = warp_id;
    uint32_t he_start = hedge_offsets[he_idx];
    uint32_t he_end   = hedge_offsets[he_idx + 1];
    uint32_t he_len   = he_end - he_start;

    // We store assigned hyperedge nodes in local memory (stack array).
    // To avoid large stack usage we process them one at a time in batches of WARP_SIZE.
    // But we also need to iterate nodes of our hyperedge as "sources".
    // We'll first load them into a compact local buffer if they are small.
    // We'll set a safety cap for local storage:
    const uint32_t LOCAL_CAP = 256; // tweakable; if a hyperedge longer, we'll stream nodes
    // allocate local storage in registers/local memory
    // NOTE: large LOCAL_CAP can increase register pressure; choose carefully.
    uint32_t he_local[LOCAL_CAP];
    uint32_t he_local_len = 0;

    // load into local buffer (all lanes cooperatively load)
    for (uint32_t i = lane; i < he_len && i < LOCAL_CAP; i += WARP_SIZE) {
        he_local[i] = hedges_flat[he_start + i];
    }
    // compute actual local_len (warp-synchronously)
    // lane 0 finds max index loaded
    if (lane == 0) {
        he_local_len = (he_len <= LOCAL_CAP) ? he_len : LOCAL_CAP;
    }
    // broadcast he_local_len to all lanes
    he_local_len = __shfl_sync(0xffffffffu, he_local_len, 0);

    // If hyperedge longer than LOCAL_CAP, we'll process remaining nodes streaming later.
    // For simplicity here we assume typical hyperedges fit LOCAL_CAP. If not, code can be extended.

    // ==== For each source node in the assigned hyperedge (iterate he_local entries) ====
    for (uint32_t src_i = 0; src_i < he_local_len; ++src_i) {
        uint32_t src_node;
        // lane 0 has the node in he_local[src_i], broadcast it
        if (lane == 0) src_node = he_local[src_i];
        src_node = __shfl_sync(0xffffffffu, src_node, 0); // now every lane knows src_node

        // Initialize hash table slice for this warp (keys = UINT32_MAX, vals = 0)
        // We'll have all lanes zero the table cooperatively
        // Clear keys
        for (uint32_t k = lane; k < HASH_SIZE; k += WARP_SIZE) {
            keys[k] = UINT32_MAX;
            vals[k] = 0.0f;
        }
        __syncwarp();

        // Scan *all* hyperedges and accumulate into shared-hash table
        for (uint32_t other = 0; other < num_hedges; ++other) {
            if (other == he_idx) continue; // skip same hyperedge (or include, as you like)

            uint32_t o_start = hedge_offsets[other];
            uint32_t o_end = hedge_offsets[other + 1];
            uint32_t o_len = o_end - o_start;

            // process in warp-sized batches
            for (uint32_t batch = 0; batch < o_len; batch += WARP_SIZE) {
                uint32_t idx = batch + lane;
                uint32_t node = (idx < o_len) ? hedges_flat[o_start + idx] : UINT32_MAX;
                // broadcast each element of the batch so all lanes see it in turn
                // But instead of broadcasting with a loop over i 0..WARP_SIZE-1 on all lanes
                // we instead use shfl to get each element: lane i's local 'node' will be shfl'd
                // to all lanes via a loop.
                #pragma unroll
                for (int i = 0; i < (int)WARP_SIZE; ++i) {
                    uint32_t n = __shfl_sync(0xffffffffu, node, i);
                    if (n == UINT32_MAX) continue;

                    // Optionally weight by the hyperedge's weight:
                    float add_val = 1.0f;
                    if (hedge_weight) {
                        add_val = hedge_weight[other];
                    }

                    // Insert/add to shared hash table (open addressing, linear probing)
                    uint32_t h = mix32(n) & (HASH_SIZE - 1u);
                    bool done = false;
                    // We attempt up to PROBE_LIMIT probes
                    for (uint32_t p = 0; p < PROBE_LIMIT; ++p) {
                        uint32_t idx_k = (h + p) & (HASH_SIZE - 1u);
                        uint32_t prev = atomicCAS(&keys[idx_k], UINT32_MAX, n);
                        if (prev == UINT32_MAX || prev == n) {
                            // we are the thread that either inserted the key or found it
                            atomicAdd(&vals[idx_k], add_val);
                            done = true;
                            break;
                        }
                        // otherwise continue probing
                    }
                    if (!done) {
                        // Hash table is saturated or high collision; as fallback, do nothing
                        // Could implement global-memory accumulator or larger table; omitted for performance
                    }
                } // end loop over warp elements
            } // end batches
        } // end other hyperedges scan

        __syncwarp(); // make sure all atomic adds are done

        // Now find best candidate in the hash table for this src_node
        // We'll let lane 0 scan the hash table and find max value
        float best_val = 0.0f;
        uint32_t best_key = UINT32_MAX;
        // Each lane scans a disjoint subset and then we reduce across warp (lane 0 collects)
        float local_best_val = 0.0f;
        uint32_t local_best_key = UINT32_MAX;
        for (uint32_t k = lane; k < HASH_SIZE; k += WARP_SIZE) {
            uint32_t key = keys[k];
            float v = vals[k];
            if (key != UINT32_MAX && v > local_best_val) {
                local_best_val = v;
                local_best_key = key;
            }
        }

        // Now reduce local_best across warp to lane 0
        // We perform pairwise max reduction using shuffles (we need to carry both key and val).
        // Strategy: reduce values only, and then lane 0 finds the corresponding key by scanning
        float warp_best_val = local_best_val;
        // warp-reduce maximum (float)
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffffu, warp_best_val, offset);
            if (other_val > warp_best_val) warp_best_val = other_val;
        }

        if (lane == 0) {
            // lane 0 now knows the warp_best_val; find its key by scanning table
            best_val = warp_best_val;
            if (best_val > 0.0f) {
                uint32_t found_key = UINT32_MAX;
                // linear scan table for the first matching entry with value == best_val (within tolerance).
                // In pathological ties, this picks the first found.
                for (uint32_t k = 0; k < HASH_SIZE; ++k) {
                    if (keys[k] != UINT32_MAX) {
                        float v = vals[k];
                        if (v == best_val) { found_key = keys[k]; break; }
                    }
                }
                best_key = found_key;
            } else {
                best_key = UINT32_MAX;
            }

            // store results to global memory for source node (src_node)
            if (src_node < num_nodes) {
                out_best[src_node] = best_key;
                out_best_score[src_node] = best_val;
            }
        }

        __syncwarp(); // ensure lane 0 finished write before next src node
    } // end loop over src nodes in the assigned hyperedge
}


////////////////////
// OTHER GPT CRAP //
////////////////////

// shared_hash_helpers.cu
#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32u
#define HASH_SIZE 1024u // per-warp constant; you'll set per-table capacity at runtime too
#define PROBE_LIMIT 8u

// Sentinel for "empty" key
#ifndef EMPTY_KEY
#define EMPTY_KEY 0xFFFFFFFFu // UINT32_MAX
#endif

// Multiply-shift hash for 32-bit key -> good enough; requires table_size power-of-two.
__device__ __forceinline__ uint32_t hash32(uint32_t key, uint32_t mask) {
    // Knuth multiplicative hash
    return (uint32_t)((key * 2654435761u) & mask);
}

// -----------------------------------------------------------------------------
// NOTE about concurrency:
// - Multiple threads may increment the same table concurrently.
// - We use atomicCAS to claim an empty slot (EMPTY_KEY) and atomicAdd on the
//   float counter to increment the weight once the slot contains the key.
// - We limit probing to PROBE_LIMIT attempts to bound runtime. If probe limit
//   is exceeded the increment is dropped (could be accumulated into overflow
//   bucket in a future version).
// -----------------------------------------------------------------------------

// Attempt to insert/increment `key` in shared table. The tables are stored
// as contiguous blocks in shared memory: keys_base + table_idx * table_capacity,
// vals_base + table_idx * table_capacity.
// Arguments:
//   table_idx      : which table (0..num_tables-1)
//   key            : neighbor node id to increment (if key == EMPTY_KEY -> no-op)
//   weight         : weight to add
//   keys_base      : pointer to shared keys region (uint32_t*)
//   vals_base      : pointer to shared values region (float*)
//   table_capacity : power-of-two capacity (entries per table)
//   mask           : table_capacity - 1
__device__ inline void shared_table_increment(
    uint32_t table_idx,
    uint32_t key,
    float weight,
    uint32_t* keys_base,
    float* vals_base,
    uint32_t table_capacity,
    uint32_t mask)
{
    if (key == EMPTY_KEY) return;

    // base pointer into the shared memory region for this table
    uint32_t* table_keys = keys_base + (uint64_t)table_idx * table_capacity;
    float* table_vals = vals_base + (uint64_t)table_idx * table_capacity;

    // initial probe position
    uint32_t pos = hash32(key, mask);

    // linear probing bounded by PROBE_LIMIT
    for (uint32_t p = 0; p < PROBE_LIMIT; ++p) {
        uint32_t idx = (pos + p) & mask;

        // attempt to read the key currently stored
        uint32_t cur = atomicAdd(&table_keys[idx], 0u); // fetch (atomic read)
        if (cur == key) {
            // same key — increment value
            atomicAdd(&table_vals[idx], weight);
            return;
        }
        if (cur == EMPTY_KEY) {
            // try to claim this slot by CAS: EMPTY -> key
            uint32_t old = atomicCAS(&table_keys[idx], EMPTY_KEY, key);
            if (old == EMPTY_KEY || old == key) {
                // we've claimed the slot (or another thread inserted same key concurrently).
                // Add weight to the value. Use atomicAdd because multiple threads may
                // increment same slot concurrently.
                atomicAdd(&table_vals[idx], weight);
                return;
            } else {
                // someone else wrote a different key in the meantime -> continue probing
            }
        } // else cur != EMPTY and cur != key -> continue
    }

    // Probe limit reached: drop the update (or you can accumulate into an overflow).
    // Optionally you can implement a fallback global atomic accumulation here.
}

// Scan a single shared-table and return the best (key, score).
// Returns (best_key, best_score). If no entries, best_key == EMPTY_KEY.
__device__ inline void shared_table_find_best(
    uint32_t table_idx,
    uint32_t* keys_base,
    float* vals_base,
    uint32_t table_capacity,
    uint32_t mask,
    uint32_t* out_best_key,   // out param
    float* out_best_score)    // out param
{
    uint32_t* table_keys = keys_base + (uint64_t)table_idx * table_capacity;
    float* table_vals = vals_base + (uint64_t)table_idx * table_capacity;

    uint32_t best_k = EMPTY_KEY;
    float best_s = 0.0f;

    // Note: scanning is done cooperatively by all threads in a block for efficiency.
    // Here we present a simple single-thread scan; callers should parallelize if needed.
    // We'll implement a warp-cooperative scan pattern below in the kernel skeleton.
    for (uint32_t i = 0; i < table_capacity; ++i) {
        uint32_t k = table_keys[i];
        if (k == EMPTY_KEY) continue;
        float s = table_vals[i];
        if (s > best_s) {
            best_s = s;
            best_k = k;
        }
    }
    *out_best_key = best_k;
    *out_best_score = best_s;
}

// Atomic replace-if-greater for a single node's best score; update best key accordingly.
// We update out_best_score[node] using atomicCAS on the bit representation of float;
// if the CAS succeeds we then set out_best[node] to candidate_key.
//
// Note: This compares floats using > semantics. We do the compare in a loop using
// atomicCAS on the uint32_t representation of the float.
// Returns true if the replacement happened (i.e., we wrote the new score & key).
__device__ inline bool atomic_replace_best_if_greater(
    uint32_t node_index,           // which source node's global slot to try update
    uint32_t candidate_key,
    float candidate_score,
    uint32_t* out_best,            // global array (uint32_t per node)
    float* out_best_score)         // global array (float per node)
{
    uint32_t* score_words = (uint32_t*)(out_best_score + node_index);
    uint32_t old_bits = *score_words;
    float old_score = __int_as_float(old_bits);

    // Fast path: if candidate <= old_score skip the CAS loop
    if (!(candidate_score > old_score)) return false;

    uint32_t new_bits = __float_as_int(candidate_score);

    // CAS loop: try to replace old_bits with new_bits only if old_bits still equal
    // currently observed bits. If the observed value changed, reload and retry only
    // if candidate_score is still greater than the new observed score.
    while (true) {
        uint32_t cur_bits = atomicCAS(score_words, old_bits, new_bits);
        if (cur_bits == old_bits) {
            // we installed new score successfully — now set out_best
            // we use atomicExch to write the key; it's OK if another thread updates
            // the score afterwards — that other thread will also update the key.
            atomicExch(out_best + node_index, candidate_key);
            return true;
        }
        // failed: cur_bits is the new observed value. If candidate <= observed, give up.
        float observed = __int_as_float(cur_bits);
        if (!(candidate_score > observed)) return false;
        // otherwise attempt again with new old_bits
        old_bits = cur_bits;
        // loop
    }
}

// -----------------------------------------------------------------------------
// Kernel skeleton showing how to declare shared memory regions and use the above.
// This skeleton assumes:
//   - tables_per_block : number of node-tables in this block (i.e. number of distinct
//                        source nodes covered by the block).
//   - table_capacity   : capacity (power-of-two) for each table.
// Shared memory layout (contiguous):
//   shared_keys  : uint32_t [ tables_per_block * table_capacity ]
//   shared_vals  : float    [ tables_per_block * table_capacity ]
// -----------------------------------------------------------------------------

extern "C"
__global__
void hyperedge_candidate_kernel(
    const uint32_t* hedge_offsets, // [num_hedges+1]
    const uint32_t* hedges_flat,   // concatenated hyperedge node lists
    const float* hedge_weight,     // per-hyperedge weight
    uint32_t num_hedges,
    uint32_t num_nodes,
    uint32_t* out_best,            // per-node best candidate (global)
    float* out_best_score          // per-node best score (global)
)
{
    // Parameters you choose at launch:
    // - tables_per_block: how many distinct source-nodes this block will handle.
    //   In the hypothesis you said "one warp per hyperedge" — hence tables_per_block
    //   could be equal to the number of hyperedges processed by this block (or number
    //   of distinct source nodes among those hyperedges). For simplicity here, we
    //   assume tables_per_block equals the number of warps in the block (warps_per_block),
    //   and each warp gets a table to accumulate neighbor weights for one source node.
    //
    // - table_capacity must be power-of-two and large enough per your doubled-average heuristic.
    //
    // We'll read these two numbers from the kernel launch via dynamic shared memory size
    // or via compile-time constants. In this sample we assume the caller computed
    // `table_capacity` and `tables_per_block` and placed them in registers before calling;
    // for clarity we hardcode small defaults here.

    const uint32_t table_capacity = 256u; // <-- set by you (power of two)
    const uint32_t tables_per_block = (blockDim.x / WARP_SIZE); // e.g. one table per warp
    const uint32_t mask = table_capacity - 1;

    // Shared memory pointers: we use externally allocated dynamic shared memory (declared in launch)
    // Layout: [ keys (uint32_t) ][ vals (float) ]
    extern __shared__ unsigned char shmem_raw[];
    uint32_t* shared_keys = (uint32_t*)shmem_raw;
    float* shared_vals = (float*)(shmem_raw + sizeof(uint32_t) * ((size_t)tables_per_block * table_capacity));

    // initialize shared table: set keys to EMPTY and vals to 0
    // Done cooperatively by threads in block
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    size_t total_entries = (size_t)tables_per_block * table_capacity;

    // initialize keys
    for (size_t i = tid; i < total_entries; i += threads) {
        shared_keys[i] = EMPTY_KEY;
    }
    // initialize vals
    for (size_t i = tid; i < total_entries; i += threads) {
        shared_vals[i] = 0.0f;
    }
    __syncthreads();

    // --- Hypothetical per-hyperedge processing: ---
    // Example approach: each warp handles one hyperedge (source node -> many targets).
    // Within the warp, threads load nodes and call shared_table_increment for their source's table
    // for each pair (source -> target) with weight.

    // identify this warp's index in the block
    uint32_t warp_id_in_block = (threadIdx.x / WARP_SIZE);
    uint32_t lane = threadIdx.x & (WARP_SIZE - 1);

    // map warp -> hyperedge index. This mapping depends on global offset.
    // For demonstration, assume warp_global_idx = blockIdx.x * warps_per_block + warp_id_in_block
    uint32_t warps_per_block = blockDim.x / WARP_SIZE;
    uint32_t warp_global_idx = blockIdx.x * warps_per_block + warp_id_in_block;

    if (warp_global_idx < num_hedges) {
        // load hyperedge
        uint32_t he_off = hedge_offsets[warp_global_idx];
        uint32_t he_next = hedge_offsets[warp_global_idx + 1];
        uint32_t he_len = he_next - he_off; // >= 2 (source + at least one dest)

        // source node is at hedges_flat[he_off]
        uint32_t src_node = hedges_flat[he_off];
        // the table for this warp is table_idx = warp_id_in_block (one per warp)
        uint32_t table_idx = warp_id_in_block;

        // We will iterate the hyperedge's nodes; each thread processes a subset
        // Thread-lane processes nodes starting from lane, stride=WARP_SIZE
        float he_w = hedge_weight ? hedge_weight[warp_global_idx] : 1.0f;

        // Starting from i=1 to skip source (we don't pair node with itself)
        for (uint32_t i = 1 + lane; i < he_len; i += WARP_SIZE) {
            uint32_t target = hedges_flat[he_off + i];
            // Each thread increments table for the source node with (target, he_w)
            shared_table_increment(table_idx, target, he_w, shared_keys, shared_vals, table_capacity, mask);
        }

        // If hyperedge contains other sources (i.e. multi-source hyperedges) you'll need
        // to repeat this logic for each source. The above code assumes 1 source per hyperedge.

    }
    __syncthreads();

    // --- Collapse / find best per table and write back to global arrays ---
    // We'll let one thread per table find the best (e.g. lane 0 of each warp),
    // or parallelize the scan across threads then reduce. For simplicity, here one
    // thread per table (thread with lane==0 in each warp) will scan the table and update global result.
    if (lane == 0) {
        uint32_t table_idx = warp_id_in_block;
        // Only process tables that belong to this block (some warps may be idle)
        // Compute the corresponding global source node index for this table:
        // (in this example we assumed one warp per hyperedge; thus table->warp_global_idx)
        uint32_t source_hyperedge_idx = blockIdx.x * warps_per_block + table_idx;
        if (source_hyperedge_idx < num_hedges) {
            uint32_t he_off = hedge_offsets[source_hyperedge_idx];
            uint32_t src_node = hedges_flat[he_off];

            // find best candidate from this table
            uint32_t best_k = EMPTY_KEY;
            float best_s = 0.0f;

            // simple scan; table_capacity is expected to be moderate
            uint32_t base = table_idx * table_capacity;
            for (uint32_t i = 0; i < table_capacity; ++i) {
                uint32_t k = shared_keys[base + i];
                if (k == EMPTY_KEY) continue;
                float s = shared_vals[base + i];
                if (s > best_s && k != src_node) { // avoid self-cycle if desired
                    best_s = s;
                    best_k = k;
                }
            }

            if (best_k != EMPTY_KEY) {
                // update global arrays only if we improved the score (atomic replace-if-greater)
                atomic_replace_best_if_greater(src_node, best_k, best_s, out_best, out_best_score);
            }
        }
    }

    // end kernel: cleanup happens automatically as thread block ends
}
