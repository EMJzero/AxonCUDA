#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// USED BY: everyone

#define WARP_SIZE 32u
// TODO: determine this at runtime w.r.t. the mean and variance of the spike frequency!
#define FIXED_POINT_SCALE 262144u // used to convert scores to fixed point

// initialize shared memory: one consecutive chunk per warp
template <typename T>
__device__ __forceinline__ void sm_init(T* sm, const uint32_t size, const T val) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int num_warps = blockDim.x / WARP_SIZE;
    for (int i = warp_id * WARP_SIZE + lane_id; i < size; i += num_warps * WARP_SIZE)
        sm[i] = val;
}

// initialize per-warp shared memory: for shared memory dedicated to a single warp
template <typename T>
__device__ __forceinline__ void wm_init(T* sm, const uint32_t size, const T val) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    for (int i = lane_id; i < size; i += WARP_SIZE)
        sm[i] = val;
}

// initialize local memory
template <typename T>
__device__ __forceinline__ void lm_init(T* lm, const uint32_t size, const T val) {
    for (int i = 0; i < size; i++)
        lm[i] = val;
}

// every warp lane sees the max
template <typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// only lane == 0 sees the max
template <typename T>
__forceinline__ __device__ T warpReduceSumLN0(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


// USED BY: neighborhoods kernel

#define SM_MAX_DEDUPE_BUFFER_SIZE 8192u // 16384 is too big for an A100...


// USED BY: candidates kernel

#define HIST_SIZE 512u // must be a multiple of WARP_SIZE (for the histogram max reduction)
#define HIST_SPARE_SIZE 32u // additional histogram space to reduce hash collisions, must also be a multiple of WARP_SIZE
#define MAX_CANDIDATES 4u // => how many candidates are proposed for a node (ranked by score)

#define DETERMINISTIC_SCORE_NOISE 64u // => adds a +[0, DETERMINISTIC_SCORE_NOISE - 1]/FIXED_POINT_SCALE symmetric noise while calculating pairing scores; set to 0 to disable; keep it a power of 2 otherwise

typedef struct __align__(8) {
    uint32_t node;
    uint32_t score;
} bin;

__forceinline__ __device__ bin warpReduceMax(uint32_t val, uint32_t payload) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        uint32_t other_val = __shfl_xor_sync(0xffffffff, val, offset);
        uint32_t other_payload = __shfl_xor_sync(0xffffffff, payload, offset);
        if (other_val > val || other_val == val && other_payload < payload) {
            val = other_val;
            payload = other_payload;
        }
    }
    return {.node = payload, .score = val};
}

// symmetric and deterministic pseudo-random hash
__device__ __forceinline__ uint32_t deterministic_noise(uint32_t a, uint32_t b) {
    uint32_t lo = min(a, b), hi = max(a, b);
    uint32_t x = lo * 0x9E3779B1u; // golden-ratio :)
    x ^= hi + 0x85EBCA6Bu + (x << 6) + (x >> 2);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 13;
    x *= 0x9E3779B1u;
    x ^= x >> 16;
    return x & (DETERMINISTIC_SCORE_NOISE - 1);
}

// USED BY: grouping kernel

#define MAX_GROUP_SIZE 1u // => MAX_GROUP_SIZE - 1 slots per node; 2 means pairs
#define PATH_SIZE 192u // initial slots for nodes to see while traversing the pairs tree, TODO: automatically extend if needed (costly...)

typedef struct __align__(8) {
    uint32_t id; // lower 32 bits (Nvidia GPUs are little-endian)
    uint32_t score; // converted from float to fixed point! higher 32 bits
} slot;

__device__ __forceinline__ unsigned long long pack_slot(uint32_t score, uint32_t node) {
    // high 32 bits = score, low 32 bits = node
    return ( (unsigned long long)score << 32 ) | ( (unsigned long long)node & 0xFFFFFFFFull );
}

__device__ __forceinline__ void unpack_slot(unsigned long long v, uint32_t &score, uint32_t &node) {
    score = (uint32_t)(v >> 32);
    node = (uint32_t)(v & 0xFFFFFFFFu);
}

// 64bit atomic store to a slot
__device__ __forceinline__ void set_slot(slot* s, uint32_t idx, uint32_t new_node, uint32_t new_score) {
    unsigned long long* s64 = reinterpret_cast<unsigned long long*>(s);
    unsigned long long new_val = pack_slot(new_score, new_node);
    //s64[idx] = new_val; // => needs to be atomic because 64bit stores are not by default...
    //__nv_atomic_store_n(&s64[idx], new_val, 0); // => not cross-version compatible...
    //cuda::std::atomic_ref<unsigned long long> a(* (unsigned long long*) &s64[idx]);
    //a.store(new_val, cuda::std::memory_order_relaxed); // wants "#include <cuda/std/atomic>"
    atomicExch(&s64[idx], new_val);
}

// 64bit atomic max on a slot, returns true iff the slot's content is the new node and score (even if they already were)
__device__ __forceinline__ bool atomic_max_on_slot(slot* s, uint32_t idx, uint32_t new_node, uint32_t new_score) {
    unsigned long long* s64 = reinterpret_cast<unsigned long long*>(s);
    unsigned long long new_val = pack_slot(new_score, new_node);
    unsigned long long read_val = atomicMax(&s64[idx], new_val);
    return read_val <= new_val;
}

// 64bit atomic max on a slot, returns true iff the slot's content was changed from something else to the new node and score
__device__ __forceinline__ bool atomic_max_on_slot_strict(slot* s, uint32_t idx, uint32_t new_node, uint32_t new_score) {
    unsigned long long* s64 = reinterpret_cast<unsigned long long*>(s) + idx;
    unsigned long long new_val = pack_slot(new_score, new_node);
    while (true) {
        unsigned long long old_val = *s64;
        if (old_val >= new_val) return false; // no change needed
        unsigned long long prev = atomicCAS(s64, old_val, new_val);
        if (prev == old_val) return true; // successfully updated
    }
}

// same as 'atomic_max_on_slot' but provides the previous slot's content via 'ret'
__device__ __forceinline__ bool atomic_max_on_slot_ret(slot* s, uint32_t idx, uint32_t new_node, uint32_t new_score, slot &ret) {
    unsigned long long* s64 = reinterpret_cast<unsigned long long*>(s);
    unsigned long long new_val = pack_slot(new_score, new_node);
    unsigned long long read_val = atomicMax(&s64[idx], new_val);
    unpack_slot(read_val, ret.score, ret.id);
    return read_val <= new_val;
}


// USED BY: coarsening routines

// NOTE: these are local memory! No theoretical size limit!
#define MAX_DEDUPE_BUFFER_SIZE 16384u //8192u // for hedges and touching sets
#define MAX_LARGE_DEDUPE_BUFFER_SIZE 32768u // for neighbors


// USED BY: fm refinement kernel

#define PART_HIST_SIZE 64u // best if it is a multiple of WARP_SIZE, best if partitions_per_thread * WARP_SIZE <= num_partitions


// USED BY: refinement constraints checks

// valid values filtering functor
struct masked_value_functor {
    const float* value;
    const uint32_t* valid_1;
    const uint32_t* valid_2;
    __host__ __device__ float operator()(uint32_t i) const { return valid_1[i] == 0 && valid_2[i] == 0 ? value[i] : -FLT_MAX; }
};


// HASH-SET

// NOTE: before using the set, call "sm_init" with HASH_EMPTY as the value!

// TODO: replace all "%" operations in those helpers!!!!

#define HASH_EMPTY 0xFFFFFFFFu

// simple 32-bit hash, should be good enough for a small shared-memory hash-set
__device__ __forceinline__ uint32_t hash_uint32(uint32_t x) {
    x ^= x >> 17;
    x *= 0xED5AD4BBu;
    x ^= x >> 11;
    x *= 0xAC4C1B51u;
    x ^= x >> 15;
    x *= 0x31848BABu;
    x ^= x >> 14;
    return x;
}

// SHARED MEMORY VERSION

// insert a value into a shared-memory hash-set, returns "true" if the value was not in the set before
__device__ __forceinline__ bool sm_hashset_insert(uint32_t* table, const uint32_t size, const uint32_t value) {
    uint32_t h = hash_uint32(value);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot_idx = (idx + probe) % size;
        uint32_t* slot = &table[slot_idx];
        uint32_t old = atomicCAS(slot, HASH_EMPTY, value);
        if (old == HASH_EMPTY)
            return true; // new value
        if (old == value)
            return false; // value already present
        // else: collision with a different value, keep probing
    }
    assert(false && "SM hash-set full!");
    return false; // unreachable, but keeps compiler happy
}

// lookup a value into a shared-memory hash-set, returns "true" if the value was found
__device__ __forceinline__ bool sm_hashset_contains(const uint32_t* table, const uint32_t size, const uint32_t value) {
    uint32_t h = hash_uint32(value);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot_idx = (idx + probe) % size;
        uint32_t cur = table[slot_idx];
        if (cur == value)
            return true;
        if (cur == HASH_EMPTY)
            return false;
        // else: collision, keep probing
    }
    // table full: considered a not-found
    return false;
}

// LOCAL MEMORY VERSION

__device__ __forceinline__ bool lm_hashset_insert(uint32_t* table, const uint32_t size, const uint32_t value) {
    uint32_t h = hash_uint32(value);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot_idx = (idx + probe) % size;
        uint32_t cur = table[slot_idx];
        if (cur == HASH_EMPTY) {
            table[slot_idx] = value;
            return true;
        }
        if (cur == value)
            return false;
    }
    assert(false && "LM hash-set full!");
    return false;
}

__device__ __forceinline__ bool lm_hashset_contains(const uint32_t* table, const uint32_t size, const uint32_t value) {
    uint32_t h = hash_uint32(value);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot_idx = (idx + probe) % size;
        uint32_t cur = table[slot_idx];
        if (cur == value)
            return true;
        if (cur == HASH_EMPTY)
            return false;
    }
    return false;
}


//  HASH-MAP

// NOTE: this reuses defines and hash function from the hash-set above

// SHARED MEMORY VERSION

typedef struct {
    uint32_t key;
    uint32_t value;
} hashmap_entry;

__device__ __forceinline__ bool sm_hashmap_insert(hashmap_entry* table, const uint32_t size, uint32_t key, uint32_t value) {
    uint32_t h = hash_uint32(key);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot = (idx + probe) % size;
        uint32_t* key_ptr = &table[slot].key;
        uint32_t old_key = atomicCAS(key_ptr, HASH_EMPTY, key);
        if (old_key == HASH_EMPTY) { // insert new value
            table[slot].value = value;
            return true;
        }
        if (old_key == key) { // update existing value
            table[slot].value = value;
            return false;
        }
        // else: collision, continue probing
    }
    assert(false && "SM hash-map full!");
    return false;
}

__device__ __forceinline__ bool sm_hashmap_unsafe_insert(hashmap_entry* table, const uint32_t size, uint32_t key, uint32_t value) {
    uint32_t h = hash_uint32(key);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot = (idx + probe) % size;
        uint32_t old_key = table[slot].key;
        if (old_key == HASH_EMPTY) { // insert new value
            table[slot] = { key, value };
            return true;
        }
        if (old_key == key) { // update existing value
            table[slot].value = value;
            return false;
        }
        // else: collision, keep probing
    }
    assert(false && "SM hash-map full!");
    return false; 
}

__device__ __forceinline__ bool sm_hashmap_lookup(const hashmap_entry* table, const uint32_t size, uint32_t key, uint32_t* out_value) {
    uint32_t h = hash_uint32(key);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot = (idx + probe) % size;
        uint32_t cur_key = table[slot].key;
        if (cur_key == key) {
            *out_value = table[slot].value;
            return true;
        }
        if (cur_key == HASH_EMPTY)
            return false;
        // else: collision, keep probing
    }
    // table full: considered a not-found
    return false; 
}

__device__ __forceinline__ bool sm_hashmap_increment(hashmap_entry* table, const uint32_t size, uint32_t key, uint32_t inc) {
    uint32_t h = hash_uint32(key);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot = (idx + probe) % size;
        uint32_t cur_key = table[slot].key;
        if (cur_key == key) {
            table[slot].value += inc;
            return true;
        }
        if (cur_key == HASH_EMPTY)
            return false;
        // else: collision, keep probing
    }
    // table full: considered a not-found
    return false; 
}

// LOCAL MEMORY VERSION

__device__ __forceinline__ bool lm_hashmap_insert(hashmap_entry* table, const uint32_t size, uint32_t key, uint32_t value) {
    uint32_t h = hash_uint32(key);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot = (idx + probe) % size;
        uint32_t cur_key = table[slot].key;
        if (cur_key == HASH_EMPTY) {
            table[slot].key   = key;
            table[slot].value = value;
            return true;
        }
        if (cur_key == key) {
            table[slot].value = value;
            return false;
        }
    }
    assert(false && "LM hash-map full!");
    return false;
}

__device__ __forceinline__ bool lm_hashmap_lookup(const hashmap_entry* table, const uint32_t size, uint32_t key, uint32_t* out_value) {
    uint32_t h = hash_uint32(key);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot = (idx + probe) % size;
        uint32_t cur_key = table[slot].key;
        if (cur_key == key) {
            *out_value = table[slot].value;
            return true;
        }
        if (cur_key == HASH_EMPTY)
            return false;
    }
    return false;
}


// SORTING and SEARCHING

__device__ __forceinline__ void swap32(uint32_t &a, uint32_t &b) { uint32_t t = a; a = b; b = t; }

__device__ __forceinline__ int ilog2(int x) {
    int r = 0;
    while (x >>= 1) r++;
    return r;
}

// insertion-sort fallback (32 elements or less)
__device__ __forceinline__ void insertion_sort(uint32_t* a, int lo, int hi) {
    for (int i = lo + 1; i <= hi; i++) {
        uint32_t key = a[i];
        int j = i - 1;
        while (j >= lo && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

__device__ __forceinline__ void heapify(uint32_t* a, int n, int i) {
    while (true) {
        int l = (i << 1) + 1;
        int r = l + 1;
        int largest = i;
        
        if (l < n && a[l] > a[largest]) largest = l;
        if (r < n && a[r] > a[largest]) largest = r;

        if (largest == i) return;

        swap32(a[i], a[largest]);
        i = largest;
    }
}

// heapsort fallback
__device__ __forceinline__ void heapsort(uint32_t* a, int n) {
    for (int i = (n >> 1) - 1; i >= 0; i--)
        heapify(a, n, i);

    for (int i = n - 1; i > 0; i--) {
        swap32(a[0], a[i]);
        heapify(a, i, 0);
    }
}

// sequential sort in one thread: entry point
__device__ __forceinline__ void introsort(uint32_t* a, int n) {
    if (n <= 32) {
        insertion_sort(a, 0, n - 1);
        return;
    }

    int max_depth = (ilog2(n) << 1);

    int lo_stack[32], hi_stack[32], sp = 0;
    lo_stack[0] = 0;
    hi_stack[0] = n - 1;

    while (sp >= 0) {

        int lo = lo_stack[sp];
        int hi = hi_stack[sp];
        sp--;

        while (hi - lo > 32) {

            if (max_depth-- == 0) {
                heapsort(a + lo, hi - lo + 1);
                goto next_segment;
            }

            // median-of-3 pivot selection
            int mid = (lo + hi) >> 1;
            uint32_t p = a[mid];
            if (a[lo] > p) swap32(a[lo], a[mid]);
            if (a[mid] > a[hi]) swap32(a[mid], a[hi]);
            if (a[lo] > a[mid]) swap32(a[lo], a[mid]);
            p = a[mid];

            int i = lo;
            int j = hi;

            while (true) {
                while (a[i] < p) i++;
                while (a[j] > p) j--;
                if (i >= j) break;
                swap32(a[i], a[j]);
                i++; j--;
            }

            if (j - lo < hi - i) {
                if (lo < j) {
                    sp++;
                    lo_stack[sp] = lo;
                    hi_stack[sp] = j;
                }
                lo = i;
            } else {
                if (i < hi) {
                    sp++;
                    lo_stack[sp] = i;
                    hi_stack[sp] = hi;
                }
                hi = j;
            }
        }
        insertion_sort(a, lo, hi);
    next_segment:;
    }
}

// bitonic sort within a single warp over a portion of shared memory
// IMPORTANT: 'N' must be a multiple of 'WARP_SIZE' !!
template <typename T, uint32_t N>
__device__ __forceinline__ void wm_bitonic_sort(T* sm) {
    static_assert(N % WARP_SIZE == 0);
    constexpr uint32_t STRIDE = WARP_SIZE;
    constexpr uint32_t BLOCKS = N / WARP_SIZE;
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
    #pragma unroll
    for (uint32_t k = 2; k <= N; k <<= 1) {
        #pragma unroll
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            #pragma unroll
            for (uint32_t b = 0; b < BLOCKS; ++b) {
                uint32_t i = lane + b * STRIDE;
                uint32_t ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    T a = sm[i];
                    T bval = sm[ixj];
                    if ((a > bval) == ascending) {
                        sm[i] = bval;
                        sm[ixj] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}



// binary search, returns the index of 'value' in 'a' or UINT32_MAX if it is not found
template <typename T>
__device__ __forceinline__ uint32_t binary_search(const T* a, uint32_t n, T value) {
    int lo = 0;
    int hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        T v = a[mid];
        if (v < value)
            lo = mid + 1;
        else if (v > value)
            hi = mid - 1;
        else
            return mid; // found -> return idx
    }
    return UINT32_MAX; // not found
}