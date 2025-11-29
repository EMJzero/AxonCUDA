#pragma once
#include <stdint.h>

// USED BY: everyone

#define WARP_SIZE 32u

// initialize shared memory: one consecutive chunk per warp
__device__ __forceinline__ void sm_init(uint32_t* sm, const uint32_t size, const uint32_t val) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    for (int i = warp_id * WARP_SIZE + lane_id; i < size; i += num_warps * WARP_SIZE)
        sm[i] = val;
}


// USED BY: neighborhoods kernel

#define SM_MAX_DEDUPE_BUFFER_SIZE 8192 // 16384 is too big for an A100...

__forceinline__ __device__ uint32_t warpReduceSumInt(uint32_t val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


// USED BY: candidates kernel

#define HIST_SIZE 64u // must be a multiple of WARP_SIZE (for the histogram max reduction)

typedef struct {
    uint32_t node;
    float score;
} bin;

__forceinline__ __device__ float warpReduceSumFloat(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__forceinline__ __device__ bin warpReduceMax(float val, uint32_t payload) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        uint32_t other_payload = __shfl_down_sync(0xffffffff, payload, offset);
        if (other_val > val) {
            val = other_val;
            payload = other_payload;
        }
    }
    return {.node = payload, .score = val};
}


// USED BY: grouping kernel

#define MAX_GROUP_SIZE 4 // => MAX_GROUP_SIZE - 1 slots per node
// TODO: determine this at runtime w.r.t. the mean and variance of the spike frequency!
#define FIXED_POINT_SCALE 1024 // used to convert scores to fixed point
#define PATH_SIZE 16384 // 128 // initial slots for nodes to see while traversing the pairs three, TODO: automatically extend if needed (costly...)

typedef struct __align__(8) {
    uint32_t id; // lower 32 bits (Nvidia GPUs are little-endian)
    uint32_t score; // converted from float to fixed point! higher 32 bits
} slot;

__device__ __forceinline__ unsigned long long pack_slot(uint32_t score, uint32_t node) {
    // high 32 bits = score, low 32 bits = node
    return ( (unsigned long long)score << 32 ) | ( (unsigned long long)node  & 0xffffffffull );
}

__device__ __forceinline__ void unpack_slot(unsigned long long v, uint32_t &score, uint32_t &node) {
    score = (uint32_t)(v >> 32);
    node = (uint32_t)(v & 0xffffffffu);
}

__device__ __forceinline__ bool atomic_max_on_slot(slot* s, uint32_t idx, uint32_t new_node, uint32_t new_score) {
    unsigned long long* s64 = reinterpret_cast<unsigned long long*>(s);
    unsigned long long new_val = pack_slot(new_score, new_node);
    unsigned long long read_val = atomicMax(&s64[idx], new_val);
    return read_val <= new_val;
}

__device__ __forceinline__ bool atomic_max_on_slot_ret(slot* s, uint32_t idx, uint32_t new_node, uint32_t new_score, slot &ret) {
    unsigned long long* s64 = reinterpret_cast<unsigned long long*>(s);
    unsigned long long new_val = pack_slot(new_score, new_node);
    unsigned long long read_val = atomicMax(&s64[idx], new_val);
    unpack_slot(read_val, ret.score, ret.id);
    return read_val <= new_val;
}


// USED BY: coarsening routines

#define MAX_DEDUPE_BUFFER_SIZE 8192 // 16384 is too big for an A100...


// SHARED MEMORY HASH-SET

// NOTE: before using the set, call "sm_init" with SM_HASH_EMPTY as the value!

#define SM_HASH_EMPTY 0xFFFFFFFFu

// simple 32-bit hash, should be good enough for a small shared-memory hash-set
__device__ __forceinline__ uint32_t sm_hash_uint32(uint32_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

// insert a value into a shared-memory hash-set, returns "true" if the value was not in the set before
__device__ __forceinline__ bool sm_hashset_insert(uint32_t* table, const uint32_t size, const uint32_t value) {
    uint32_t h = sm_hash_uint32(value);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot_idx = (idx + probe) % size;
        uint32_t* slot = &table[slot_idx];
        uint32_t old = atomicCAS(slot, SM_HASH_EMPTY, value);
        if (old == SM_HASH_EMPTY)
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
    uint32_t h = sm_hash_uint32(value);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot_idx = (idx + probe) % size;
        uint32_t cur = table[slot_idx];
        if (cur == value)
            return true;
        if (cur == SM_HASH_EMPTY)
            return false;
        // else: collision, keep probing
    }
    // table full: considered a not-found
    return false;
}


// SHARED MEMORY HASH-MAP

// NOTE: this reuses defines and hash function from the hash-set above

typedef struct {
    uint32_t key;
    uint32_t value;
} hashmap_entry;

__device__ __forceinline__ bool sm_hashmap_insert(hashmap_entry* table, const uint32_t size, uint32_t key, uint32_t value) {
    uint32_t h = sm_hash_uint32(key);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot = (idx + probe) % size;
        uint32_t* key_ptr = &table[slot].key;
        uint32_t old_key  = atomicCAS(key_ptr, SM_HASH_EMPTY, key);
        if (old_key == SM_HASH_EMPTY) { // insert new value
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

__device__ __forceinline__ bool sm_hashmap_lookup(const hashmap_entry* table, const uint32_t size, uint32_t key, uint32_t* out_value) {
    uint32_t h = sm_hash_uint32(key);
    int idx = h % size;
    for (int probe = 0; probe < size; ++probe) {
        int slot = (idx + probe) % size;
        uint32_t cur_key = table[slot].key;
        if (cur_key == key) {
            *out_value = table[slot].value;
            return true;
        }
        if (cur_key == SM_HASH_EMPTY) {
            return false;
        }
        // else: collision, keep probing
    }
    // table full: considered a not-found
    return false; 
}