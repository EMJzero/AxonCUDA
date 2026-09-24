#pragma once
#include <atomic>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "defines.hpp"
#include "data_types.hpp"

// USED BY: everyone

//     verbosity level vs config flags
// v | info | err | launch | log | dbg |
// --|------|-----|--------|-----|-----|
// 0 |      |     |        |     |     |
// 1 |   x  |  x  |        |     |     |
// 2 |   x  |  x  |    x   |     |     |
// 3 |   x  |  x  |    x   |  x  |     |
// 4 |   x  |  x  |    x   |  x  |  x  |

// insight in algorithms' logic and decisions
#define LOG(cfg) \
    if (!cfg.verbose_logs) {} else

// what the program is up to now, the algorithm's step and phase
#define INFO(cfg) \
    if (!cfg.verbose_info) {} else

// error and warnings (on stderr)
#define ERR(cfg) \
    if (!cfg.verbose_errs_and_warns) {} else

// parallel loops ("kernel launches")
#define LAUNCH(cfg) \
    if (!cfg.verbose_kernel_launches) {} else std::cout

// extra debug code
#define DBG(cfg) \
    if (!cfg.debug) {} else

// additional common logging text

#define RUN \
    << "Running "

#define TID(tid) \
    << "[tid=" << (tid) << "] "


// ATOMICS
// => relaxed ordering everywhere: every parallel loop ends with an implicit barrier, that is the only synchronization we rely upon

template <typename T>
inline T atomic_add(T* __restrict__ ptr, const T val) {
    return std::atomic_ref<T>(*ptr).fetch_add(val, std::memory_order_relaxed);
}

template <typename T>
inline T atomic_sub(T* __restrict__ ptr, const T val) {
    return std::atomic_ref<T>(*ptr).fetch_sub(val, std::memory_order_relaxed);
}

template <typename T>
inline T atomic_load(const T* __restrict__ ptr) {
    return std::atomic_ref<T>(*const_cast<T*>(ptr)).load(std::memory_order_relaxed);
}

template <typename T>
inline void atomic_store(T* __restrict__ ptr, const T val) {
    std::atomic_ref<T>(*ptr).store(val, std::memory_order_relaxed);
}

// returns the previous value
template <typename T>
inline T atomic_max(T* __restrict__ ptr, const T val) {
    std::atomic_ref<T> ref(*ptr);
    T old = ref.load(std::memory_order_relaxed);
    while (old < val && !ref.compare_exchange_weak(old, val, std::memory_order_relaxed));
    return old;
}

// returns true iff the swap happened
template <typename T>
inline bool atomic_cas(T* __restrict__ ptr, T expected, const T desired) {
    return std::atomic_ref<T>(*ptr).compare_exchange_strong(expected, desired, std::memory_order_relaxed);
}


// WARP EMULATION
// NOTE: in CUDA a warp's lanes accumulate floats separately, then reduce them with shuffles, and float addition is not
//       associative: wherever the result depends on it, the same lanes and the same shuffle tree are replayed here

// lane of the idx-th item, when lanes accumulate items idx, idx + WARP_SIZE, idx + 2*WARP_SIZE, ...
#define LANE_OF(idx) ((idx) & (WARP_SIZE - 1))

// same tree as 'warpReduceSum' (xor butterfly), every lane would see the same sum
template <typename T>
inline T lanesReduceSum(const T (&lanes)[WARP_SIZE]) {
    T val[WARP_SIZE], tmp[WARP_SIZE];
    for (uint32_t lane = 0; lane < WARP_SIZE; lane++) val[lane] = lanes[lane];
    for (uint32_t offset = 1; offset < WARP_SIZE; offset <<= 1) {
        for (uint32_t lane = 0; lane < WARP_SIZE; lane++) tmp[lane] = val[lane] + val[lane ^ offset];
        for (uint32_t lane = 0; lane < WARP_SIZE; lane++) val[lane] = tmp[lane];
    }
    return val[0];
}

// same tree as 'warpReduceSumLN0' (shuffle down), only lane 0's sum is returned
template <typename T>
inline T lanesReduceSumLN0(const T (&lanes)[WARP_SIZE]) {
    T val[WARP_SIZE];
    for (uint32_t lane = 0; lane < WARP_SIZE; lane++) val[lane] = lanes[lane];
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        for (uint32_t lane = 0; lane < offset; lane++) // lanes >= offset never feed lane 0 anymore
            val[lane] += val[lane + offset];
    return val[0];
}


// HASHING

// simple 32-bit hash
inline uint32_t hash_uint32(uint32_t x) {
    x ^= x >> 17;
    x *= 0xED5AD4BBu;
    x ^= x >> 11;
    x *= 0xAC4C1B51u;
    x ^= x >> 15;
    x *= 0x31848BABu;
    x ^= x >> 14;
    return x;
}

// symmetric and deterministic pseudo-random hash
template <uint32_t MAX_NOISE>
inline uint32_t deterministic_noise(uint32_t a, uint32_t b) {
    static_assert((MAX_NOISE & (MAX_NOISE - 1)) == 0, "MAX_NOISE must be power-of-two");
    uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
    uint32_t x = lo * 0x9E3779B1u; // golden-ratio :)
    x ^= hi + 0x85EBCA6Bu + (x << 6) + (x >> 2);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 13;
    x *= 0x9E3779B1u;
    x ^= x >> 16;
    return x & (MAX_NOISE - 1);
}


// USED BY: grouping kernel

inline slot pack_slot(uint32_t score, uint32_t node) {
    // high 32 bits = score, low 32 bits = node
    return ((slot)score << 32) | ((slot)node & 0xFFFFFFFFull);
}

inline uint32_t slot_score(const slot s) { return (uint32_t)(s >> 32); }
inline uint32_t slot_id(const slot s) { return (uint32_t)(s & 0xFFFFFFFFull); }


// MISC

// number of bits needed to represent 'x'
inline uint32_t bits_for(const uint64_t x) {
    return x == 0 ? 0u : 64u - (uint32_t)__builtin_clzll(x);
}
