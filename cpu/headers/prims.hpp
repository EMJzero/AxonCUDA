#pragma once
#include <memory>
#include <vector>
#include <cstdint>
#include <cstring>
#include <utility>
#include <algorithm>

#include <omp.h>

#include "defines.hpp"
#include "data_types.hpp"

// NOTE: in CUDA these are thrust/CUB calls, here they are plain parallel loops
// => every primitive is deterministic, its result does not depend on the number of threads
// => integer results are identical to thrust/CUB ones, float scans and reductions associate differently (see README)

// NOTE: primitives split their items in one contiguous chunk per thread, and run sequentially below PARALLEL_GRAIN items


// MEMORY

// owning, fixed-size, uninitialized array
// => uninitialized on purpose: the first write, done in parallel, places each page on the NUMA node of its writer
template <typename T>
class buffer {
    std::unique_ptr<T[]> data_;
    dim_t size_ = 0;

    public:
    buffer() = default;
    explicit buffer(const dim_t size) : data_(size > 0 ? std::make_unique_for_overwrite<T[]>(size) : nullptr), size_(size) {}
    buffer(buffer&&) noexcept = default;
    buffer& operator=(buffer&&) noexcept = default;
    buffer(const buffer&) = delete;
    buffer& operator=(const buffer&) = delete;

    T* data() { return data_.get(); }
    const T* data() const { return data_.get(); }
    dim_t size() const { return size_; }
    T& operator[](const dim_t idx) { return data_[idx]; }
    const T& operator[](const dim_t idx) const { return data_[idx]; }
    void release() { data_.reset(); size_ = 0; } // free the array early, as a "cudaFree"
};

template <typename T>
void par_fill(T* __restrict__ data, const dim_t n, const T val) {
    #pragma omp parallel for schedule(static) if(n > PARALLEL_GRAIN)
    for (dim_t i = 0; i < n; i++)
        data[i] = val;
}

template <typename T>
void par_copy(T* __restrict__ dst, const T* __restrict__ src, const dim_t n) {
    #pragma omp parallel for schedule(static) if(n > PARALLEL_GRAIN)
    for (dim_t i = 0; i < n; i++)
        dst[i] = src[i];
}

template <typename T>
void par_sequence(T* __restrict__ data, const dim_t n) {
    #pragma omp parallel for schedule(static) if(n > PARALLEL_GRAIN)
    for (dim_t i = 0; i < n; i++)
        data[i] = (T)i;
}

// out[i] = in[map[i]]
template <typename T, typename I>
void par_gather(const I* __restrict__ map, const dim_t n, const T* __restrict__ in, T* __restrict__ out) {
    #pragma omp parallel for schedule(static) if(n > PARALLEL_GRAIN)
    for (dim_t i = 0; i < n; i++)
        out[i] = in[map[i]];
}


// SCANS
// => two passes: each thread scans its chunk, then each chunk adds the carry of all chunks before it
// => items are read via 'get(i)' and written via 'set(i, val)', so that a scan can run over a field of a struct

// in-place inclusive scan
template <typename T, typename Get, typename Set, typename Op>
void par_inclusive_scan_with(const dim_t n, Get get, Set set, Op op) {
    if (n == 0) return;
    if (n <= PARALLEL_GRAIN) {
        T acc = get(0);
        for (dim_t i = 1; i < n; i++) {
            acc = op(acc, get(i));
            set(i, acc);
        }
        return;
    }
    std::vector<T> carry(omp_get_max_threads()); // carry[tid] -> total of tid's chunk, then total of all chunks up to tid's
    #pragma omp parallel
    {
        // STYLE: one chunk per thread!
        const int num_threads = omp_get_num_threads(), tid = omp_get_thread_num();
        const dim_t begin = n * tid / num_threads, end = n * (tid + 1) / num_threads; // HP: n > num_threads -> no empty chunk
        T acc = get(begin);
        for (dim_t i = begin + 1; i < end; i++) {
            acc = op(acc, get(i));
            set(i, acc);
        }
        carry[tid] = acc;
        #pragma omp barrier
        #pragma omp single
        {
            for (int t = 1; t < num_threads; t++)
                carry[t] = op(carry[t - 1], carry[t]);
        }
        if (tid > 0) {
            const T prefix = carry[tid - 1];
            for (dim_t i = begin; i < end; i++)
                set(i, op(prefix, get(i)));
        }
    }
}

// in-place exclusive sum-scan
template <typename T, typename Get, typename Set>
void par_exclusive_scan_with(const dim_t n, Get get, Set set) {
    if (n == 0) return;
    if (n <= PARALLEL_GRAIN) {
        T acc = T{};
        for (dim_t i = 0; i < n; i++) {
            const T val = get(i);
            set(i, acc);
            acc = acc + val;
        }
        return;
    }
    std::vector<T> carry(omp_get_max_threads()); // carry[tid] -> total of tid's chunk, then total of all chunks before tid's
    #pragma omp parallel
    {
        // STYLE: one chunk per thread!
        const int num_threads = omp_get_num_threads(), tid = omp_get_thread_num();
        const dim_t begin = n * tid / num_threads, end = n * (tid + 1) / num_threads;
        T acc = T{};
        for (dim_t i = begin; i < end; i++)
            acc = acc + get(i);
        carry[tid] = acc;
        #pragma omp barrier
        #pragma omp single
        {
            T prefix = T{};
            for (int t = 0; t < num_threads; t++) {
                const T total = carry[t];
                carry[t] = prefix;
                prefix = prefix + total;
            }
        }
        acc = carry[tid];
        for (dim_t i = begin; i < end; i++) {
            const T val = get(i);
            set(i, acc);
            acc = acc + val;
        }
    }
}

template <typename T>
void par_inclusive_scan(T* __restrict__ data, const dim_t n) {
    par_inclusive_scan_with<T>(n, [=](dim_t i) { return data[i]; }, [=](dim_t i, T v) { data[i] = v; }, [](T a, T b) { return a + b; });
}

template <typename T, typename Op>
void par_inclusive_scan(T* __restrict__ data, const dim_t n, Op op) {
    par_inclusive_scan_with<T>(n, [=](dim_t i) { return data[i]; }, [=](dim_t i, T v) { data[i] = v; }, op);
}

template <typename T>
void par_exclusive_scan(T* __restrict__ data, const dim_t n) {
    par_exclusive_scan_with<T>(n, [=](dim_t i) { return data[i]; }, [=](dim_t i, T v) { data[i] = v; });
}

// in-place inclusive sum-scan by key, restarting at every segment
// => 'same(i - 1, i)' is true iff items i - 1 and i share a segment (have the same key)
template <typename T, typename Same>
void par_inclusive_scan_by_key(T* __restrict__ data, const dim_t n, Same same) {
    if (n == 0) return;
    if (n <= PARALLEL_GRAIN) {
        for (dim_t i = 1; i < n; i++)
            if (same(i - 1, i)) data[i] += data[i - 1];
        return;
    }
    const int max_threads = omp_get_max_threads();
    std::vector<T> carry(max_threads); // carry[tid] -> value to add to the first segment of tid's chunk
    std::vector<uint8_t> closed(max_threads); // closed[tid] -> true if a segment starts inside tid's chunk (past its first item)
    #pragma omp parallel
    {
        // STYLE: one chunk per thread!
        const int num_threads = omp_get_num_threads(), tid = omp_get_thread_num();
        const dim_t begin = n * tid / num_threads, end = n * (tid + 1) / num_threads;
        bool my_closed = false;
        for (dim_t i = begin + 1; i < end; i++) {
            if (same(i - 1, i)) data[i] += data[i - 1];
            else my_closed = true;
        }
        closed[tid] = my_closed;
        #pragma omp barrier
        #pragma omp single
        {
            // a chunk's carry is the sum of its first segment in all chunks before it
            carry[0] = T{};
            for (int t = 1; t < num_threads; t++) {
                const dim_t chunk_begin = n * t / num_threads;
                if (!same(chunk_begin - 1, chunk_begin)) carry[t] = T{};
                else carry[t] = data[chunk_begin - 1] + (closed[t - 1] ? T{} : carry[t - 1]); // data[chunk_begin - 1] holds the previous chunk's last segment sum
            }
        }
        const T my_carry = carry[tid];
        if (tid > 0 && same(begin - 1, begin))
            for (dim_t i = begin; i < end && (i == begin || same(i - 1, i)); i++)
                data[i] += my_carry;
    }
}


// COMPACTION

// indices in [0, n) that satisfy 'pred(i)', in increasing order
template <typename Pred>
buffer<uint32_t> par_copy_if(const uint32_t n, Pred pred) {
    buffer<uint32_t> flags((dim_t)n + 1); // flags[i + 1] -> 1 if i satisfies the predicate, then (after the scan) its idx in the output + 1
    flags[0] = 0;
    #pragma omp parallel for schedule(static) if(n > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < n; i++)
        flags[(dim_t)i + 1] = pred(i) ? 1u : 0u;
    par_inclusive_scan<uint32_t>(flags.data(), (dim_t)n + 1);
    buffer<uint32_t> selected(flags[n]);
    #pragma omp parallel for schedule(static) if(n > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < n; i++)
        if (flags[(dim_t)i + 1] != flags[i]) selected[flags[i]] = i;
    return selected;
}


// SORTING

// stable LSD radix sort of (key, value) pairs, over the lowest 'key_bits' bits of the keys, 8 bits per pass
// => each pass: one digit histogram per chunk, digit-major and chunk-minor offsets (they keep the sort stable), scatter
template <typename V>
void par_radix_sort_pairs(uint64_t* keys, V* vals, const dim_t n, const uint32_t key_bits) {
    if (n < 2 || key_bits == 0) return;
    constexpr uint32_t RADIX_BITS = 8u;
    constexpr uint32_t RADIX = 1u << RADIX_BITS;
    const uint32_t passes = (key_bits + RADIX_BITS - 1) / RADIX_BITS;
    buffer<uint64_t> keys_alt(n);
    buffer<V> vals_alt(n);
    uint64_t *keys_in = keys, *keys_out = keys_alt.data();
    V *vals_in = vals, *vals_out = vals_alt.data();
    std::vector<dim_t> hist((size_t)omp_get_max_threads() * RADIX); // hist[tid * RADIX + digit] -> count of 'digit' in tid's chunk, then its scatter offset
    bool skip = false; // true if all keys share the current digit
    #pragma omp parallel if(n > PARALLEL_GRAIN)
    {
        // STYLE: one chunk per thread!
        const int num_threads = omp_get_num_threads(), tid = omp_get_thread_num();
        const dim_t begin = n * tid / num_threads, end = n * (tid + 1) / num_threads;
        dim_t* my_hist = hist.data() + (size_t)tid * RADIX;
        for (uint32_t pass = 0; pass < passes; pass++) {
            const uint32_t shift = pass * RADIX_BITS;
            for (uint32_t d = 0; d < RADIX; d++) my_hist[d] = 0;
            for (dim_t i = begin; i < end; i++) my_hist[(keys_in[i] >> shift) & (RADIX - 1)]++;
            #pragma omp barrier
            #pragma omp single
            {
                dim_t offset = 0;
                skip = false;
                for (uint32_t d = 0; d < RADIX; d++) {
                    dim_t digit_total = 0;
                    for (int t = 0; t < num_threads; t++) {
                        const dim_t count = hist[(size_t)t * RADIX + d];
                        hist[(size_t)t * RADIX + d] = offset;
                        offset += count;
                        digit_total += count;
                    }
                    if (digit_total == n) skip = true;
                }
            }
            if (!skip) {
                for (dim_t i = begin; i < end; i++) {
                    const dim_t pos = my_hist[(keys_in[i] >> shift) & (RADIX - 1)]++;
                    keys_out[pos] = keys_in[i];
                    vals_out[pos] = vals_in[i];
                }
                #pragma omp barrier
                #pragma omp single
                {
                    std::swap(keys_in, keys_out);
                    std::swap(vals_in, vals_out);
                }
            }
        }
    }
    if (keys_in != keys) {
        par_copy(keys, keys_in, n);
        par_copy(vals, vals_in, n);
    }
}

// stable sort of items [0, n) by 'key(i)', returns the sorting permutation
// => perm[pos] -> item in sorted position pos (items with the same key stay in increasing order)
template <typename Key>
buffer<uint32_t> par_sort_permutation(const dim_t n, const uint32_t key_bits, Key key) {
    buffer<uint64_t> keys(n);
    buffer<uint32_t> perm(n);
    #pragma omp parallel for schedule(static) if(n > PARALLEL_GRAIN)
    for (dim_t i = 0; i < n; i++) {
        keys[i] = key(i);
        perm[i] = (uint32_t)i;
    }
    par_radix_sort_pairs<uint32_t>(keys.data(), perm.data(), n, key_bits);
    return perm;
}

// reorder 'data' along a sorting permutation: data[pos] <- data[perm[pos]]
template <typename T>
void par_permute(const uint32_t* __restrict__ perm, const dim_t n, buffer<T> &data) {
    buffer<T> permuted(n);
    par_gather(perm, n, data.data(), permuted.data());
    data = std::move(permuted);
}


// REDUCTIONS

// index of the maximum in [0, n) w.r.t. 'less(a, b)' (true -> b is greater), the first one in case of ties
template <typename Less>
uint32_t par_max_element(const uint32_t n, Less less) {
    if (n == 0) return 0u;
    std::vector<uint32_t> best(omp_get_max_threads(), UINT32_MAX); // best[tid] -> index of the maximum in tid's chunk
    int used_threads = 1;
    #pragma omp parallel if(n > PARALLEL_GRAIN)
    {
        // STYLE: one chunk per thread!
        const int num_threads = omp_get_num_threads(), tid = omp_get_thread_num();
        const uint32_t begin = (uint32_t)((dim_t)n * tid / num_threads), end = (uint32_t)((dim_t)n * (tid + 1) / num_threads);
        if (begin < end) {
            uint32_t my_best = begin;
            for (uint32_t i = begin + 1; i < end; i++)
                if (less(my_best, i)) my_best = i;
            best[tid] = my_best;
        }
        #pragma omp single nowait
        used_threads = num_threads;
    }
    uint32_t global_best = UINT32_MAX;
    for (int t = 0; t < used_threads; t++) {
        if (best[t] == UINT32_MAX) continue;
        if (global_best == UINT32_MAX || less(global_best, best[t])) global_best = best[t];
    }
    return global_best;
}

// reduction of 'get(i)' over [0, n) with the (associative and commutative) operator 'op'
template <typename T, typename Get, typename Op>
T par_reduce(const dim_t n, const T init, Get get, Op op) {
    std::vector<T> partial(omp_get_max_threads(), init); // partial[tid] -> reduction of tid's chunk
    int used_threads = 1;
    #pragma omp parallel if(n > PARALLEL_GRAIN)
    {
        // STYLE: one chunk per thread!
        const int num_threads = omp_get_num_threads(), tid = omp_get_thread_num();
        const dim_t begin = n * tid / num_threads, end = n * (tid + 1) / num_threads;
        T acc = init;
        for (dim_t i = begin; i < end; i++)
            acc = op(acc, get(i));
        partial[tid] = acc;
        #pragma omp single nowait
        used_threads = num_threads;
    }
    T acc = init;
    for (int t = 0; t < used_threads; t++)
        acc = op(acc, partial[t]);
    return acc;
}


// CSR CONSTRUCTION

// build a CSR structure (offsets + packed segments) when the size of each item's segment is not known in advance
// => each thread appends the segments of its items to its own buffer, then segments are packed at their final offsets
// NOTE: in CUDA this is done in two passes, count then scatter (or dedupe in an oversized buffer then pack), since a kernel
//       cannot grow a buffer; here one pass suffices, but segments are copied once more
//
// usage, by the thread that builds the item's segment:
//   std::vector<uint32_t> &segment = builder.open(item);
//   segment.push_back(...);
//   builder.close(item);
class csr_builder {
    std::vector<std::vector<uint32_t>> local_; // local_[tid] -> concatenation of the segments built by thread tid
    buffer<uint32_t> owner_; // owner_[item] -> thread that built the item's segment
    buffer<dim_t> start_; // start_[item] -> start idx of the item's segment in its owner's buffer

    public:
    const uint32_t num_items;
    buffer<dim_t> offsets; // offsets[item] -> start idx of the item's segment in the packed array (num_items + 1 entries)

    explicit csr_builder(const uint32_t num_items) : local_(omp_get_max_threads()), owner_(num_items), start_(num_items), num_items(num_items), offsets((dim_t)num_items + 1) {}

    // begin the item's segment, returns the calling thread's buffer, where to append the segment's entries
    std::vector<uint32_t>& open(const uint32_t item) {
        const int tid = omp_get_thread_num();
        owner_[item] = (uint32_t)tid;
        start_[item] = local_[tid].size();
        return local_[tid];
    }

    // end the item's segment
    void close(const uint32_t item) {
        offsets[(dim_t)item + 1] = local_[owner_[item]].size() - start_[item];
    }

    // scan segment sizes into offsets, copy each segment at its offset, returns the packed array
    buffer<uint32_t> pack() {
        offsets[0] = 0;
        par_inclusive_scan<dim_t>(offsets.data(), (dim_t)num_items + 1);
        buffer<uint32_t> packed(offsets[num_items]);
        // STYLE: one item (segment) per iteration!
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK) if(num_items > PARALLEL_GRAIN)
        for (uint32_t item = 0; item < num_items; item++) {
            const dim_t len = offsets[(dim_t)item + 1] - offsets[item];
            if (len > 0) std::memcpy(packed.data() + offsets[item], local_[owner_[item]].data() + start_[item], len * sizeof(uint32_t));
        }
        std::vector<std::vector<uint32_t>>().swap(local_);
        owner_.release();
        start_.release();
        return packed;
    }
};
