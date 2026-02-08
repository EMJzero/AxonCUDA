#include <stdint.h>

#include </home/mronzani/cuda/include/cuda_runtime.h>

#include </home/mronzani/cuda/include/thrust/device_vector.h>
#include </home/mronzani/cuda/include/thrust/host_vector.h>
#include </home/mronzani/cuda/include/thrust/sort.h>
#include </home/mronzani/cuda/include/thrust/sequence.h>
#include </home/mronzani/cuda/include/thrust/iterator/zip_iterator.h>
#include </home/mronzani/cuda/include/thrust/tuple.h>
#include </home/mronzani/cuda/include/thrust/gather.h>
#include </home/mronzani/cuda/include/thrust/scatter.h>
#include </home/mronzani/cuda/include/thrust/reduce.h>
#include </home/mronzani/cuda/include/thrust/transform.h>
#include </home/mronzani/cuda/include/thrust/binary_search.h>
#include </home/mronzani/cuda/include/thrust/copy.h>

#include "utils.cuh"


// ================================================================
// Device helpers
// ================================================================

__device__ __forceinline__
uint32_t size_bucket(uint32_t s) {
    return (s == 0) ? 0u : (31u - __clz(s));
}

__device__ inline
float atomicMaxFloat(float* addr, float value) {
    int* addr_i = reinterpret_cast<int*>(addr);
    int old = *addr_i;
    int assumed;

    while (__int_as_float(old) < value) {
        assumed = old;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
        if (assumed == old) {
            break;
        }
    }
    return __int_as_float(old);
}

// ================================================================
// Sorting key for (src, -weight) to get outgoing lists by src,
// with heavier edges first inside each src group.
// ================================================================

struct SrcNegWKey {
    uint32_t src;
    float negw;
};

struct SrcNegWLess {
    __host__ __device__
    bool operator()(const SrcNegWKey& a, const SrcNegWKey& b) const {
        if (a.src != b.src) return a.src < b.src;
        return a.negw < b.negw;
    }
};

__global__
void build_src_keys(
    int n,
    const uint32_t* src,
    const float* w,
    SrcNegWKey* keys
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keys[i].src = src[i];
        keys[i].negw = -w[i];
    }
}

// ================================================================
// Multi-iteration greedy chaining
//
// We maintain:
//   next[i] : chosen successor edge index (or UINT32_MAX)
//   prev[j] : chosen predecessor edge index (or UINT32_MAX)
//
// Each iteration proposes a successor for edges that don't yet have next.
// Candidates are from outgoing list of dst[i] (i.e. edges whose src == dst[i]),
// preferring high weight and similar size.
// Conflicts are resolved so each successor has at most one predecessor.
// ================================================================

__global__
void init_array_u32(int n, uint32_t* a, uint32_t v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

__global__
void init_array_f(int n, float* a, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

__global__
void propose_successor(
    int n,
    const uint32_t* dst,
    const uint32_t* size,
    const float* w,
    const int* out_begin,
    const int* out_end,
    const uint32_t* prev, // availability of candidate successor (prev[cand] == UINT32_MAX)
    const uint32_t* next, // only propose if next[i] == UINT32_MAX
    int K,
    float alpha,
    uint32_t* succ_choice,
    float* succ_score
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (next[i] != UINT32_MAX) {
        succ_choice[i] = UINT32_MAX;
        succ_score[i] = -1e30f;
        return;
    }

    int b = out_begin[i];
    int e = out_end[i];
    if (b >= e) {
        succ_choice[i] = UINT32_MAX;
        succ_score[i] = -1e30f;
        return;
    }

    uint32_t si = size[i];
    uint32_t bi = size_bucket(si);

    float best = -1e30f;
    uint32_t best_j = UINT32_MAX;

    int limit = b + K;
    if (limit > e) limit = e;

    for (int j = b; j < limit; ++j) {
        if ((uint32_t)j == (uint32_t)i) continue;
        if (prev[j] != UINT32_MAX) continue; // candidate successor already taken

        uint32_t sj = size[j];
        uint32_t bj = size_bucket(sj);
        if (abs((int)bi - (int)bj) > 1) continue;

        float penalty = alpha * fabsf((float)si - (float)sj);
        float sc = w[j] - penalty;

        if (sc > best) {
            best = sc;
            best_j = (uint32_t)j;
        }
    }

    succ_choice[i] = best_j;
    succ_score[i] = best;
}

__global__
void resolve_successor_conflicts(
    int n,
    const uint32_t* succ_choice,
    const float* succ_score,
    const uint32_t* prev, // only allow assignment if prev[succ] == UINT32_MAX at time of resolve
    float* best_score_for_succ,
    uint32_t* best_pred_for_succ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t s = succ_choice[i];
    if (s == UINT32_MAX) return;
    if (prev[s] != UINT32_MAX) return;

    float sc = succ_score[i];
    float old = atomicMaxFloat(&best_score_for_succ[s], sc);
    if (sc > old) {
        best_pred_for_succ[s] = (uint32_t)i;
    }
}

__global__
void commit_links(
    int n,
    uint32_t* next,
    uint32_t* prev,
    const uint32_t* best_pred_for_succ
) {
    // one thread per successor edge s: if it has a chosen predecessor p and is free, claim it
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;

    uint32_t p = best_pred_for_succ[s];
    if (p == UINT32_MAX) return;

    // only commit if successor still free
    if (atomicCAS((unsigned int*)&prev[s], (unsigned int)UINT32_MAX, (unsigned int)p) == (unsigned int)UINT32_MAX) {
        // set next[p] if still unset. If p somehow got set concurrently, keep first
        atomicCAS((unsigned int*)&next[p], (unsigned int)UINT32_MAX, (unsigned int)s);
    }
}

// ================================================================
// Component and position extraction
//
// For paths, we want pos ~ distance from head (prev==UINT32_MAX).
// For cycles, we just pick a representative (min index found) and pos is best-effort.
// ================================================================

__global__
void compute_comp_and_pos(
    int n,
    const uint32_t* prev,
    uint32_t* comp,
    uint32_t* pos
) {
    // TODO: tune, increase if the typical chain length could exceeds it
    // => O(n * MAX_STEPS) total work
    constexpr int MAX_STEPS = 256;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t cur = (uint32_t)i;
    uint32_t min_seen = cur;

    for (int d = 0; d < MAX_STEPS; ++d) {
        uint32_t p = prev[cur];
        if (p == UINT32_MAX) {
            comp[i] = cur; // head edge id
            pos[i] = (uint32_t)d; // distance from head
            return;
        }
        cur = p;
        min_seen = min(min_seen, cur);
    }

    // likely cycle or very long chain; choose rep as min seen
    comp[i] = min_seen;
    pos[i] = 0;
}

// ================================================================
// Final sequence ordering (collision-free)
//
// We compute component total weight and count, rank components by weight,
// then sort edges by (component_rank, pos, edge_id) and assign sequence_idx.
// This guarantees sequence_idx is a permutation [0..n-1].
// ================================================================

struct EdgeOrderKey {
    uint32_t comp_rank;
    uint32_t pos;
    uint32_t edge_id;
};

struct EdgeOrderLess {
    __host__ __device__
    bool operator()(const EdgeOrderKey& a, const EdgeOrderKey& b) const {
        if (a.comp_rank != b.comp_rank) return a.comp_rank < b.comp_rank;
        if (a.pos != b.pos) return a.pos < b.pos;
        return a.edge_id < b.edge_id;
    }
};

__global__
void build_edge_order_keys(
    int n,
    const uint32_t* comp,
    const uint32_t* pos,
    const uint32_t* comp_to_rank,
    EdgeOrderKey* keys
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t c = comp[i];
        keys[i].comp_rank = comp_to_rank[c];
        keys[i].pos = pos[i];
        keys[i].edge_id = (uint32_t)i;
    }
}

__global__
void scatter_sequence_idx(
    int n,
    const uint32_t* sorted_edge_id, // edge indices in final order (sorted space)
    const uint32_t* orig_index_sorted, // mapping from sorted space -> original edge index
    uint32_t* sequence_idx_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t e_sorted = sorted_edge_id[i]; // edge id in sorted space
        uint32_t e_orig = orig_index_sorted[e_sorted];
        sequence_idx_out[e_orig] = (uint32_t)i;
    }
}

// ================================================================
// Entry Point
// ================================================================

// !!!!!!!!!!!!!!!!!!!!!!!!!!
// WARNING: NOT DETERMINISTIC
// !!!!!!!!!!!!!!!!!!!!!!!!!!

void chaining(
    const uint32_t* srcs,
    const uint32_t* dsts,
    const uint32_t* size,
    const float* weight,
    uint32_t num,
    uint32_t* sequence_idx,
    cudaStream_t stream = 0
) {
    if (num == 0) return;

    const int n = (int)num;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // parameters:
    const int iters = 4; // multi-iteration greedy chaining
    const int K = 256; // candidates scanned per node per iteration
    const float alpha = 1e-6f; // size penalty scale (adjust based on your size magnitude)

    auto exec = thrust::cuda::par.on(stream);

    // copy input device arrays into vectors so we can sort/reorder
    thrust::device_vector<uint32_t> d_src(n), d_dst(n), d_size(n), d_orig(n);
    thrust::device_vector<float> d_w(n);

    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_src.data()), srcs, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_dst.data()), dsts, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_size.data()), size, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_w.data()), weight, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    thrust::sequence(exec, d_orig.begin(), d_orig.end());

    // sort edges by (src, -weight) so OUT[v] is a contiguous range and heavier edges come first
    thrust::device_vector<SrcNegWKey> d_keys(n);
    build_src_keys<<<blocks, threads, 0, stream>>>(
        n,
        thrust::raw_pointer_cast(d_src.data()),
        thrust::raw_pointer_cast(d_w.data()),
        thrust::raw_pointer_cast(d_keys.data())
    );

    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(d_src.begin(), d_dst.begin(), d_size.begin(), d_w.begin(), d_orig.begin()));

    thrust::sort_by_key(exec, d_keys.begin(), d_keys.end(), zipped, SrcNegWLess());

    // for each edge i, its outgoing candidate list is OUT[dst[i]] = edges whose src == dst[i]
    thrust::device_vector<int> d_out_begin(n), d_out_end(n);
    thrust::lower_bound(exec, d_src.begin(), d_src.end(), d_dst.begin(), d_dst.end(), d_out_begin.begin());
    thrust::upper_bound(exec, d_src.begin(), d_src.end(), d_dst.begin(), d_dst.end(), d_out_end.begin());

    // chaining state
    thrust::device_vector<uint32_t> d_next(n), d_prev(n);
    init_array_u32<<<blocks, threads, 0, stream>>>(n, thrust::raw_pointer_cast(d_next.data()), UINT32_MAX);
    init_array_u32<<<blocks, threads, 0, stream>>>(n, thrust::raw_pointer_cast(d_prev.data()), UINT32_MAX);

    // temporary proposal / conflict buffers
    thrust::device_vector<uint32_t> d_succ_choice(n);
    thrust::device_vector<float> d_succ_score(n);

    thrust::device_vector<float> d_best_score(n);
    thrust::device_vector<uint32_t> d_best_pred(n);

    // multi-iteration greedy build
    for (int it = 0; it < iters; ++it) {
        init_array_f<<<blocks, threads, 0, stream>>>(n, thrust::raw_pointer_cast(d_best_score.data()), -1e30f);
        init_array_u32<<<blocks, threads, 0, stream>>>(n, thrust::raw_pointer_cast(d_best_pred.data()), UINT32_MAX);

        propose_successor<<<blocks, threads, 0, stream>>>(
            n,
            thrust::raw_pointer_cast(d_dst.data()),
            thrust::raw_pointer_cast(d_size.data()),
            thrust::raw_pointer_cast(d_w.data()),
            thrust::raw_pointer_cast(d_out_begin.data()),
            thrust::raw_pointer_cast(d_out_end.data()),
            thrust::raw_pointer_cast(d_prev.data()),
            thrust::raw_pointer_cast(d_next.data()),
            K, alpha,
            thrust::raw_pointer_cast(d_succ_choice.data()),
            thrust::raw_pointer_cast(d_succ_score.data())
        );

        resolve_successor_conflicts<<<blocks, threads, 0, stream>>>(
            n,
            thrust::raw_pointer_cast(d_succ_choice.data()),
            thrust::raw_pointer_cast(d_succ_score.data()),
            thrust::raw_pointer_cast(d_prev.data()),
            thrust::raw_pointer_cast(d_best_score.data()),
            thrust::raw_pointer_cast(d_best_pred.data())
        );

        commit_links<<<blocks, threads, 0, stream>>>(
            n,
            thrust::raw_pointer_cast(d_next.data()),
            thrust::raw_pointer_cast(d_prev.data()),
            thrust::raw_pointer_cast(d_best_pred.data())
        );
    }

    // component id and position
    thrust::device_vector<uint32_t> d_comp(n), d_pos(n);
    compute_comp_and_pos<<<blocks, threads, 0, stream>>>(
        n,
        thrust::raw_pointer_cast(d_prev.data()),
        thrust::raw_pointer_cast(d_comp.data()),
        thrust::raw_pointer_cast(d_pos.data())
    );

    // compute component weight sums and counts via reduce_by_key, sort edges by comp to reduce
    thrust::device_vector<uint32_t> d_comp_key = d_comp;
    thrust::device_vector<float> d_w_val = d_w;
    thrust::device_vector<uint32_t> d_one(n, 1);

    thrust::sort_by_key(exec, d_comp_key.begin(), d_comp_key.end(), thrust::make_zip_iterator(thrust::make_tuple(d_w_val.begin(), d_one.begin())));

    thrust::device_vector<uint32_t> d_unique_comp(n);
    thrust::device_vector<float> d_comp_wsum(n);
    thrust::device_vector<uint32_t> d_comp_count(n);

    auto end1 = thrust::reduce_by_key(
        exec,
        d_comp_key.begin(), d_comp_key.end(),
        d_w_val.begin(),
        d_unique_comp.begin(),
        d_comp_wsum.begin()
    );
    int m = (int)(end1.first - d_unique_comp.begin());

    auto end2 = thrust::reduce_by_key(
        exec,
        d_comp_key.begin(), d_comp_key.end(),
        d_one.begin(),
        d_unique_comp.begin(), // overwrite ok (same keys)
        d_comp_count.begin()
    );
    int m2 = (int)(end2.first - d_unique_comp.begin());
    if (m2 < m) m = m2;

    // rank components by total weight descending
    thrust::device_vector<uint32_t> d_comp_idx(m);
    thrust::sequence(exec, d_comp_idx.begin(), d_comp_idx.end());

    thrust::sort(exec, d_comp_idx.begin(), d_comp_idx.end(),
        [wsum = thrust::raw_pointer_cast(d_comp_wsum.data())] __device__ (uint32_t a, uint32_t b) {
            float wa = wsum[a];
            float wb = wsum[b];
            if (wa > wb) return true;
            if (wa < wb) return false;
            return a < b;
        }
    );

    // map comp_id (which is an edge index in [0..n)) -> comp_rank in [0..m)
    thrust::device_vector<uint32_t> d_comp_to_rank(n, 0);
    thrust::device_vector<uint32_t> d_comp_id_rank(m);
    thrust::gather(exec, d_comp_idx.begin(), d_comp_idx.end(), d_unique_comp.begin(), d_comp_id_rank.begin());

    // fill comp_to_rank
    thrust::for_each_n(
        exec,
        thrust::make_counting_iterator(0),
        m,
        [comp_id_rank = thrust::raw_pointer_cast(d_comp_id_rank.data()), comp_to_rank = thrust::raw_pointer_cast(d_comp_to_rank.data())] __device__ (int r) {
            uint32_t c = comp_id_rank[r];
            comp_to_rank[c] = (uint32_t)r;
        }
    );

    // build per-edge sort keys (comp_rank, pos, edge_id)
    thrust::device_vector<EdgeOrderKey> d_edge_keys(n);
    build_edge_order_keys<<<blocks, threads, 0, stream>>>(
        n,
        thrust::raw_pointer_cast(d_comp.data()),
        thrust::raw_pointer_cast(d_pos.data()),
        thrust::raw_pointer_cast(d_comp_to_rank.data()),
        thrust::raw_pointer_cast(d_edge_keys.data())
    );

    // sort edges by final order keys; carry edge_id along
    thrust::device_vector<uint32_t> d_edge_id(n);
    thrust::sequence(exec, d_edge_id.begin(), d_edge_id.end());

    thrust::sort_by_key(exec, d_edge_keys.begin(), d_edge_keys.end(), d_edge_id.begin(), EdgeOrderLess());

    // assign final sequence_idx = global position w.r.t. that sorted order
    scatter_sequence_idx<<<blocks, threads, 0, stream>>>(
        n,
        thrust::raw_pointer_cast(d_edge_id.data()),
        thrust::raw_pointer_cast(d_orig.data()),
        sequence_idx
    );
}