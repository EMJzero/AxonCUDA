#include "chaining.cuh"
#include "constants.cuh"
#include "utils.cuh"

// HELPERS

__device__ __forceinline__
uint32_t size_bucket(uint32_t s) {
    return (s == 0) ? 0u : (31u - __clz(s));
}

__device__ __forceinline__
uint32_t float_to_ordered_uint(float value) {
    uint32_t bits = __float_as_uint(value);
    return bits ^ ((bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

__device__ __forceinline__
uint64_t pack_succ_claim(float score, uint32_t pred) {
    return (uint64_t(float_to_ordered_uint(score)) << 32) | uint64_t(UINT32_MAX - pred);
}

__device__ __forceinline__
uint32_t unpack_succ_claim_pred(uint64_t claim) {
    return UINT32_MAX - uint32_t(claim & 0xFFFFFFFFull);
}


// CHAINING

__global__
void build_src_keys(
    const uint32_t num_edges,
    const uint32_t* src,
    const float* w,
    src_negweight_key* keys
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges) {
        keys[i].src = src[i];
        keys[i].negw = -w[i];
    }
}

// multi-iteration greedy chaining
//
// data structures:
//   next[i] : chosen successor edge index (or UINT32_MAX)
//   prev[j] : chosen predecessor edge index (or UINT32_MAX)
//
// each iteration proposes a successor for edges that don't yet have next, searching up to "window" candidates
// candidates are from outgoing list of dst[i] (i.e. edges whose src == dst[i]), preferring high weight and similar node size / inbound set size
// conflicts are resolved so each successor has at most one predecessor
__global__
void propose_successor(
    const uint32_t num_edges,
    const uint32_t* dst,
    const uint32_t* size,
    //const uint32_t* icnt,
    const float* w,
    const int* out_begin,
    const int* out_end,
    const uint32_t* prev, // availability of candidate successor (prev[cand] == UINT32_MAX)
    const uint32_t* next, // only propose if next[i] == UINT32_MAX
    int window,
    float alpha,
    uint32_t* succ_choice,
    float* succ_score
) {
    // STYLE: one edge per thread!
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;

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

    int limit = b + window;
    if (limit > e) limit = e;

    for (int j = b; j < limit; ++j) {
        if (best_j != UINT32_MAX && w[j] <= best) break;
        if ((uint32_t)j == (uint32_t)i) continue;
        if (prev[j] != UINT32_MAX) continue; // candidate successor already taken

        uint32_t sj = size[j];
        uint32_t bj = size_bucket(sj);
        if (abs((int)bi - (int)bj) > 1) continue;

        float penalty = alpha * fabsf((float)si - (float)sj);
        float sc = w[j] - penalty;

        if (sc > best || (sc == best && (uint32_t)j < best_j)) {
            best = sc;
            best_j = (uint32_t)j;
        }
    }

    succ_choice[i] = best_j;
    succ_score[i] = best;
}

__global__
void resolve_successor_conflicts(
    const uint32_t num_edges,
    const uint32_t* succ_choice,
    const float* succ_score,
    const uint32_t* prev, // only allow assignment if prev[succ] == UINT32_MAX at time of resolve
    uint64_t* best_claim_for_succ
) {
    // STYLE: one edge per thread!
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;

    uint32_t s = succ_choice[i];
    if (s == UINT32_MAX) return;
    if (prev[s] != UINT32_MAX) return;

    atomicMax(reinterpret_cast<unsigned long long*>(&best_claim_for_succ[s]), (unsigned long long)pack_succ_claim(succ_score[i], (uint32_t)i));
}

__global__
void commit_links(
    const uint32_t num_edges,
    uint32_t* next,
    uint32_t* prev,
    const uint64_t* best_claim_for_succ
) {
    // STYLE: one edge per thread!
    // => for every successor edge s: if it has a chosen predecessor p and p is free, claim it
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_edges) return;

    uint64_t claim = best_claim_for_succ[s];
    if (claim == 0ull) return;
    uint32_t p = unpack_succ_claim_pred(claim);

    // only commit if successor still free
    if (atomicCAS((unsigned int*)&prev[s], (unsigned int)UINT32_MAX, (unsigned int)p) == (unsigned int)UINT32_MAX) {
        // set next[p] if still unset; if p somehow got set concurrently, keep who got it first => should never happen
        atomicCAS((unsigned int*)&next[p], (unsigned int)UINT32_MAX, (unsigned int)s);
    }
}

// extract component and position for each chain:
// - for paths, we want pos ~ distance from head (prev == UINT32_MAX)
// - for cycles, we just pick a representative (min index found) and pos is best-effort
__global__
void compute_comp_and_pos(
    const uint32_t num_edges,
    const uint32_t* prev,
    uint32_t* comp,
    uint32_t* pos
) {
    // STYLE: one edge per thread!
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    
    uint32_t cur = (uint32_t)i;
    uint32_t min_seen = cur;
    
    // WARNING => O(n * CHAIN_MAX_STEPS) total work
    for (int d = 0; d < CHAIN_MAX_STEPS; ++d) {
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

// final sequence ordering (collision-free)
// compute component total weight and count, rank components by weight, then sort edges by (component_rank, pos, edge_id) and assign sequence_idx
// => guarantees that sequence_idx is a permutation of [0..n-1]
__global__
void build_edge_order_keys(
    const uint32_t num_edges,
    const uint32_t* comp,
    const uint32_t* pos,
    const uint32_t* comp_to_rank,
    edge_order_key* keys
) {
    // STYLE: one edge per thread!
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;

    uint32_t c = comp[i];
    keys[i].comp_rank = comp_to_rank[c];
    keys[i].pos = pos[i];
    keys[i].edge_id = (uint32_t)i;
}

__global__
void scatter_sequence_idx(
    const uint32_t num_edges,
    const uint32_t* sorted_edge_id, // edge indices in final order (sorted space)
    const uint32_t* orig_index_sorted, // mapping from sorted space -> original edge index
    uint32_t* sequence_idx_out
) {
    // STYLE: one edge per thread!
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;

    uint32_t e_sorted = sorted_edge_id[i]; // edge id in sorted space
    uint32_t e_orig = orig_index_sorted[e_sorted];
    sequence_idx_out[e_orig] = (uint32_t)i;
}


// PAIR ORPHANS

// try to pair k-th smallest with k-th largest in parallel
__global__
void pair_kth_smallest_with_kth_largest(
    const uint32_t* __restrict__ sorted_indices, // length = K
    const uint32_t num_free,
    const uint32_t* __restrict__ d_nodes_sizes,
    const uint32_t* __restrict__ d_inbound_count,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    uint32_t* __restrict__ d_groups
) {
    // STYLE: one orphan-in-two per thread!
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = num_free / 2;
    if (k >= half) return;

    uint32_t idxL = sorted_indices[k];
    uint32_t idxR = sorted_indices[num_free - 1 - k];

    // read sizes and inbound counts
    uint32_t sL = d_nodes_sizes[idxL];
    uint32_t sR = d_nodes_sizes[idxR];

    // check sizes constraint
    if (sL + sR > h_max_nodes_per_part) return;

    uint32_t inL = d_inbound_count[idxL];
    uint32_t inR = d_inbound_count[idxR];

    if (inL + inR > h_max_inbound_per_part) return;

    // both constraints satisfied -> write group id
    uint32_t gid = (idxL < idxR) ? idxL : idxR;
    d_groups[idxL] = gid;
    d_groups[idxR] = gid;
}
