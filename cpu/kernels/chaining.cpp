#include <cmath>
#include <cstdlib>

#include "utils.hpp"
#include "chaining.hpp"
#include "constants.hpp"

// HELPERS

inline uint32_t size_bucket(uint32_t s) {
    return (s == 0) ? 0u : (31u - (uint32_t)__builtin_clz(s));
}

inline uint64_t pack_succ_claim(float score, uint32_t pred) {
    return (uint64_t(float_to_ordered_uint(score)) << 32) | uint64_t(UINT32_MAX - pred);
}

inline uint32_t unpack_succ_claim_pred(uint64_t claim) {
    return UINT32_MAX - uint32_t(claim & 0xFFFFFFFFull);
}


// CHAINING

// multi-iteration greedy chaining
//
// data structures:
//   next[i] : chosen successor edge index (or UINT32_MAX)
//   prev[j] : chosen predecessor edge index (or UINT32_MAX)
//
// each iteration proposes a successor for edges that don't yet have next, searching up to "window" candidates
// candidates are from outgoing list of dst[i] (i.e. edges whose src == dst[i]), preferring high weight and similar node size / inbound set size
// conflicts are resolved so each successor has at most one predecessor
void propose_successor(
    const uint32_t num_edges,
    const uint32_t* dst,
    const uint32_t* size,
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
    (void)dst;
    // STYLE: one edge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK) if(num_edges > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < num_edges; i++) {
        if (next[i] != UINT32_MAX) {
            succ_choice[i] = UINT32_MAX;
            succ_score[i] = -1e30f;
            continue;
        }

        int b = out_begin[i];
        int e = out_end[i];
        if (b >= e) {
            succ_choice[i] = UINT32_MAX;
            succ_score[i] = -1e30f;
            continue;
        }

        uint32_t si = size[i];
        uint32_t bi = size_bucket(si);

        float best = -1e30f;
        uint32_t best_j = UINT32_MAX;

        int limit = b + window;
        if (limit > e) limit = e;

        for (int j = b; j < limit; ++j) {
            if (best_j != UINT32_MAX && w[j] <= best) break;
            if ((uint32_t)j == i) continue;
            if (prev[j] != UINT32_MAX) continue; // candidate successor already taken

            uint32_t sj = size[j];
            uint32_t bj = size_bucket(sj);
            if (std::abs((int)bi - (int)bj) > 1) continue;

            // NOTE: in CUDA nvcc fuses "w[j] - alpha * |si - sj|" into a single FFMA, hence the explicit 'fmaf'
            float sc = std::fmaf(-std::fabs((float)si - (float)sj), alpha, w[j]);

            if (sc > best || (sc == best && (uint32_t)j < best_j)) {
                best = sc;
                best_j = (uint32_t)j;
            }
        }

        succ_choice[i] = best_j;
        succ_score[i] = best;
    }
}

void resolve_successor_conflicts(
    const uint32_t num_edges,
    const uint32_t* succ_choice,
    const float* succ_score,
    const uint32_t* prev, // only allow assignment if prev[succ] == UINT32_MAX at time of resolve
    uint64_t* best_claim_for_succ
) {
    // STYLE: one edge per iteration!
    #pragma omp parallel for schedule(static) if(num_edges > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < num_edges; i++) {
        uint32_t s = succ_choice[i];
        if (s == UINT32_MAX) continue;
        if (prev[s] != UINT32_MAX) continue;
        atomic_max<uint64_t>(&best_claim_for_succ[s], pack_succ_claim(succ_score[i], i));
    }
}

void commit_links(
    const uint32_t num_edges,
    uint32_t* next,
    uint32_t* prev,
    const uint64_t* best_claim_for_succ
) {
    // STYLE: one edge per iteration!
    // => for every successor edge s: if it has a chosen predecessor p and p is free, claim it
    #pragma omp parallel for schedule(static) if(num_edges > PARALLEL_GRAIN)
    for (uint32_t s = 0; s < num_edges; s++) {
        uint64_t claim = best_claim_for_succ[s];
        if (claim == 0ull) continue;
        uint32_t p = unpack_succ_claim_pred(claim);

        // only commit if successor still free
        if (atomic_cas<uint32_t>(&prev[s], UINT32_MAX, p)) {
            // set next[p] if still unset; if p somehow got set concurrently, keep who got it first => should never happen
            atomic_cas<uint32_t>(&next[p], UINT32_MAX, s);
        }
    }
}

// extract component and position for each chain:
// - for paths, we want pos ~ distance from head (prev == UINT32_MAX)
// - for cycles, we just pick a representative (min index found) and pos is best-effort
void compute_comp_and_pos(
    const uint32_t num_edges,
    const uint32_t* prev,
    uint32_t* comp,
    uint32_t* pos
) {
    // STYLE: one edge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK) if(num_edges > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < num_edges; i++) {
        uint32_t cur = i;
        uint32_t min_seen = cur;
        bool headed = false;

        // WARNING => O(n * CHAIN_MAX_STEPS) total work
        for (int d = 0; d < CHAIN_MAX_STEPS; ++d) {
            uint32_t p = prev[cur];
            if (p == UINT32_MAX) {
                comp[i] = cur; // head edge id
                pos[i] = (uint32_t)d; // distance from head
                headed = true;
                break;
            }
            cur = p;
            min_seen = min_seen < cur ? min_seen : cur;
        }

        // likely cycle or very long chain; choose rep as min seen
        if (!headed) {
            comp[i] = min_seen;
            pos[i] = 0;
        }
    }
}


// PAIR ORPHANS

// try to pair k-th smallest with k-th largest in parallel
void pair_kth_smallest_with_kth_largest(
    const uint32_t* __restrict__ sorted_indices, // length = K
    const uint32_t num_free,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ nodes_pins,
    const uint32_t* __restrict__ inbound_count,
    uint32_t* __restrict__ groups
) {
    const uint32_t half = num_free / 2;
    // STYLE: one orphan-in-two per iteration!
    #pragma omp parallel for schedule(static) if(half > PARALLEL_GRAIN)
    for (uint32_t k = 0; k < half; k++) {
        uint32_t idxL = sorted_indices[k];
        uint32_t idxR = sorted_indices[num_free - 1 - k];

        // check sizes constraint
        if (nodes_sizes[idxL] + nodes_sizes[idxR] > max_nodes_per_part) continue;

        // check inbound constraint
        if (inbound_count[idxL] + inbound_count[idxR] > max_inbound_per_part) continue;

        // check pins constraint
        if (nodes_pins[idxL] + nodes_pins[idxR] > max_pins_per_part) continue;

        // all constraints satisfied -> write group id
        uint32_t gid = (idxL < idxR) ? idxL : idxR;
        groups[idxL] = gid;
        groups[idxR] = gid;
    }
}
