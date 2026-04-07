#pragma once
#include <cuda_runtime.h>

#include "defines.cuh"
#include "data_types.cuh"

namespace config {
    struct runconfig;
}

using namespace config;


// USED BY: chaining

#define CHAIN_ITERS 4 // multi-iteration greedy chaining
#define CHAIN_WINDOW 256 // candidates scanned per node per iteration
#define CHAIN_ALPHA 1e-6f // node size penalty scale (adjust based on size magnitude)
//#define BETA 1e-7f // inbound set size penalty scale (adjust based on inbound set size magnitude)

#define CHAIN_MAX_STEPS 256 // maximum nodes explored to form a chain, increase if the typical chain length could exceeds it

// sorting key for (src, -weight) to group outgoing lists by src, heavier edges going first inside each src group
struct src_negweight_key {
    uint32_t src;
    float negw;
    __host__ __device__
    bool operator<(const src_negweight_key& other) const {
        if (src != other.src) return src < other.src;
        return negw < other.negw;
    }
};

// sorting key for edges in sequences
struct edge_order_key {
    uint32_t comp_rank;
    uint32_t pos;
    uint32_t edge_id;
    __host__ __device__
    bool operator<(const edge_order_key& other) const {
        if (comp_rank != other.comp_rank) return comp_rank < other.comp_rank;
        if (pos != other.pos) return pos < other.pos;
        return edge_id < other.edge_id;
    }
};


// STEPS

void chaining(
    const runconfig &cfg,
    const uint32_t *srcs,
    const uint32_t *dsts,
    const uint32_t *size,
    const float *weight,
    const uint32_t num_edges,
    uint32_t *sequence_idx
);

void build_orphan_pairs(
    const runconfig &cfg,
    const uint32_t *d_nodes_sizes,
    const uint32_t *d_inbound_count,
    const uint32_t *d_pairs,
    const uint32_t curr_num_nodes,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    const uint32_t candidates_count,
    uint32_t* d_groups
);


// KERNELS

__global__
void build_src_keys(
    const uint32_t num_edges,
    const uint32_t* src,
    const float* w,
    src_negweight_key* keys
);

__global__
void propose_successor(
    const uint32_t num_edges,
    const uint32_t* dst,
    const uint32_t* size,
    const float* w,
    const int* out_begin,
    const int* out_end,
    const uint32_t* prev,
    const uint32_t* next,
    int window,
    float alpha,
    uint32_t* succ_choice,
    float* succ_score
);

__global__
void resolve_successor_conflicts(
    const uint32_t num_edges,
    const uint32_t* succ_choice,
    const float* succ_score,
    const uint32_t* prev,
    uint64_t* best_claim_for_succ
);

__global__
void commit_links(
    const uint32_t num_edges,
    uint32_t* next,
    uint32_t* prev,
    const uint64_t* best_claim_for_succ
);

__global__
void compute_comp_and_pos(
    const uint32_t num_edges,
    const uint32_t* prev,
    uint32_t* comp,
    uint32_t* pos
);

__global__
void build_edge_order_keys(
    const uint32_t num_edges,
    const uint32_t* comp,
    const uint32_t* pos,
    const uint32_t* comp_to_rank,
    edge_order_key* keys
);

__global__
void scatter_sequence_idx(
    const uint32_t num_edges,
    const uint32_t* sorted_edge_id,
    const uint32_t* orig_index_sorted,
    uint32_t* sequence_idx_out
);

__global__
void pair_kth_smallest_with_kth_largest(
    const uint32_t* __restrict__ sorted_indices,
    const uint32_t num_free,
    const uint32_t* __restrict__ d_nodes_sizes,
    const uint32_t* __restrict__ d_inbound_count,
    const uint32_t h_max_nodes_per_part,
    const uint32_t h_max_inbound_per_part,
    uint32_t* __restrict__ d_groups
);
