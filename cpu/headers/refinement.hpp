#pragma once
#include <bit>
#include <cstdint>
#include <algorithm>

#include "defines.hpp"
#include "data_types.hpp"

namespace config {
    struct runconfig;
}

using namespace config;


// USED BY: fm refinement kernel

// NOTE: must match 'headers/refinement.cuh', it shapes the tie-breaking among equal-gain moves
#define PART_HIST_SIZE 64u // best if it is a multiple of WARP_SIZE, best if partitions_per_lane * WARP_SIZE <= num_partitions


// PINS PER PARTITION ACCESSORS
// => the refinement kernels are templated on these, each written once for both the dense and sparse pins per partition
// NOTE: in CUDA the dense and sparse pins per partition have twin kernels, differing only in how they access counters

// dense pins per partition: ppp[hedge idx * num_partitions + partition idx] -> number of pins of that partition owned by this hedge
struct dense_ppp {
    uint32_t* ppp;
    uint32_t num_partitions;

    // number of pins of the partition owned by the hedge
    uint32_t get(const uint32_t hedge_idx, const uint32_t part_idx) const {
        return ppp[static_cast<dim_t>(hedge_idx) * num_partitions + part_idx];
    }

    // counter of the number of pins of the partition owned by the hedge
    uint32_t* at(const uint32_t hedge_idx, const uint32_t part_idx) const {
        return ppp + static_cast<dim_t>(hedge_idx) * num_partitions + part_idx;
    }

    // loss[part] += weight for every partition that holds no pin of the hedge
    void add_where_empty(const uint32_t hedge_idx, const float weight, float* __restrict__ loss) const {
        const uint32_t* __restrict__ row = ppp + static_cast<dim_t>(hedge_idx) * num_partitions;
        #pragma omp simd
        for (uint32_t part = 0; part < num_partitions; part++)
            loss[part] = row[part] == 0 ? loss[part] + weight : loss[part];
    }
};

// sparse pins per partition: one bitmap per (hedge, 64 partitions) flags the partitions that hold pins of the hedge, and
// tells where their counters start in the segmented array
// => ppp[ppp_offsets[e * ppp_per_hedge + p / 64].cnt + bits-at-one-before-the(p % 64)th-in(ppp_offsets[e * ppp_per_hedge + p / 64].flg)] -> number of pins of partition p owned by hedge e
struct sparse_ppp {
    const bitmap* ppp_offsets;
    uint32_t* ppp;
    uint32_t ppp_per_hedge;
    uint32_t num_partitions;

    // number of pins of the partition owned by the hedge
    // - if your flag bit (the one part-idx from the least-significant one) is zero, return zero
    // - otherwise, count how many ones are in less-significant positions than part-idx, add that to the base count, and that's your offset to go fetch
    uint32_t get(const uint32_t hedge_idx, const uint32_t part_idx) const {
        const bitmap ppp_bitmap = ppp_offsets[static_cast<dim_t>(hedge_idx) * ppp_per_hedge + (part_idx >> BITMAP_CAPLOG)];
        const uint64_t bitmap_part_idx = part_idx & (BITMAP_CAPACITY - 1u);
        if (((ppp_bitmap.flg >> bitmap_part_idx) & 1ull) == 0ull) return 0u; // no pins
        return ppp[ppp_bitmap.cnt + std::popcount(ppp_bitmap.flg & ((1ull << bitmap_part_idx) - 1ull))];
    }

    // counter of the number of pins of the partition owned by the hedge
    // HP: the partition holds at least a pin of the hedge (its flag bit is one)
    uint32_t* at(const uint32_t hedge_idx, const uint32_t part_idx) const {
        const bitmap ppp_bitmap = ppp_offsets[static_cast<dim_t>(hedge_idx) * ppp_per_hedge + (part_idx >> BITMAP_CAPLOG)];
        const uint64_t bitmap_part_idx = part_idx & (BITMAP_CAPACITY - 1u);
        return ppp + ppp_bitmap.cnt + std::popcount(ppp_bitmap.flg & ((1ull << bitmap_part_idx) - 1ull));
    }

    // loss[part] += weight for every partition that holds no pin of the hedge
    // HP: all pins are still counted (not just inbound ones), hence the flag bit alone tells whether a partition holds no pin
    void add_where_empty(const uint32_t hedge_idx, const float weight, float* __restrict__ loss) const {
        const bitmap* my_ppp_offsets = ppp_offsets + static_cast<dim_t>(hedge_idx) * ppp_per_hedge;
        for (uint32_t b = 0; b < ppp_per_hedge; b++) {
            const uint64_t flg = my_ppp_offsets[b].flg;
            const uint32_t base_part = b * BITMAP_CAPACITY;
            const uint32_t parts = std::min(BITMAP_CAPACITY, num_partitions - base_part);
            #pragma omp simd
            for (uint32_t bit = 0; bit < parts; bit++)
                loss[base_part + bit] = ((flg >> bit) & 1ull) ? loss[base_part + bit] : loss[base_part + bit] + weight;
        }
    }
};


// STEPS

void refinementRepeats(
    const runconfig &cfg,
    const uint32_t *hedges,
    const dim_t *hedges_offsets,
    const uint32_t *srcs_count,
    const uint32_t *touching,
    const dim_t *touching_offsets,
    const uint32_t *inbound_count,
    const float *hedge_weights,
    const uint32_t *nodes_sizes,
    const uint32_t *nodes_pins,
    const uint32_t level_idx,
    const uint32_t curr_num_nodes,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    const dim_t touching_size,
    const bool update_final_inbound_counts,
    uint32_t *pairs,
    float *f_scores,
    uint32_t *partitions,
    uint32_t *partitions_sizes,
    uint32_t *partitions_inbound_sizes,
    uint32_t *partitions_pins
);

void logPartitions(
    const uint32_t *partitions,
    const uint32_t *partitions_sizes,
    const uint32_t *partitions_inbound_sizes,
    const uint32_t *partitions_pins,
    const uint32_t curr_num_nodes,
    const uint32_t num_partitions
);

void logMoves(
    const uint32_t *pairs,
    const float *f_scores,
    const uint32_t *partitions,
    const uint32_t curr_num_nodes
);


// KERNELS

void pins_per_partition_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ pins_per_partitions,
    uint32_t* __restrict__ partitions_inbound_sizes
);

void sparse_pins_per_partition_count_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_hedges,
    const uint32_t ppp_per_hedge,
    bitmap* __restrict__ ppp_offsets
);

void sparse_pins_per_partition_write_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ partitions,
    const bitmap* __restrict__ ppp_offsets,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    const uint32_t ppp_per_hedge,
    uint32_t* __restrict__ ppp,
    uint32_t* __restrict__ partitions_incident_sizes
);

template <typename PPP>
void inbound_pins_per_partition_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    const PPP inbound_pins_per_partitions,
    uint32_t* __restrict__ partitions_inbound_sizes
);

template <typename PPP>
void fm_refinement_gains_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ partitions,
    const PPP pins_per_partitions,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ partitions_sizes,
    const uint32_t num_nodes,
    const uint32_t num_partitions,
    const uint32_t randomizer,
    const uint32_t discount,
    const bool encourage_all_moves,
    uint32_t* __restrict__ moves,
    float* __restrict__ scores
);

template <typename PPP>
void fm_refinement_cascade_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ move_ranks,
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ partitions,
    const PPP pins_per_partitions,
    const uint32_t num_nodes,
    const bool encourage_all_moves,
    float* __restrict__ scores
);

void fm_refinement_apply_kernel(
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ move_ranks,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ nodes_pins,
    const uint32_t num_nodes,
    const uint32_t num_good_moves,
    uint32_t* __restrict__ partitions,
    uint32_t* __restrict__ partitions_sizes,
    uint32_t* __restrict__ partitions_pins
);

void build_size_events_kernel(
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ node_of_rank,
    const uint32_t* __restrict__ moving_ranks,
    const uint32_t* __restrict__ partitions,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ nodes_pins,
    const uint32_t num_moving,
    uint32_t* __restrict__ ev_partition,
    uint32_t* __restrict__ ev_index,
    int32_t* __restrict__ ev_delta,
    int32_t* __restrict__ ev_pins_delta
);

void flag_size_events_kernel(
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const int32_t* __restrict__ ev_delta,
    const uint32_t* __restrict__ partitions_sizes,
    const dim_t num_events,
    const uint32_t max_per_part,
    int32_t* __restrict__ valid_moves
);

void build_hedge_events_kernel(
    const uint32_t* __restrict__ moves,
    const uint32_t* __restrict__ node_of_rank,
    const uint32_t* __restrict__ moving_ranks,
    const uint32_t* __restrict__ partitions,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const dim_t* __restrict__ ev_offsets,
    const uint32_t num_moving,
    uint32_t* __restrict__ ev_partition,
    uint32_t* __restrict__ ev_index,
    uint32_t* __restrict__ ev_hedge,
    int32_t* __restrict__ ev_delta
);

template <typename PPP>
void count_inbound_size_events_kernel(
    const PPP partitions_inbound_counts,
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_hedge,
    const int32_t* __restrict__ ev_delta,
    const dim_t num_events,
    dim_t* __restrict__ inbound_size_events_offsets
);

template <typename PPP>
void build_inbound_size_events_kernel(
    const PPP partitions_inbound_counts,
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const uint32_t* __restrict__ ev_hedge,
    const int32_t* __restrict__ ev_delta,
    const dim_t* __restrict__ inbound_size_events_offsets,
    const dim_t num_events,
    uint32_t* __restrict__ new_ev_partition,
    uint32_t* __restrict__ new_ev_index,
    int32_t* __restrict__ new_ev_delta
);

void flag_inbound_events_kernel(
    const uint32_t* __restrict__ ev_partition,
    const uint32_t* __restrict__ ev_index,
    const int32_t* __restrict__ ev_delta,
    const uint32_t* __restrict__ partitions_inbound_sizes,
    const dim_t num_events,
    int32_t* __restrict__ valid_moves
);

void inbound_sets_size_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ partitions,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    uint32_t* __restrict__ partitions_inbound_sizes
);
