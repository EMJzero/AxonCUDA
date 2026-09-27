#include <tuple>
#include <vector>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include "topology.hpp"
#include "runconfig_plc.hpp"

#include "utils.hpp"
#include "prims.hpp"
#include "utils_plc.hpp"
#include "defines_plc.hpp"
#include "placement.hpp"

template<Topology T>
void forceDirectedRefinement(
    const runconfig &cfg,
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const float* hedge_weights,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    Coord_t<T>* placement,
    uint32_t* inv_placement
) {
    const int tid = 0;

    // the whole batch of multi-starts is refined by one call of every kernel per step
    // => every per-multi-start array below is "batch_size" contiguous segments, multi-start "b" owning [b*size, (b+1)*size)
    const uint32_t batch_nodes = batch_size * num_nodes;

    // refinement structures
    buffer<float> forces((dim_t)T::neighborsCount() * batch_nodes); // forces[neighborsCount*node idx + neigh_idx] -> gain from moving the node towards its neigh_idx-th neighbor
    buffer<uint32_t> pairs((dim_t)cfg.candidates_count * batch_nodes); // pairs[candidates_count*node idx + 0..] -> nodes the current one wants to swap with, ordered by decreasing score
    buffer<uint32_t> scores((dim_t)cfg.candidates_count * batch_nodes); // scores[candidates_count*node idx + 0..] -> score with which node wants to pair with other nodes
    buffer<slot> swap_slots(batch_nodes); // slot to finalize node pairs while computing exclusive swaps
    // events structures
    // NOTE: events are not compacted, node idx "i" owns event slot "i" and empty slots carry -FLT_MAX, sorting behind every real event
    buffer<swap> ev_swaps(batch_nodes); // ev_swaps[event idx] -> pair of nodes involved in the event's swap
    buffer<float> ev_scores(batch_nodes); // ev_scores[event idx] -> score (cost gain) achieved by the event's swap
    buffer<uint32_t> nodes_rank(batch_nodes); // node_rank[node idx] -> rank (index) in the sorted events by score of the node, local to its multi-start
    // batch control structures
    buffer<uint32_t> num_good_swaps(batch_size); // num_good_swaps[start] -> length of that multi-start's improving event prefix
    buffer<uint8_t> active(batch_size); // active[start] -> 0 once that multi-start converged, its iterations are skipped from then on

    // every multi-start starts out active, and retires itself from 'prefix_gain_kernel' once it stops improving
    par_fill<uint8_t>(active.data(), batch_size, 1u);
    par_fill<uint32_t>(num_good_swaps.data(), batch_size, 0u);

    for (uint32_t iter = 0; iter < cfg.fd_iterations; iter++) {
        INFO(cfg) std::cout TID(tid) << "Force-directed refinement, iteration " << iter << "\n";

        /*
        * Flow:
        * 1) compute forces from each node to the 4 cardinal placements around it
        * 2) compute the tension between each node and the 4 places around it
        * 3) select the highest-tension pairs of nodes to swap/move
        *   - each node can be moved at most once -> upward and downward passes (same as in grouping for coarsening)
        * 4) find and apply the highest subsequence of improving moves
        *   - create one move-event per pair
        *   - rank events
        *   - update each event's gain assuming all higher-ranked ones already applied
        *   - scan all updated gains, find the highest point in the sequence, and apply all moves up to it
        */

        LAUNCH(cfg) TID(tid) RUN << "forces kernel (threads=" << cfg.threads << ") ...\n";
        forces_kernel<T>(
            hedges,
            hedges_offsets,
            touching,
            touching_offsets,
            hedge_weights,
            placement,
            num_nodes,
            batch_size,
            active.data(),
            forces.data()
        );

        // =============================
        // print some temporary results
        LOG(cfg) logForces<T>(
            forces.data(),
            batch_nodes
        );
        // =============================

        LAUNCH(cfg) TID(tid) RUN << "tensions kernel (threads=" << cfg.threads << ") ...\n";
        tensions_kernel<T>(
            placement,
            inv_placement,
            forces.data(),
            num_nodes,
            batch_size,
            volume,
            active.data(),
            cfg.candidates_count,
            pairs.data(),
            scores.data()
        );

        // =============================
        // print some temporary results
        LOG(cfg) logTensions<T>(
            cfg,
            pairs.data(),
            scores.data(),
            batch_nodes
        );
        // =============================

        // zero-out swap slots
        par_fill<slot>(swap_slots.data(), batch_nodes, pack_slot(0u, UINT32_MAX)); // upper 32 bits to 0x00, lower 32 to 0xFF

        LAUNCH(cfg) TID(tid) RUN << "exclusive swaps kernel (threads=" << cfg.threads << ") ...\n";
        exclusive_swaps_kernel<T>(
            pairs.data(),
            scores.data(),
            num_nodes,
            batch_size,
            active.data(),
            cfg.candidates_count,
            swap_slots.data()
        );

        // =============================
        // print some temporary results
        LOG(cfg) logSwapPairs<T>(
            swap_slots.data(),
            batch_nodes
        );
        // =============================

        LAUNCH(cfg) TID(tid) RUN << "events kernel (threads=" << cfg.threads << ") ...\n";
        swap_events_kernel<T>(
            swap_slots.data(),
            num_nodes,
            batch_size,
            active.data(),
            ev_swaps.data(),
            ev_scores.data()
        );

        // sort (descending) each multi-start's events by score while carrying swapped nodes along
        // NOTE: segmented, and not batch-wide, so that a multi-start's event order never depends on what it is batched with
        // NOTE: in CUDA this is a segmented radix sort, here a stable radix sort of (multi-start, descending score) packed in one key
        {
            buffer<uint32_t> perm = par_sort_permutation(batch_nodes, 32u + bits_for(batch_size), [&](dim_t i) {
                return ((uint64_t)(i / num_nodes) << 32) | (uint64_t)(~float_to_ordered_uint(ev_scores[i]));
            });
            par_permute(perm.data(), batch_nodes, ev_scores);
            par_permute(perm.data(), batch_nodes, ev_swaps);
        }
        par_fill<uint32_t>(nodes_rank.data(), batch_nodes, UINT32_MAX);

        // =============================
        // print some temporary results
        LOG(cfg) logEvents(
            ev_swaps.data(),
            ev_scores.data(),
            batch_nodes,
            "sorted - in isolation"
        );
        // =============================

        LAUNCH(cfg) TID(tid) RUN << "scatter ranks kernel (threads=" << cfg.threads << ") ...\n";
        scatter_ranks_kernel<T>(
            ev_swaps.data(),
            num_nodes,
            batch_size,
            active.data(),
            nodes_rank.data()
        );

        LAUNCH(cfg) TID(tid) RUN << "resolve empty conflicts kernel (threads=" << cfg.threads << ") ...\n";
        resolve_empty_conflicts_kernel<T>(
            placement,
            inv_placement,
            nodes_rank.data(),
            num_nodes,
            batch_size,
            volume,
            active.data(),
            ev_swaps.data()
        );

        LAUNCH(cfg) TID(tid) RUN << "cascade kernel (threads=" << cfg.threads << ") ...\n";
        cascade_kernel<T>(
            hedges,
            hedges_offsets,
            touching,
            touching_offsets,
            hedge_weights,
            placement,
            ev_swaps.data(),
            nodes_rank.data(),
            num_nodes,
            batch_size,
            active.data(),
            ev_scores.data()
        );

        // =============================
        // print some temporary results
        LOG(cfg) logEvents(
            ev_swaps.data(),
            ev_scores.data(),
            batch_nodes,
            "cascade - in sequence"
        );
        // =============================

        LAUNCH(cfg) TID(tid) RUN << "prefix gain kernel (threads=" << cfg.threads << ") ...\n";
        prefix_gain_kernel(
            ev_scores.data(),
            num_nodes,
            batch_size,
            num_good_swaps.data(),
            active.data()
        );

        // update placement and inv_placement
        LAUNCH(cfg) TID(tid) RUN << "apply swaps kernel (threads=" << cfg.threads << ") ...\n";
        apply_swaps_kernel<T>(
            ev_swaps.data(),
            num_good_swaps.data(),
            num_nodes,
            batch_size,
            volume,
            active.data(),
            placement,
            inv_placement
        );

        // multi-starts retire themselves, so only check every so often
        if ((iter + 1) % FD_ACTIVE_CHECK_PERIOD == 0) {
            uint32_t active_count = 0u;
            for (uint32_t start = 0; start < batch_size; start++) active_count += active[start];
            INFO(cfg) std::cout TID(tid) << "Multi-starts still improving after iteration " << iter << ": " << active_count << "/" << batch_size << "\n";
            if (active_count == 0u) {
                INFO(cfg) std::cout TID(tid) << "Stopping, the whole batch converged on iteration " << iter << "\n";
                break;
            }
        }
    }
}

// per multi-start, return the weighted average hedge max src-dst manhattan distance and weighted average hedge Steiner tree span (<2x upper bound)
template<Topology T>
void getLocalityMetrics(
    const runconfig &cfg,
    const Coord_t<T>* placement,
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const uint32_t* srcs_count,
    const float* hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    float* src_dst_distance, // src_dst_distance[start] -> weighted avg. hedge max src-dst manhattan distance of that multi-start
    float* steiner_span // steiner_span[start] -> weighted avg. hedge Steiner tree span of that multi-start
) {
    const int tid = 0;
    /*
    * IDEA:
    * - NOTE: this is a good-guess/temporary solution
    * - kernel that computes, for each hedge, the max src-dst manhattan distance (proxy for latency)
    * - kernel that computes, for each hedge, an upper estimate of the min Steiner tree span
    * - weight each estimate by the spiking frequency and reduce across hedges
    * - merge the two reduced values into a single quality metric based on the ratio between E_T, E_R, and L_T, L_R
    *
    * Feasible Steiner approximation:
    * - you cannot "ricochet" off of other lattice points
    * - for each node, add to the total distance the distance between it, and the closes among all other nodes in the same hedge
    *   => this the facto creates a minimum spanning tree over the hedge's pins, connecting each pin to the other one closes to it
    *     => the "minimum spanning tree" is defined over a higher-level fully connected graph where only lattice points occupied by
    *        the hedge's pins exist, and each edge is weighted by the manhattan distance between said points
    *     => hence, with the minimum spanning tree, you always pay the minimum path distance between node pairs, even if you reuse links
    *   => the minimum spanning tree will surely have span <= 2x the minimum Steiner tree
    * - minimum spanning tree complexity: iterate over hedges, over pins of each, and for each pin over pins again (to find the closest), e+d^2
    *   => little issue:
    *     - a pin already part of the minimum spanning tree must not be reconsidered by subsequent nodes...
    *     => Prim-style algoritm, expanding sequentially the connected spanning tree inside each hedge's graph
    */

    // the whole batch is graded by one call of every kernel per metric, then reduced back down to one figure per multi-start
    const uint32_t batch_hedges = batch_size * num_hedges;

    buffer<float> result(batch_hedges); // result[hedge idx] -> temporary result for the hedge

    // sum each multi-start's hedge results
    // NOTE: in CUDA this is a segmented reduce, whose float sums associate differently (see README)
    auto reduce_per_start = [&](float* out) {
        // STYLE: one multi-start per iteration!
        #pragma omp parallel for schedule(dynamic, 1)
        for (uint32_t start = 0; start < batch_size; start++) {
            float sum = 0.0f;
            for (uint32_t hedge_idx = start * num_hedges; hedge_idx < (start + 1) * num_hedges; hedge_idx++) sum += result[hedge_idx];
            out[start] = sum;
        }
    };

    // compute the max (or tot) src-dst manhattan distance per hedge
    //LAUNCH(cfg) TID(tid) RUN << "max src-dst distance kernel (threads=" << cfg.threads << ") ...\n";
    LAUNCH(cfg) TID(tid) RUN << "tot src-dst distance kernel (threads=" << cfg.threads << ") ...\n";
    //max_src_dst_distance_kernel<T>(
    tot_src_dst_distance_kernel<T>(
        placement,
        hedges,
        hedges_offsets,
        srcs_count,
        hedge_weights,
        num_hedges,
        num_nodes,
        batch_size,
        result.data() // <- alredy multiplied by hedge weight
    );
    reduce_per_start(src_dst_distance);

    // compute the Steiner tree span upper bound per hedge, given by the weighted spanning tree overs its pins' complete graph
    LAUNCH(cfg) TID(tid) RUN << "min spanning tree weight kernel (threads=" << cfg.threads << ") ...\n";
    min_spanning_tree_weight_kernel<T>(
        placement,
        hedges,
        hedges_offsets,
        hedge_weights,
        num_hedges,
        num_nodes,
        batch_size,
        result.data() // <- alredy multiply by hedge weight
    );
    reduce_per_start(steiner_span);
}


// LOGGING

template<Topology T>
void logForces(
    const float *forces,
    const uint32_t num_nodes
) {
    const int tid = 0;
    std::cout TID(tid) << "Forces:\n";
    for (uint32_t i = 0; i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH); ++i) {
        std::cout TID(tid) << "  node " << i << " -> " << forces[T::neighborsCount() * i];
        for (uint32_t neigh_idx = 1; neigh_idx < T::neighborsCount(); neigh_idx++)
            std::cout << ", " << forces[T::neighborsCount() * i + neigh_idx];
        std::cout << "\n";
    }
}

template<Topology T>
void logTensions(
    const runconfig &cfg,
    const uint32_t *pairs,
    const uint32_t *scores,
    const uint32_t num_nodes
) {
    const int tid = 0;
    std::cout TID(tid) << "Tensions:\n";
    for (uint32_t i = 0; i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH); ++i) {
        std::cout TID(tid) << "  node " << i << " ->";
        for (uint32_t j = 0; j < cfg.candidates_count; ++j) {
            uint32_t target = pairs[i * cfg.candidates_count + j];
            uint32_t score = scores[i * cfg.candidates_count + j];
            if (target == UINT32_MAX) std::cout << " (" << j << " target=none score=" << score << ")";
            else if (target >= UINT32_MAX - T::neighborsCount()) std::cout << " (" << j << " target=NEIGH(" << UINT32_MAX - target - 1 << ") score=" << score << ")";
            else std::cout << " (" << j << " target=" << target << " score=" << score << ")";
        }
        std::cout << "\n";
    }
}

template<Topology T>
void logSwapPairs(
    const slot *swap_slots,
    const uint32_t num_nodes
) {
    const int tid = 0;
    std::cout TID(tid) << "Swap pairs:\n";
    for (uint32_t i = 0; i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH); ++i) {
        const slot node_slot = swap_slots[i];
        if (slot_id(node_slot) == UINT32_MAX) std::cout TID(tid) << "  node " << i << " -> target=none score=" << slot_score(node_slot) << "\n";
        else if (slot_id(node_slot) >= UINT32_MAX - T::neighborsCount()) std::cout TID(tid) << "  node " << i << " -> target=NEIGH(" << UINT32_MAX - slot_id(node_slot) - 1 << ") score=" << slot_score(node_slot) << "\n";
        else std::cout TID(tid) << "  node " << i << " -> target=" << slot_id(node_slot) << " score=" << slot_score(node_slot) << "\n";
    }
}

void logEvents(
    const swap *ev_swaps,
    const float *ev_scores,
    const uint32_t num_nodes,
    const std::string flare
) {
    const int tid = 0;
    std::cout TID(tid) << "Events (" << flare << "):\n";
    for (uint32_t i = 0; i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH); ++i)
        std::cout TID(tid) << "  event " << i << " -> lo=" << ev_swaps[i].lo << " hi=" << ev_swaps[i].hi << " score=" << ev_scores[i] << "\n";
}

// TEMPLATE INSTANTIATIONS

#define INSTANTIATE_PLACEMENT_FUNCTIONS(T) \
    template void forceDirectedRefinement<T>( \
        const runconfig&, \
        const uint32_t*, const dim_t*, \
        const uint32_t*, const dim_t*, \
        const float*, uint32_t, uint32_t, uint32_t, \
        Coord_t<T>*, uint32_t*); \
 \
    template void getLocalityMetrics<T>( \
        const runconfig&, const Coord_t<T>*, \
        const uint32_t*, const dim_t*, \
        const uint32_t*, const float*, \
        uint32_t, uint32_t, uint32_t, float*, float*);

INSTANTIATE_PLACEMENT_FUNCTIONS(Lattice2D)
INSTANTIATE_PLACEMENT_FUNCTIONS(Torus6D)
INSTANTIATE_PLACEMENT_FUNCTIONS(ArbitraryGraph)

#undef INSTANTIATE_PLACEMENT_FUNCTIONS
