#include <bit>
#include <tuple>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include <unistd.h>

#include "runconfig.hpp"

#include "refinement.hpp"

#include "utils.hpp"
#include "prims.hpp"
#include "defines.hpp"
#include "chaining.hpp"
#include "constants.hpp"

using namespace config;

// memory available for new allocations (without swapping), in bytes
static size_t availableMemory() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::istringstream fields(line.substr(13));
            size_t kb = 0;
            fields >> kb;
            return kb * 1024;
        }
    }
    return (size_t)sysconf(_SC_AVPHYS_PAGES) * (size_t)sysconf(_SC_PAGESIZE);
}

// outcome of a refinement repeat
struct refinement_outcome {
    uint32_t best_rank;
    int32_t size_validity, inbounds_validity, pins_validity;
    float acquired_gain;
    bool applied;
};

// one refinement repeat, from the gains of moves to their application, given the pins per partition of the current partitioning
// NOTE: in CUDA this is the body of the refinement loop, written twice, once per pins per partition representation
template <typename PPP>
static refinement_outcome refinementRepeat(
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
    const uint32_t curr_num_nodes,
    const uint32_t num_hedges,
    const uint32_t num_partitions,
    const PPP pins_per_partitions,
    const bool flag_invalid_moves, // if true, invalid moves (no target partition) violate the size and pins constraints, making every later move invalid too
    const bool chainup,
    const bool encourage,
    const uint32_t fm_repeat,
    const uint32_t discount,
    uint32_t *pairs,
    float *f_scores,
    uint32_t *partitions,
    uint32_t *partitions_sizes,
    uint32_t *partitions_inbound_sizes,
    uint32_t *partitions_pins
) {
    // NOTE: the kernel writes every node's move and score, no need to zero-out its outputs
    LAUNCH(cfg) RUN << "fm-ref gains kernel (threads=" << cfg.threads << ") ...\n";
    fm_refinement_gains_kernel<PPP>(
        touching,
        touching_offsets,
        hedge_weights,
        partitions,
        pins_per_partitions,
        nodes_sizes,
        partitions_sizes,
        curr_num_nodes,
        num_partitions,
        fm_repeat,
        discount,
        encourage, // encourage all moves only when not doing k-way partitioning
        // NOTE: repurposing those from the candidates kernel!
        pairs, // -> moves: pairs[node] -> partition the node wants to join
        f_scores
    );

    // =============================
    // print some temporary results
    LOG(cfg) {
        logMoves(
            pairs,
            f_scores,
            partitions,
            curr_num_nodes
        );
    }
    // =============================

    buffer<uint32_t> ranks(curr_num_nodes); // rank[node idx] -> position of node's move in the sorted sequence

    // alternate between sequence ordering techniques
    if (chainup) {
        // sort scores and build an array of ranks (node id -> his move's idx in sorted scores)
        // use node ids as a tie-breaker when sorting moves (highest score first)
        // NOTE: in CUDA this is a comparison sort, where -0 and +0 are equal, hence "+ 0.0f" folds -0 into +0 before making radix keys
        const uint32_t node_bits = bits_for(curr_num_nodes);
        buffer<uint32_t> by_score = par_sort_permutation(curr_num_nodes, 32u + node_bits, [=](dim_t node) {
            return ((uint64_t)(~float_to_ordered_uint(f_scores[node] + 0.0f)) << node_bits) | (uint64_t)node;
        });
        #pragma omp parallel for schedule(static) if(curr_num_nodes > PARALLEL_GRAIN)
        for (uint32_t pos = 0; pos < curr_num_nodes; pos++)
            ranks[by_score[pos]] = pos; // invert the permutation such that: ranks[original_index] = sorted_position
    } else {
        // build move-chains to approximate high-gain swaps, then sort by chain total gain
        chaining(
            cfg,
            partitions,
            pairs,
            nodes_sizes,
            f_scores,
            curr_num_nodes,
            ranks.data()
        );
    }

    LAUNCH(cfg) RUN << "fm-ref cascade kernel (threads=" << cfg.threads << ") ...\n";
    fm_refinement_cascade_kernel<PPP>(
        hedges,
        hedges_offsets,
        touching,
        touching_offsets,
        hedge_weights,
        ranks.data(),
        pairs,
        partitions,
        pins_per_partitions,
        curr_num_nodes,
        encourage,
        f_scores
    );

    // not re-sorting the scores array means you have the array ordered as per the initial scores,
    // but now, this scan updates the scores "as if all previous moves were applied"!
    par_inclusive_scan<float>(f_scores, curr_num_nodes); // in-place (we don't need scores anymore anyway)
    // Remember: moves never get re-ranked (re-sorted) after the first time with in-isolation gains. Keep them like that and just find the valid sequence of maximum gain! This is an heuristics!

    // moves in rank order
    buffer<uint32_t> node_of_rank(curr_num_nodes); // node_of_rank[rank] -> node whose move has that rank
    #pragma omp parallel for schedule(static) if(curr_num_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < curr_num_nodes; node++)
        node_of_rank[ranks[node]] = node;
    buffer<uint32_t> moving_ranks = par_copy_if(curr_num_nodes, [&](uint32_t rank) { return pairs[node_of_rank[rank]] != UINT32_MAX; }); // moving_ranks[idx] -> rank of the idx-th valid move
    const uint32_t num_moving = (uint32_t)moving_ranks.size();

    // ======================================
    // extra step: compute moves validity by size and by inbound pins (same HP as the kernel above: all previous higher-gain moves will be applied)
    // explode each move into two events, one decrementing and incrementing the size of the src and dst partition respectively
    // => seeing each move as two distinct events makes us able to identify sequences of useful events first, then moves
    buffer<int32_t> valid_moves(curr_num_nodes); // valid_move[rank idx] -> 0 if applying all moves up to the idx one in the ordered sequence gives a valid state
    buffer<int32_t> pins_valid_moves(curr_num_nodes); // pins_valid_move[rank idx] -> same, w.r.t. the inbound pins constraint
    #pragma omp parallel for schedule(static) if(curr_num_nodes > PARALLEL_GRAIN)
    for (uint32_t rank = 0; rank < curr_num_nodes; rank++) {
        // NOTE: in CUDA the dense pins per partition path emits events for invalid moves too, only to flag them as invalid
        const int32_t invalid = flag_invalid_moves && pairs[node_of_rank[rank]] == UINT32_MAX ? 1 : 0;
        valid_moves[rank] = invalid;
        pins_valid_moves[rank] = invalid;
    }
    {
        const dim_t num_size_events = 2ull * num_moving;
        buffer<uint32_t> size_events_partition(num_size_events); // size_events_partition[ev] -> partition affected by the event
        buffer<uint32_t> size_events_index(num_size_events); // size_events_index[ev] -> sequence position / idx of the move (w.r.t. ranks) that originated the event
        buffer<int32_t> size_events_delta(num_size_events); // size_events_delta[ev] -> size variation brought by the event
        buffer<int32_t> pins_events_delta(num_size_events); // pins_events_delta[ev] -> inbound pins variation brought by the event
        LAUNCH(cfg) RUN << "build size events kernel (threads=" << cfg.threads << ") ...\n";
        build_size_events_kernel(
            pairs,
            node_of_rank.data(),
            moving_ranks.data(),
            partitions,
            nodes_sizes,
            nodes_pins,
            num_moving,
            size_events_partition.data(),
            size_events_index.data(),
            size_events_delta.data(),
            pins_events_delta.data()
        );

        // sort events by (partition, rank) [events are already in rank order, a stable sort by partition suffices] and carry both deltas along
        buffer<uint32_t> size_events_perm = par_sort_permutation(num_size_events, bits_for(num_partitions), [&](dim_t ev) { return (uint64_t)size_events_partition[ev]; });
        par_permute(size_events_perm.data(), num_size_events, size_events_partition);
        par_permute(size_events_perm.data(), num_size_events, size_events_index);
        par_permute(size_events_perm.data(), num_size_events, size_events_delta);
        par_permute(size_events_perm.data(), num_size_events, pins_events_delta);
        size_events_perm.release();
        // inclusive scan inside each key (= partition) on the event deltas => for each event we get the cumulative size and pins deltas for that partition at that point in the sequence
        const uint32_t *size_events_partition_ptr = size_events_partition.data();
        const auto same_partition = [=](dim_t a, dim_t b) { return size_events_partition_ptr[a] == size_events_partition_ptr[b]; };
        par_inclusive_scan_by_key<int32_t>(size_events_delta.data(), num_size_events, same_partition);
        par_inclusive_scan_by_key<int32_t>(pins_events_delta.data(), num_size_events, same_partition);

        // now mark moves that would violate the size or the pins constraint if the sequence were to end on them
        LAUNCH(cfg) RUN << "flag size events kernel (threads=" << cfg.threads << ") ...\n";
        flag_size_events_kernel(
            size_events_partition.data(),
            size_events_index.data(),
            size_events_delta.data(),
            partitions_sizes,
            num_size_events,
            max_nodes_per_part,
            valid_moves.data()
        );
        // NOTE: same kernel over the same events, it only cares that the per-node quantity is additive
        LAUNCH(cfg) RUN << "flag pins events kernel (threads=" << cfg.threads << ") ...\n";
        flag_size_events_kernel(
            size_events_partition.data(),
            size_events_index.data(),
            pins_events_delta.data(),
            partitions_pins,
            num_size_events,
            max_pins_per_part,
            pins_valid_moves.data()
        );
    }
    // compute, as of each event, the cumulative number of partitions that are invalid by summing the count of those made/unmade invalid at each event
    par_inclusive_scan<int32_t>(valid_moves.data(), curr_num_nodes);
    par_inclusive_scan<int32_t>(pins_valid_moves.data(), curr_num_nodes);

    // ======================================
    // preparatory step: update pins per partition into inbound (only) pins partition
    // simultaneously, also correct the calculation for partitions_inbound_sizes by removing outbounds
    LAUNCH(cfg) RUN << "inbound pins per partition kernel (threads=" << cfg.threads << ") ...\n";
    inbound_pins_per_partition_kernel<PPP>(
        hedges,
        hedges_offsets,
        srcs_count,
        partitions,
        num_hedges,
        num_partitions,
        pins_per_partitions, // from now it represents inbound sets only
        partitions_inbound_sizes
    );

    // ======================================
    // extra step: compute moves validity by inbound set cardinality (same HP as the kernel above: all previous higher-gain moves will be applied)
    // explode each move into two events for every inbound hedge of the moved node, one decrementing and one incrementing the hedge's
    // occurrencies in the src partition's inbound set and dst partition's inbound set respectively
    // => results in n*h events (better than the n*h*p volume of conditions/counters we need to check)
    buffer<int32_t> inbound_valid_moves(curr_num_nodes); // inbound_valid_move[rank idx] -> 0 if applying all moves up to the idx one in the ordered sequence gives a valid state
    par_fill<int32_t>(inbound_valid_moves.data(), curr_num_nodes, 0);
    {
        buffer<dim_t> inbound_count_events_offsets((dim_t)num_moving + 1); // inbound_count_events_offsets[idx] -> first event of the idx-th valid move
        #pragma omp parallel for schedule(static) if(num_moving > PARALLEL_GRAIN)
        for (uint32_t idx = 0; idx < num_moving; idx++)
            inbound_count_events_offsets[idx] = 2ull * inbound_count[node_of_rank[moving_ranks[idx]]];
        inbound_count_events_offsets[num_moving] = 0;
        par_exclusive_scan<dim_t>(inbound_count_events_offsets.data(), (dim_t)num_moving + 1);
        const dim_t num_inbound_count_events = inbound_count_events_offsets[num_moving];
        if (num_inbound_count_events > (dim_t)UINT32_MAX) {
            ERR(cfg) std::cerr << "ABORTING: refinement needs " << num_inbound_count_events
                << " inbound events, more than the " << UINT32_MAX << " addressable by the event kernels !!\n";
            abort();
        }

        buffer<uint32_t> inbound_count_events_partition(num_inbound_count_events); // inbound_count_events_partition[ev] -> partition affected by the event
        buffer<uint32_t> inbound_count_events_index(num_inbound_count_events); // inbound_count_events_index[ev] -> sequence position / idx of the move (w.r.t. ranks) that originated the event
        buffer<uint32_t> inbound_count_events_hedge(num_inbound_count_events); // inbound_count_events_hedge[ev] -> hedge involved in the event
        buffer<int32_t> inbound_count_events_delta(num_inbound_count_events); // inbound_count_events_delta[ev] -> inbound_count variation brought by the event
        LAUNCH(cfg) RUN << "build hedge events kernel (threads=" << cfg.threads << ") ...\n";
        build_hedge_events_kernel(
            pairs,
            node_of_rank.data(),
            moving_ranks.data(),
            partitions,
            touching,
            touching_offsets,
            inbound_count,
            inbound_count_events_offsets.data(),
            num_moving,
            inbound_count_events_partition.data(),
            inbound_count_events_index.data(),
            inbound_count_events_hedge.data(),
            inbound_count_events_delta.data()
        );
        inbound_count_events_offsets.release();

        // sort events by (partition, hedge, rank) [events are already in rank order, a stable sort by (partition, hedge) suffices] and carry events_delta along
        // the resulting array will have events sorted by partition, and inside each partition sorted by hedge, and inside each hedge sorted by rank!
        buffer<uint32_t> count_events_perm = par_sort_permutation(num_inbound_count_events, bits_for((uint64_t)num_partitions * num_hedges), [&](dim_t ev) {
            return (uint64_t)inbound_count_events_partition[ev] * num_hedges + inbound_count_events_hedge[ev];
        });
        par_permute(count_events_perm.data(), num_inbound_count_events, inbound_count_events_partition);
        par_permute(count_events_perm.data(), num_inbound_count_events, inbound_count_events_index);
        par_permute(count_events_perm.data(), num_inbound_count_events, inbound_count_events_hedge);
        par_permute(count_events_perm.data(), num_inbound_count_events, inbound_count_events_delta);
        count_events_perm.release();
        // inclusive scan by key of the deltas, the key being (partition, hedge) -> we now have the total number of times each hedge appears in the inbound set as of each move (in order of rank)
        const uint32_t *count_events_partition_ptr = inbound_count_events_partition.data();
        const uint32_t *count_events_hedge_ptr = inbound_count_events_hedge.data();
        par_inclusive_scan_by_key<int32_t>(inbound_count_events_delta.data(), num_inbound_count_events, [=](dim_t a, dim_t b) {
            return count_events_partition_ptr[a] == count_events_partition_ptr[b] && count_events_hedge_ptr[a] == count_events_hedge_ptr[b];
        });

        // new array of events, one event for each time the counter of an hedge in the inbound set (+ the overall inbounds per partition counter) goes from 0 to >0,
        // the event carrying a +1 to the inbound set size, one event for each time the counter of an hedge goes from >0 to 0 carrying a -1 to the inbound set size for that partition
        buffer<dim_t> inbound_size_events_offsets(num_inbound_count_events + 1); // inbound_size_events_offsets[event idx] -> initially a flag of whether each event will produce an increase/decrese in inbound counts, after the scan it becomes the offset of each new event
        inbound_size_events_offsets[0] = 0;
        LAUNCH(cfg) RUN << "count inbound events kernel (threads=" << cfg.threads << ") ...\n";
        count_inbound_size_events_kernel<PPP>(
            pins_per_partitions,
            inbound_count_events_partition.data(),
            inbound_count_events_hedge.data(),
            inbound_count_events_delta.data(),
            num_inbound_count_events,
            inbound_size_events_offsets.data()
        );

        // transform the counts in offsets with a scan and find the total count of new size events
        par_inclusive_scan<dim_t>(inbound_size_events_offsets.data(), num_inbound_count_events + 1);
        const dim_t num_inbound_size_events = inbound_size_events_offsets[num_inbound_count_events]; // last value in the inclusive scan = full reduce
        buffer<uint32_t> inbound_size_events_partition(num_inbound_size_events); // inbound_size_events_partition[ev] -> partition affected by the event
        buffer<uint32_t> inbound_size_events_index(num_inbound_size_events); // inbound_size_events_index[ev] -> sequence position / idx of the move (w.r.t. ranks) that originated the event
        buffer<int32_t> inbound_size_events_delta(num_inbound_size_events); // inbound_size_events_delta[ev] -> inbound set size variation brought by the event
        LAUNCH(cfg) RUN << "build inbound events kernel (threads=" << cfg.threads << ") ...\n";
        build_inbound_size_events_kernel<PPP>(
            pins_per_partitions,
            inbound_count_events_partition.data(),
            inbound_count_events_index.data(),
            inbound_count_events_hedge.data(),
            inbound_count_events_delta.data(),
            inbound_size_events_offsets.data(),
            num_inbound_count_events,
            inbound_size_events_partition.data(),
            inbound_size_events_index.data(),
            inbound_size_events_delta.data()
        );
        inbound_count_events_partition.release();
        inbound_count_events_index.release();
        inbound_count_events_hedge.release();
        inbound_count_events_delta.release();
        inbound_size_events_offsets.release();

        // sort events by (partition, rank) and carry inbound_size_events_delta along
        // NOTE: events of the same move on the same partition all carry the same delta, their relative order is irrelevant
        buffer<uint32_t> size_events_perm = par_sort_permutation(num_inbound_size_events, bits_for((uint64_t)num_partitions * curr_num_nodes), [&](dim_t ev) {
            return (uint64_t)inbound_size_events_partition[ev] * curr_num_nodes + inbound_size_events_index[ev];
        });
        par_permute(size_events_perm.data(), num_inbound_size_events, inbound_size_events_partition);
        par_permute(size_events_perm.data(), num_inbound_size_events, inbound_size_events_index);
        par_permute(size_events_perm.data(), num_inbound_size_events, inbound_size_events_delta);
        size_events_perm.release();
        // inclusive scan inside each key (= partition) on the event deltas => for each event we get the cumulative size delta for that partition's inbound set at that point in the sequence
        const uint32_t *size_events_partition_ptr = inbound_size_events_partition.data();
        par_inclusive_scan_by_key<int32_t>(inbound_size_events_delta.data(), num_inbound_size_events, [=](dim_t a, dim_t b) { return size_events_partition_ptr[a] == size_events_partition_ptr[b]; });

        // now mark moves that would violate the inbound set size constraint if the sequence were to end on them
        LAUNCH(cfg) RUN << "flag inbound events kernel (threads=" << cfg.threads << ") ...\n";
        flag_inbound_events_kernel(
            inbound_size_events_partition.data(),
            inbound_size_events_index.data(),
            inbound_size_events_delta.data(),
            partitions_inbound_sizes,
            num_inbound_size_events,
            inbound_valid_moves.data()
        );
    }
    // compute, as of each event, the cumulative number of partitions that are invalid by summing the count of those made/unmade invalid at each event
    par_inclusive_scan<int32_t>(inbound_valid_moves.data(), curr_num_nodes);

    // ======================================
    // find the move in the sequence that yields both the highest gain and a valid state (when all moves before it are applied)
    // functor comparing sequence entries, skipping invalid ones by inbound size (only 0 allowed), prioritizing size and pins events (zero or negative), and then picking the highest score
    const best_move_functor best_scores { f_scores, valid_moves.data(), inbound_valid_moves.data(), pins_valid_moves.data() };
    // max over valid endpoints only, find the point in the sequence of moves where applying them further never nets a higher gain in a valid state
    uint32_t best_rank = par_max_element(curr_num_nodes, best_scores);
#ifdef ABLATE_ONE_MOVE_PER_ROUND
    // Ablation (evaluation.md 4.10): conservative one-move-per-round apply --
    // take only the single highest-gain move instead of the best improving valid prefix.
    best_rank = 0u;
#endif
    const uint32_t num_good_moves = best_rank + 1; // "+1" to make this the improving moves count, rather than the last improving move's idx
    // validity double-check
    refinement_outcome outcome {
        best_rank,
        valid_moves[best_rank],
        inbound_valid_moves[best_rank],
        pins_valid_moves[best_rank],
        f_scores[best_rank],
        false
    };
    INFO(cfg) std::cout << "Best fm-ref move:\n  Move rank: " << best_rank << ", Acquired gain: " << outcome.acquired_gain << "\n";
    if (outcome.size_validity <= 0 && outcome.inbounds_validity <= 0 && outcome.pins_validity <= 0 && outcome.acquired_gain >= 0) {
        LAUNCH(cfg) RUN << "fm-ref apply (" << num_good_moves << " good moves) kernel (threads=" << cfg.threads << ") ...\n";
        fm_refinement_apply_kernel(
            pairs,
            ranks.data(),
            nodes_sizes,
            nodes_pins,
            curr_num_nodes,
            num_good_moves,
            partitions,
            partitions_sizes,
            partitions_pins
        );
        outcome.applied = true;
    }
    return outcome;
}

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
) {
    // prepare this level's pins per partition
    // NOTE: the inbound counters per partition are just the transposed of pins per partition! No need to compute them separately!
    const dim_t pins_per_partitions_size = static_cast<dim_t>(num_hedges) * num_partitions;
    // |
    // if we are short on memory, go for the sparse pins-per-partition representation
    // NOTE: "8 * 2 * touching_size" roughly accounts for the events buffers
    const size_t dense_bytes = (pins_per_partitions_size + 6ull * curr_num_nodes + 8ull * 2 * touching_size) * sizeof(uint32_t);
    const bool sparse = cfg.ppp_mode == PinsPerPartMode::SPARSE || (cfg.ppp_mode == PinsPerPartMode::AUTO && (double)dense_bytes > DENSE_PPP_RAM_FRACTION * (double)availableMemory());
    if (sparse && cfg.ppp_mode == PinsPerPartMode::AUTO)
        INFO(cfg) std::cout << "Not enough memory to allocate the dense pins per partition matrix: switching to the sparse version\n";

    buffer<uint32_t> pins_per_partitions; // dense: pins_per_partitions[hedge idx * num_partitions + partition idx] -> number of pins of that partition owned by this hedge
    buffer<bitmap> ppp_offsets; // sparse: ppp_offsets[hedge-idx * ceil(num_partitions / 64) + part-idx / 64] -> bitmap to access the pin count for all (hedge, part / 64), ... (hedge, part / 64 + 63) pairs
    buffer<uint32_t> ppp; // sparse: ppp[ppp_offsets[...].cnt + bits-at-one-before-the(p%64)th-in(ppp_offsets[...].flg)] -> number of pins of partition p owned by hedge e
    const uint32_t ppp_per_hedge = (num_partitions + BITMAP_CAPACITY - 1) / BITMAP_CAPACITY; // aka: ceil(num_partitions / 64)
    const dim_t ppp_offsets_size = static_cast<dim_t>(num_hedges) * ppp_per_hedge;
    if (!sparse) {
        if (pins_per_partitions_size * sizeof(uint32_t) > (1ull << 32))
            INFO(cfg) std::cout << "Allocating " << std::fixed << std::setprecision(1) << (float)(pins_per_partitions_size * sizeof(uint32_t)) / (1 << 30) << " GB for pins-per-partition ...\n";
        pins_per_partitions = buffer<uint32_t>(pins_per_partitions_size);
    } else {
        if (ppp_offsets_size * sizeof(bitmap) > (1ull << 32))
            INFO(cfg) std::cout << "Allocating " << std::fixed << std::setprecision(1) << (float)(ppp_offsets_size * sizeof(bitmap)) / (1 << 30) << " GB for pins-per-partition bitmaps ...\n";
        ppp_offsets = buffer<bitmap>(ppp_offsets_size);
    }

    // settings for refinement
    bool chainup = false; // true -> directly sort moves into a sequence by gain, false -> chain moves by size, then sort chains into a sequence
    bool encourage = cfg.mode == Mode::INCC; // true -> give a gain to moves that don't fully disconnect an hedge, doing so proportionally to how few pins the leave behind

    for (uint32_t fm_repeat = 0u; fm_repeat < cfg.refine_repeats; fm_repeat++) {
        INFO(cfg) std::cout << "Refining level " << level_idx << " repeat " << fm_repeat << ", remaining nodes=" << curr_num_nodes << " number of partitions=" << num_partitions << (sparse ? " (mode: sparse-ppp)" : "") << "\n";

        // by how much of a node's size to allow an invalid move to be proposed (but filtered later by events - if still invalid)
        uint32_t discount = fm_repeat < cfg.refine_repeats / 3 ? 1u : (fm_repeat < 2 * cfg.refine_repeats / 3 ? 2u : UINT32_MAX);

        // while computing pins per partition also compute the distinct incident counts per partition (number of pins with a count > 0)
        par_fill<uint32_t>(partitions_inbound_sizes, num_partitions, 0u);

        refinement_outcome outcome;
        if (!sparse) {
            // compute all pins per partition entries
            // NOTE: having this available during FM refinement makes its complexity linear in the connectivity, instead of quadratic!
            par_fill<uint32_t>(pins_per_partitions.data(), pins_per_partitions_size, 0u);
            LAUNCH(cfg) RUN << "pins per partition kernel (threads=" << cfg.threads << ") ...\n";
            pins_per_partition_kernel(
                hedges,
                hedges_offsets,
                partitions,
                num_hedges,
                num_partitions,
                pins_per_partitions.data(),
                partitions_inbound_sizes // as of here, this will be incorrect (also including outbounds)
            );
            outcome = refinementRepeat<dense_ppp>(
                cfg, hedges, hedges_offsets, srcs_count, touching, touching_offsets, inbound_count, hedge_weights, nodes_sizes, nodes_pins,
                curr_num_nodes, num_hedges, num_partitions,
                dense_ppp { pins_per_partitions.data(), num_partitions },
                true, // flag invalid moves
                chainup, encourage, fm_repeat, discount,
                pairs, f_scores, partitions, partitions_sizes, partitions_inbound_sizes, partitions_pins
            );
        } else {
            // build of offsets bitmaps
            par_fill<bitmap>(ppp_offsets.data(), ppp_offsets_size, bitmap { 0ull, 0ull });
            LAUNCH(cfg) RUN << "sparse pins per partition count kernel (threads=" << cfg.threads << ") ...\n";
            sparse_pins_per_partition_count_kernel(
                hedges,
                hedges_offsets,
                partitions,
                num_hedges,
                ppp_per_hedge,
                ppp_offsets.data()
            );

            // exclusive scan of each bitmap counter
            bitmap *ppp_offsets_ptr = ppp_offsets.data();
            par_exclusive_scan_with<uint64_t>(ppp_offsets_size, [=](dim_t i) { return ppp_offsets_ptr[i].cnt; }, [=](dim_t i, uint64_t v) { ppp_offsets_ptr[i].cnt = v; });

            // compute non-zero entries count, re-allocate IFF a larger segmented array is needed
            const bitmap last_bitmap = ppp_offsets[ppp_offsets_size - 1]; // last entry in the exclusive scan -> add to its cnt the number of bits set to 1 in flg to have the non-zero entries count
            const dim_t new_ppp_size = last_bitmap.cnt + std::popcount(last_bitmap.flg);
            if (new_ppp_size > ppp.size()) {
                if (new_ppp_size * sizeof(uint32_t) > (1ull << 32))
                    INFO(cfg) std::cout << "Allocating " << std::fixed << std::setprecision(1) << (float)(new_ppp_size * sizeof(uint32_t)) / (1 << 30) << " GB for sparse pins-per-partition ...\n";
                ppp = buffer<uint32_t>(new_ppp_size);
            }

            // fill in the segmented array, compute incident hedges counts per partition while you are at it
            par_fill<uint32_t>(ppp.data(), ppp.size(), 0u);
            LAUNCH(cfg) RUN << "sparse pins per partition write kernel (threads=" << cfg.threads << ") ...\n";
            sparse_pins_per_partition_write_kernel(
                hedges,
                hedges_offsets,
                partitions,
                ppp_offsets.data(),
                num_hedges,
                num_partitions,
                ppp_per_hedge,
                ppp.data(),
                partitions_inbound_sizes // NOTE: here filled with outbounds too
            );
            // NOTE: in CUDA, when short on VRAM, the sparse path discards the lowest ranked moves (an "emergency" measure), here it never does
            outcome = refinementRepeat<sparse_ppp>(
                cfg, hedges, hedges_offsets, srcs_count, touching, touching_offsets, inbound_count, hedge_weights, nodes_sizes, nodes_pins,
                curr_num_nodes, num_hedges, num_partitions,
                sparse_ppp { ppp_offsets.data(), ppp.data(), ppp_per_hedge, num_partitions },
                false, // do not flag invalid moves
                chainup, encourage, fm_repeat, discount,
                pairs, f_scores, partitions, partitions_sizes, partitions_inbound_sizes, partitions_pins
            );
        }

        if (!outcome.applied) {
            INFO(cfg) {
                std::cout << "No valid refinement move found on level " << level_idx << " - reason: "
                    << (outcome.size_validity > 0 ? (outcome.inbounds_validity > 0 ? "both size and inbounds validities" : "size validity") : (outcome.inbounds_validity > 0 ? "inbounds validity" : (outcome.pins_validity > 0 ? "pins validity" : "negative gain"))) << "\n";
                if (outcome.size_validity > 0) std::cout << "  Size constraint violations variation amount (in nodes above the limit): " << outcome.size_validity << "\n";
                if (outcome.inbounds_validity > 0) std::cout << "  Inbound constraint violations variation (in invalid partitions): " << outcome.inbounds_validity << "\n";
                if (outcome.pins_validity > 0) std::cout << "  Pins constraint violations variation amount (in pins above the limit): " << outcome.pins_validity << "\n";
            }
            if (outcome.size_validity > 0 && !chainup) chainup = true; // enable chaining when no moves are available via greedy sorting because of size constraints
            else if (fm_repeat < cfg.refine_repeats / 3) fm_repeat = cfg.refine_repeats / 2;
            else if (fm_repeat < 2 * cfg.refine_repeats / 3) fm_repeat = 2 * cfg.refine_repeats / 3;
            else fm_repeat = cfg.refine_repeats; // aka break!
        }
    }

    // recompute inbound set sizes
    if (update_final_inbound_counts) {
        par_fill<uint32_t>(partitions_inbound_sizes, num_partitions, 0u);
        LAUNCH(cfg) RUN << "inbound sets size kernel (threads=" << cfg.threads << ") ...\n";
        inbound_sets_size_kernel(
            hedges,
            hedges_offsets,
            srcs_count,
            partitions,
            num_hedges,
            num_partitions,
            partitions_inbound_sizes
        );
    }
}


// LOGGING

void logPartitions(
    const uint32_t *partitions_tmp,
    const uint32_t *partitions_sizes_tmp,
    const uint32_t *partitions_inbound_sizes_tmp,
    const uint32_t *partitions_pins_tmp,
    const uint32_t curr_num_nodes,
    const uint32_t num_partitions
) {
    std::unordered_map<uint32_t, int> part_count;
    std::cout << "Partitioning results:\n";
    for (uint32_t i = 0; i < curr_num_nodes; ++i) {
        uint32_t part = partitions_tmp[i];
        part_count[part]++;
        if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH)) {
            if (part == UINT32_MAX) std::cout << "node " << i << " -> part=none";
            else std::cout << "  node " << i << " -> " << part;
            std::cout << ((i + 1) % 4 == 0 ? "\n" : "\t");
        }
    }
    for (uint32_t i = 0; i < num_partitions; ++i) {
        uint32_t part_size = partitions_sizes_tmp[i];
        uint32_t part_inbound_size = partitions_inbound_sizes_tmp[i];
        uint32_t part_pins = partitions_pins_tmp[i];
        if (part_size > max_nodes_per_part)
           std::cerr << "  WARNING, max partition size constraint (" << max_nodes_per_part << ") violated by part=" << i << " with part_size=" << part_size << " !!\n";
        if (part_inbound_size > max_inbound_per_part)
            std::cerr << "  WARNING, max partition inbound size constraint (" << max_inbound_per_part << ") violated by part=" << i << " with part_inbound_size=" << part_inbound_size << " !!\n";
        if (part_pins > max_pins_per_part)
            std::cerr << "  WARNING, max partition pins constraint (" << max_pins_per_part << ") violated by part=" << i << " with part_pins=" << part_pins << " !!\n";
    }
    int max_ps = part_count.empty() ? 0 : std::max_element(part_count.begin(), part_count.end(), [](auto &a, auto &b){ return a.second < b.second; })->second;
    std::cout << "Non-empty partitions count: " << part_count.size() << ", Max partition size: " << max_ps << "\n";
}

void logMoves(
    const uint32_t *moves_tmp,
    const float *gains_tmp,
    const uint32_t *src_partitions_tmp,
    const uint32_t curr_num_nodes
) {
    std::cout << "Proposed moves:\n";
    for (uint32_t i = 0; i < curr_num_nodes; ++i) {
        if (i < std::min<uint32_t>(curr_num_nodes, VERBOSE_LENGTH)) {
            std::cout << "  node " << i << " : ";
            uint32_t move = moves_tmp[i];
            if (move == UINT32_MAX) std::cout << "stay " << src_partitions_tmp[i];
            else std::cout << src_partitions_tmp[i] << " -> " << move;
            std::cout << " gain=" << std::fixed << std::setprecision(3) << gains_tmp[i];
            std::cout << ((i + 1) % 2 == 0 ? "\n" : "\t");
        }
    }
}
