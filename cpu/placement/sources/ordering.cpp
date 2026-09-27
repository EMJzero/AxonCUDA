#include <vector>
#include <cassert>
#include <climits>
#include <iomanip>
#include <stdint.h>
#include <iostream>
#include <algorithm>

#include "runconfig_plc.hpp"

#include "utils.hpp"
#include "prims.hpp"
#include "utils_plc.hpp"
#include "ordering.hpp"

// NOTE: the whole batch of multi-starts is ordered by one call of every kernel per step
// => per-multi-start arrays are one flat allocation of "batch_size" equally-sized segments, multi-start "b" owning [b*size, (b+1)*size)
// => partition ids are composite, "b*num_parts + p", so a single global sort/scan/reduce keyed by partition already keeps multi-starts apart
// => node idxs are batch-flat, hypergraph pin idxs are not

// for each partition, randomly bisect it, mapping every partition id "p" to either "p*2" or "p*2+1"
void split_partitions_rand(
    const runconfig &cfg,
    uint32_t* partitions,
    uint32_t num_nodes,
    uint32_t num_parts,
    uint32_t batch_size,
    std::vector<xorwow_generator> &gens
) {
    const uint32_t batch_nodes = batch_size * num_nodes;
    const uint32_t batch_parts = batch_size * num_parts;

    // generate one random uint32 per element, seeded
    // NOTE: one generator per multi-start, so a multi-start draws the very same sequence whatever it is batched with
    buffer<uint32_t> rand_keys(batch_nodes); // rand_keys[node idx] -> random tie-breaking key for the node
    for (uint32_t start = 0; start < batch_size; start++)
        gens[start].generate(rand_keys.data() + start * num_nodes, num_nodes);

    // sort by (partition, random), carrying along the original indices
    // => now partitions_cpy is grouped by composite "p", with random order inside each group
    // NOTE: in CUDA this is a comparison sort of (partition, random) pairs, stable since thrust merge-sorts tuple keys, here a stable radix sort of both keys packed in one
    buffer<uint32_t> original_idx = par_sort_permutation(batch_nodes, 32u + bits_for(batch_parts), [&](dim_t i) {
        return ((uint64_t)partitions[i] << 32) | (uint64_t)rand_keys[i];
    }); // original_idx[i] -> idx of node currently in partition partitions_cpy[i]
    rand_keys.release();
    buffer<uint32_t> partitions_cpy(batch_nodes); // auxiliary copy of current partitions for sorting and scattering
    par_gather(original_idx.data(), batch_nodes, partitions, partitions_cpy.data());

    // build offset indices over reordered partitions
    buffer<uint32_t> part_offsets((dim_t)batch_parts + 1); // part_offsets[p] -> first index of partition p in partitions_cpy
    par_lower_bound(partitions_cpy.data(), batch_nodes, batch_parts, part_offsets.data());
    part_offsets[batch_parts] = batch_nodes;

    // split each partition in half:
    // - inside each partition, original node indices are not randomly ordered
    // - take the lower half of those indices and map it to p*2, take the upper half and map it to p*2+1
    LAUNCH(cfg) RUN << "split partitions kernel (threads=" << cfg.threads << ") ...\n";
    split_partitions_kernel(
        part_offsets.data(),
        num_nodes,
        batch_size,
        partitions_cpy.data()
    );

    // undo the sort - scatter back to updated partitions to their original idxs
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < batch_nodes; i++)
        partitions[original_idx[i]] = partitions_cpy[i];
}

void compute_partitions_cutnet(
    const runconfig &cfg,
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const float* hedge_weights,
    const uint32_t* partitions,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t num_parts,
    const uint32_t batch_size,
    const dim_t hedges_size,
    float* cutnet
) {
    /*
    * IDEA:
    * - prepare a copy of the segmented hedge buffer
    * - map operation to replace each pin with its partition
    * - segmented sort inside each hedge
    * - filter operation to keep only (within each segmente) the even numbers that are followed by their value +1 (their odd partition in the pair)
    *   - not need exactly to remove the elements, but to spot relevant ones
    * - flag surviving elements and prefix sum the flags, this gives you a unique offset per element
    * - for each surviving element create an event containing the tuple (hedge weight, partition id / 2), divide by 2 to get the parent partition's id
    * - sort events by parent partition id, and do a segmented reduce within each parent id
    *   => that yields each parent partition's weighted minority pin-cut across its bisection
    */

    // NOTE: in CUDA a batch whose pins exceed INT_MAX is handled in chunks of multi-starts, as CUB counts items with an int, here all at once
    const uint32_t batch_part_pairs = batch_size * (num_parts / 2);
    const dim_t batch_pins = (dim_t)batch_size * hedges_size;
    const uint32_t batch_hedges = batch_size * num_hedges;

    // initialize every partition pair as "fully trapped"; pairs that generate events will overwrite their true split cost
    par_fill<float>(cutnet, batch_part_pairs, 0.0f);

    // map pins to their partition -> each multi-start maps the shared pin sequence through its own partitioning
    buffer<uint32_t> part_pins(batch_pins); // part_pins[start*hedges_size + hedges_offsets[hedge idx] + pin idx] -> partition the pin is in
    #pragma omp parallel for schedule(static) if(batch_pins > PARALLEL_GRAIN)
    for (dim_t idx = 0; idx < batch_pins; idx++) {
        const dim_t my_start = idx / hedges_size;
        part_pins[idx] = partitions[my_start * num_nodes + hedges[idx - my_start * hedges_size]];
    }

    // segmented sort of part_pins (using the segments from hedges, one set of them per multi-start)
    par_segmented_sort<uint32_t>(part_pins.data(), batch_hedges, [=](uint32_t seg) {
        const uint32_t my_start = seg / num_hedges;
        return my_start * hedges_size + hedges_offsets[seg - my_start * num_hedges];
    });

    // NOTE: 32 bits are enough, these end up holding event offsets, and events never outnumber pins
    buffer<uint32_t> flags(batch_pins + 1); // flags[pin idx] -> 1 if the pin starts an event, then (after the scan) the event's idx
    par_fill<uint32_t>(flags.data(), batch_pins + 1, 0u);
    LAUNCH(cfg) RUN << "flag cutnet events kernel (threads=" << cfg.threads << ") ...\n";
    flag_cutnet_events_kernel(
        part_pins.data(),
        hedges_offsets,
        num_hedges,
        batch_size,
        hedges_size,
        flags.data()
    );

    // exclusive prefix sum of flags, then extract the last value (total sum) as the events count
    par_exclusive_scan<uint32_t>(flags.data(), batch_pins + 1);
    const uint32_t events_count = flags[batch_pins];

    if (events_count > 0) {
        buffer<float> event_weight(events_count); // event_weight[idx] -> weight of the hedge being cut in event idx
        buffer<uint32_t> event_part(events_count); // event_part[idx] -> composite partition/2 affected by event idx

        // for each part_pins entry that previously generated a flag, use the new prefix-summed flags as the index in event_weight and event_part
        // where to let that part_pins entry write its content (in event_part) and its hedge's weight (in event_weight)
        LAUNCH(cfg) RUN << "cutnet event generation kernel (threads=" << cfg.threads << ") ...\n";
        cutnet_event_generation_kernel(
            part_pins.data(),
            hedges_offsets,
            hedge_weights,
            flags.data(),
            num_hedges,
            batch_size,
            hedges_size,
            event_weight.data(),
            event_part.data()
        );

        // sort event_weight and event_part both according to event_part
        // => partitions are composite, so this single sort already groups every multi-start's events apart
        buffer<uint32_t> perm = par_sort_permutation(events_count, bits_for(batch_part_pairs), [&](dim_t i) { return (uint64_t)event_part[i]; });
        par_permute(perm.data(), events_count, event_part);
        par_permute(perm.data(), events_count, event_weight);

        // reduce-sum each segment of event_weight with the same event_part value and store the result in cutnet[event_part[.]]
        // => although the buffer is still called "cutnet", it now stores weighted minority pin-cut
        // NOTE: in CUDA this is a reduce_by_key, whose float sums associate differently (see README)
        par_for_each_run(
            events_count,
            [&](dim_t a, dim_t b) { return event_part[a] == event_part[b]; },
            [&](dim_t begin, dim_t end) {
                float sum = event_weight[begin];
                for (dim_t i = begin + 1; i < end; i++) sum += event_weight[i];
                cutnet[event_part[begin]] = sum;
            }
        );
    }
}

// return a high-locality, seeded 1D ordering of nodes
buffer<uint32_t> locality_ordering(
    const runconfig &cfg,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t num_hedges,
    const dim_t hedges_size,
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const float* hedge_weights,
    const uint32_t* touching,
    const dim_t* touching_offsets,
    const uint64_t seed
) {
    const int tid = 0;
    /*
    * IDEA:
    * - recursive bisection
    * - divide nodes in two halves, and each half in half again, and so creating a binary tree where each leaf is a single node
    *  - done with a random balanced bipartitioning and repeated label propagation
    * - then go back up the binary tree
    * - at each branch, decide how to order the two halves ("which goes first")
    *   - check whether you are on the left or right side of your grandparent
    *   - compute how strongly each half is connected with the other side of your grandparent
    *   - if the grandparent has you on the left, and your left side is more strongly connected to grandpa's right,
    *       reverse the order of every leaf under yourself, trapping good connections inside, while favoring links with grandpa's second child
    *   - else, everything stays as it is  - mirrored idea if you are on grandpa's right side
    * - recurse up until the root, that gives you a strong 1D ordering that trapped connection locality as much as possible
    */

    /*
    * How to generate partition ids:
    * - given a partition with id "p", currently being bisected, its two child partitions will have ids:
    *   - p*2   - p*2 + 1
    * - this works because the number of partitions doubles at every bisection of every partition
    * - uneven partitions will lead to some non-existing id, be wary
    * - when re-merging partitions, fuse all even "p"s with their odd successor and give both the "p/2" id
    */
    assert(num_nodes > 0); // how, why, what?!

    const uint32_t batch_nodes = batch_size * num_nodes;

    // one generator per multi-start, so a multi-start draws the very same sequence whatever it is batched with
    std::vector<xorwow_generator> gens;
    gens.reserve(batch_size);
    for (uint32_t start = 0; start < batch_size; start++)
        gens.emplace_back(seed + start);

    // everyone starts in the same partition, which at num_parts == 1 makes the composite id coincide with the multi-start's own idx
    buffer<uint32_t> partitions(batch_nodes); // partitions[node idx] -> current composite partition (of bypartitions) the node is in
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < batch_nodes; node++)
        partitions[node] = node / num_nodes;

    uint32_t num_parts = 1u;

    buffer<bool> moves(batch_nodes); // move[node idx] -> false if the node doesn't want to move, true if the node would like to switch partition p*2->p*2+1 or p*2+1->p*2
    buffer<float> scores(batch_nodes); // score[node idx] -> connectivity gain for the above move (even not moving is done with a "gain")
    buffer<uint8_t> active(batch_size); // active[start] -> 0 once that multi-start stopped improving at the current level

    // NOTE: move events are not compacted, node idx "i" owns event slot "i" in one of the two lists
    // => the slots it does not own carry UINT32_MAX as partition, so they sort behind every real event of their own multi-start
    buffer<uint32_t> even_event_part(batch_nodes); // part[idx] -> composite src partition / 2 for the idx-th move (partition being even)
    buffer<float> even_event_score(batch_nodes); // score[idx] -> gain for the idx-th move
    buffer<uint32_t> even_event_node(batch_nodes); // node[idx] -> node moved in the idx-th move
    buffer<uint32_t> odd_event_part(batch_nodes); // part[idx] -> composite (src partition - 1) / 2 ... (partition being odd)
    buffer<float> odd_event_score(batch_nodes); // score[idx] -> ...
    buffer<uint32_t> odd_event_node(batch_nodes); // node[idx] -> ...

    // one slot past the nodes is the scratch bin the empty event slots scatter their rank into
    buffer<uint32_t> even_ranks((dim_t)batch_nodes + 1); // ranks[node idx] -> even event index for node idx (UINT32_MAX if no event)
    buffer<uint32_t> odd_ranks((dim_t)batch_nodes + 1); // ranks[node idx] -> ...

    // num_parts/2 never exceeds num_nodes, so one slot per node per multi-start is always enough
    buffer<uint32_t> apply_up_to(batch_nodes); // apply_up_to[p/2] -> last absolute event idx to apply for composite partitions p and p+1

    // IDEA:
    // - initialize this on each level from current partitions
    // - after label prop, compute the new split cost for each pair of partitions
    // - iff a partitions pair's split cost improved, copy over here the new partition ids for the nodes of that pair of partitions
    // - before going to the next level, make this the actual partitioning
    buffer<uint32_t> last_best_partitions(batch_nodes); // last_best_partitions [node idx] -> last best composite partition the node was in

    uint32_t level_idx = 0u;
    while (num_parts < (num_nodes + 1) / 2) { // as long as partitions do not strictly contain 1 or 2 nodes...
        INFO(cfg) std::cout TID(tid) << "Bisection level " << level_idx << " number of partitions=" << num_parts << "\n";
        level_idx++;

        // random bisection of every partition
        split_partitions_rand(
            cfg,
            partitions.data(),
            num_nodes,
            num_parts,
            batch_size,
            gens
        );
        num_parts *= 2;
        const uint32_t batch_part_pairs = batch_size * (num_parts / 2);

        par_copy(last_best_partitions.data(), partitions.data(), batch_nodes);

        buffer<float> cutnet(batch_part_pairs); // cutnet[p/2] -> weighted minority pin-cut of the bisection of composite partition p/2
        buffer<float> last_best_cutnet(batch_part_pairs); // last_best_cutnet[p/2] -> best such cost seen so far at this level

        // baseline split cost of the freshly randomized bisection
        compute_partitions_cutnet(
            cfg,
            hedges,
            hedges_offsets,
            hedge_weights,
            partitions.data(),
            num_hedges,
            num_nodes,
            num_parts,
            batch_size,
            hedges_size,
            last_best_cutnet.data()
        );

        // build offset indices over reordered events per partition
        buffer<uint32_t> part_even_event_offsets((dim_t)batch_part_pairs + 1); // part_even_event_offsets[p] -> first index of composite partition p*2 in even_event_part
        buffer<uint32_t> part_odd_event_offsets((dim_t)batch_part_pairs + 1); // part_odd_event_offsets[p] -> first index of composite partition p*2+1 in odd_event_part

        // every multi-start re-enters each level active, the label propagation retires them as they run out of improving moves
        par_fill<uint8_t>(active.data(), batch_size, 1u);

        for (uint32_t lp_repeat = 0u; lp_repeat < cfg.labelprop_repeats; lp_repeat++) {
            // compute gains (and moves) in-isolation
            // NOTE: no need to init. "moves" and "scores", they are overwritten anyway
            LAUNCH(cfg) TID(tid) RUN << "label propagation kernel (threads=" << cfg.threads << ") ...\n";
            label_propagation_kernel(
                hedges,
                hedges_offsets,
                touching,
                touching_offsets,
                hedge_weights,
                partitions.data(),
                num_nodes,
                batch_size,
                active.data(),
                moves.data(),
                scores.data()
            );

            // build move events (partition, score, node)
            LAUNCH(cfg) TID(tid) RUN << "label move events kernel (threads=" << cfg.threads << ") ...\n";
            label_move_events_kernel(
                moves.data(),
                scores.data(),
                partitions.data(),
                num_nodes,
                batch_size,
                active.data(),
                even_event_part.data(),
                even_event_score.data(),
                even_event_node.data(),
                odd_event_part.data(),
                odd_event_score.data(),
                odd_event_node.data()
            );

            // sort events by (partition, score, node)
            // => partitions are composite, so a single sort already groups every multi-start's events apart, empty slots trailing behind
            // NOTE: in CUDA this is a comparison sort of (partition, score, node) triplets, here a stable radix sort of (partition, score),
            //       since event slots are already in node order
            {
                buffer<uint32_t> perm = par_sort_permutation(batch_nodes, 64u, [&](dim_t i) {
                    return ((uint64_t)even_event_part[i] << 32) | (uint64_t)float_to_ordered_uint(even_event_score[i]);
                });
                par_permute(perm.data(), batch_nodes, even_event_part);
                par_permute(perm.data(), batch_nodes, even_event_score);
                par_permute(perm.data(), batch_nodes, even_event_node);
            }
            // |
            {
                buffer<uint32_t> perm = par_sort_permutation(batch_nodes, 64u, [&](dim_t i) {
                    return ((uint64_t)odd_event_part[i] << 32) | (uint64_t)float_to_ordered_uint(odd_event_score[i]);
                });
                par_permute(perm.data(), batch_nodes, odd_event_part);
                par_permute(perm.data(), batch_nodes, odd_event_score);
                par_permute(perm.data(), batch_nodes, odd_event_node);
            }

            // NOTE: the search was "made to work" by storing p/2 inside event_part-s, hence it is enough to search from 0 to batch_part_pairs
            // => searching one past the last pair yields the total count of real events, so no boundary needs writing by hand
            par_lower_bound(even_event_part.data(), batch_nodes, batch_part_pairs + 1, part_even_event_offsets.data());
            // |
            par_lower_bound(odd_event_part.data(), batch_nodes, batch_part_pairs + 1, part_odd_event_offsets.data());

            // build the reverse map: ranks[node] -> event-idx (if any) of node - in other words this scatter does "ranks[event_node[i]] = i"
            // => empty slots all point at the scratch bin one past the nodes, so they write there and are ignored
            par_fill<uint32_t>(even_ranks.data(), (dim_t)batch_nodes + 1, UINT32_MAX);
            par_fill<uint32_t>(odd_ranks.data(), (dim_t)batch_nodes + 1, UINT32_MAX);
            #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
            for (uint32_t i = 0; i < batch_nodes; i++) {
                even_ranks[even_event_node[i]] = i;
                odd_ranks[odd_event_node[i]] = i;
            }

            // update gains in-sequence
            // assume moves are done in pairs => re-compute the pair's gain in-sequence, assuming all prior pairs already swapped
            // => already accumulate the two scores on the "even" segment's event (only up to the length of the smallest events segment between even and odd)
            par_fill<float>(even_event_score.data(), batch_nodes, 0.0f);
            LAUNCH(cfg) TID(tid) RUN << "label cascade kernel (threads=" << cfg.threads << ") ...\n";
            label_cascade_kernel(
                hedges,
                hedges_offsets,
                touching,
                touching_offsets,
                hedge_weights,
                partitions.data(),
                part_even_event_offsets.data(),
                part_odd_event_offsets.data(),
                even_ranks.data(),
                odd_ranks.data(),
                even_event_node.data(),
                odd_event_node.data(),
                num_nodes,
                batch_size,
                even_event_score.data()
            );

            // inclusive scan inside each key (= composite partition) on the even event scores => for each event we get the cumulative gain up to that point in the partition's move sequence
            // then extract the maximum idx (relative to the start of the overall array) for every partition's pair
            // => the trailing run of empty slots carries UINT32_MAX as key, so it lands past the real pairs and is dropped
            // NOTE: in CUDA the scan is an inclusive_scan_by_key, whose float sums associate differently (see README)
            par_fill<uint32_t>(apply_up_to.data(), batch_part_pairs, UINT32_MAX);
            par_for_each_run(
                batch_nodes,
                [&](dim_t a, dim_t b) { return even_event_part[a] == even_event_part[b]; },
                [&](dim_t begin, dim_t end) {
                    uint32_t argmax = (uint32_t)begin;
                    for (dim_t i = begin + 1; i < end; i++) {
                        even_event_score[i] += even_event_score[i - 1];
                        if (even_event_score[i] > even_event_score[argmax]) argmax = (uint32_t)i;
                    }
                    const uint32_t part = even_event_part[begin];
                    if (part < batch_part_pairs)
                        apply_up_to[part] = even_event_score[argmax] <= 0.0f ? UINT32_MAX : argmax;
                }
            );

            // retire the multi-starts with no strictly improving balanced prefix left
            LAUNCH(cfg) TID(tid) RUN << "labelprop activity kernel (threads=" << cfg.threads << ") ...\n";
            labelprop_activity_kernel(
                apply_up_to.data(),
                num_parts / 2,
                batch_size,
                active.data()
            );

            // apply pairs of improving moves
            // add together the gain of equi-ranked moves between bisected partitions as the gain of the pair to swap
            LAUNCH(cfg) TID(tid) RUN << "apply move events kernel (threads=" << cfg.threads << ") ...\n";
            apply_move_events_kernel(
                apply_up_to.data(),
                even_event_part.data(),
                even_event_node.data(),
                part_even_event_offsets.data(),
                part_odd_event_offsets.data(),
                odd_event_node.data(),
                num_nodes,
                batch_size,
                partitions.data()
            );

            // compute the new partitions split cost
            compute_partitions_cutnet(
                cfg,
                hedges,
                hedges_offsets,
                hedge_weights,
                partitions.data(),
                num_hedges,
                num_nodes,
                num_parts,
                batch_size,
                hedges_size,
                cutnet.data()
            );

            // track the best partitioning found so far at this level
            LAUNCH(cfg) TID(tid) RUN << "update best partitions kernel (threads=" << cfg.threads << ") ...\n";
            update_best_partitions_kernel(
                partitions.data(),
                cutnet.data(),
                last_best_cutnet.data(),
                num_nodes,
                batch_size,
                last_best_partitions.data()
            );

            // update the best split costs per partitions pair found so far at this level
            #pragma omp parallel for schedule(static) if(batch_part_pairs > PARALLEL_GRAIN)
            for (uint32_t pair = 0; pair < batch_part_pairs; pair++)
                last_best_cutnet[pair] = std::min(last_best_cutnet[pair], cutnet[pair]);

            uint32_t active_count = 0u;
            for (uint32_t start = 0; start < batch_size; start++) active_count += active[start];
            INFO(cfg) std::cout TID(tid) << "Label propagation on level " << level_idx << " repeat " << lp_repeat << " (multi-starts still improving=" << active_count << "/" << batch_size << ")\n";
            if (active_count == 0u) {
                INFO(cfg) std::cout TID(tid) << "Stopping label propagation on level " << level_idx << " repeat " << lp_repeat << " with no strictly improving balanced prefix left\n";
                break;
            }
        }

        // recover best partitions
        std::swap(partitions, last_best_partitions);
    }

    moves.release();
    scores.release();
    active.release();
    even_event_part.release();
    even_event_score.release();
    even_event_node.release();
    odd_event_part.release();
    odd_event_score.release();
    odd_event_node.release();
    even_ranks.release();
    odd_ranks.release();
    apply_up_to.release();
    last_best_partitions.release();

    // one final bisection to go down to 1-element partitions
    split_partitions_rand(
        cfg,
        partitions.data(),
        num_nodes,
        num_parts,
        batch_size,
        gens
    );
    num_parts *= 2;

    // sort a copy of partitions, carrying along node idxs
    buffer<uint32_t> order = par_sort_permutation(batch_nodes, bits_for((uint64_t)batch_size * num_parts), [&](dim_t i) { return (uint64_t)partitions[i]; }); // order[idx] -> batch-flat node currently in position idx
    buffer<uint32_t> ord_part(batch_nodes); // ord_part[idx] -> composite partition of node in order[idx]
    par_gather(order.data(), batch_nodes, partitions.data(), ord_part.data());

    // fuse back partitions while internally reversing them as needed to "trap" strong connections locally inside partition pairs
    while (num_parts > 2) { // go back up the bisection tree
        INFO(cfg) std::cout TID(tid) << "Tree reorientation level " << level_idx << " number of partitions=" << num_parts << "\n";
        level_idx--;

        buffer<float> sibling_score((dim_t)batch_size * num_parts); // sibling_score[p] -> total connection strength between composite partition p and the sibling subtree of floor(p/2)
        buffer<float> slot_score(batch_nodes); // slot_score[idx] -> strength contributed by the node in ordering slot idx, towards its partition

        // build offset indices over ord_part before the fold, so every partition's slots form one contiguous segment
        buffer<uint32_t> slot_part_offsets((dim_t)batch_size * num_parts + 1); // slot_part_offsets[p] -> first ordering slot of composite partition p
        par_lower_bound(ord_part.data(), batch_nodes, batch_size * num_parts + 1, slot_part_offsets.data());

        // compute connection strength of each partition with its parent's sibling subtree
        LAUNCH(cfg) TID(tid) RUN << "sibling tree connection strength kernel (threads=" << cfg.threads << ") ...\n";
        sibling_tree_connection_strength_kernel(
            hedges,
            hedges_offsets,
            touching,
            touching_offsets,
            hedge_weights,
            order.data(),
            ord_part.data(),
            partitions.data(),
            num_nodes,
            batch_size,
            slot_score.data()
        );

        // sum each partition's slot scores
        // NOTE: in CUDA this is a segmented reduce, whose float sums associate differently (see README)
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK) if(batch_size * num_parts > PARALLEL_GRAIN)
        for (uint32_t part = 0; part < batch_size * num_parts; part++) {
            float sum = 0.0f;
            for (uint32_t idx = slot_part_offsets[part]; idx < slot_part_offsets[part + 1]; idx++) sum += slot_score[idx];
            sibling_score[part] = sum;
        }

        // refold p*2 and p*2+1 back into p
        // => the composite id folds along with it, since "b*num_parts + p >> 1" is "b*(num_parts/2) + p/2" for even num_parts
        num_parts /= 2;
        #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
        for (uint32_t i = 0; i < batch_nodes; i++) {
            partitions[i] >>= 1;
            ord_part[i] >>= 1;
        }

        buffer<bool> reverse((dim_t)batch_size * num_parts); // reverse[p/2] -> true if the subtree of p/2 (well, ex-p/2, since we already folded it back in p) needs to have its leaves-order reversed
        LAUNCH(cfg) TID(tid) RUN << "flag reversals kernel (threads=" << cfg.threads << ") ...\n";
        flag_reversals_kernel(
            sibling_score.data(),
            num_parts,
            batch_size,
            reverse.data()
        );

        // build offset indices over ord_part
        // NOTE: searching one past the last composite partition yields the total node count, so no boundary needs writing by hand
        buffer<uint32_t> ord_part_offsets((dim_t)batch_size * num_parts + 1); // ord_part_offsets[p] -> first index of composite partition p in ord_part
        par_lower_bound(ord_part.data(), batch_nodes, batch_size * num_parts + 1, ord_part_offsets.data());

        // apply the reversal of leaves/nodes inside each flagged subtree
        LAUNCH(cfg) TID(tid) RUN << "apply reversals kernel (threads=" << cfg.threads << ") ...\n";
        apply_reversals_kernel(
            ord_part.data(),
            ord_part_offsets.data(),
            reverse.data(),
            batch_nodes,
            order.data()
        );
    }

    // write order_idx as the reverse map of order
    // => positions are made local to their own multi-start, so they index the shared 1D-to-(N)D map directly
    buffer<uint32_t> order_idx(batch_nodes); // order_idx[node] -> position in its multi-start's ordering for node
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < batch_nodes; i++)
        order_idx[order[i]] = i % num_nodes;

    // =============================
    // measure 1D order locality
    // metric: width spanned by each hedge (lowest pin idx - to - highest pin idx) times its weight
    // NOTE: only the first multi-start of the batch is measured, its order_idx slice already holds local positions
    LOG(cfg) {
        buffer<float> hedge_span(num_hedges); // hedge_span[hedge idx] -> max-pin-idx - min-pin-idx times the hedge's weight
        LAUNCH(cfg) TID(tid) RUN << "measure sequence locality kernel (threads=" << cfg.threads << ") ...\n";
        measure_sequence_locality_kernel(
            hedges,
            hedges_offsets,
            hedge_weights,
            order_idx.data(),
            num_hedges,
            hedge_span.data()
        );
        const float tot_span = par_reduce<float>(num_hedges, 0.0f, [&](dim_t i) { return hedge_span[i]; }, [](float a, float b) { return a + b; });
        std::cout TID(tid) << "Initial sequence (1D) weighted locality: " << std::fixed << std::setprecision(3) << tot_span << "\n";
    }
    // =============================

    return order_idx;
}
