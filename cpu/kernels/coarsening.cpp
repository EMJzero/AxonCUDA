#include <vector>

#include "coarsening.hpp"
#include "constants.hpp"
#include "utils.hpp"

// find the best neighbor for each node to stay with (edge-coarsening)
// SEQUENTIAL COMPLEXITY: n*h*d + n*(# neighbors)*h
// PARALLEL OVER: n
void candidates_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ nodes_sizes,
    const uint32_t* __restrict__ nodes_pins,
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    uint32_t* __restrict__ pairs,
    uint32_t* __restrict__ scores
) {
    /*
    * Idea:
    * - one node per iteration
    * - histogram (one bin per neighbor), with a per-thread "bin_of" array mapping each neighbor to its bin
    * - iterate touching hyperedges once, each adds its (normalized) weight to the bins of its pins
    * - keep the best 'candidates_count' bins by (score, id)
    *
    * NOTE: in CUDA the histogram lives in shared memory, sorted by node id and binary searched, HIST_SIZE neighbors at a time;
    *       scores are fixed point, sums of integers, hence the order of updates is irrelevant and the candidates are the same
    *
    * This must give a symmetry invariant, if one node sees a candidate with score "s", then that candidate must also see this node as an option with score "s"!
    */

    #pragma omp parallel
    {
        std::vector<uint32_t> bin_of(num_nodes, UINT32_MAX); // bin_of[node] -> histogram bin of 'node' while it is a neighbor of the current node
        std::vector<uint32_t> histogram_node; // histogram_node[bin] -> neighbor in the bin
        std::vector<uint32_t> histogram_score; // histogram_score[bin] -> pairing score with the neighbor in the bin
        std::vector<uint32_t> histogram_inbound; // histogram_inbound[bin] -> inbound hedges of the neighbor in the bin, not already inbound to the current node

        // STYLE: one node per iteration!
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t node_id = 0; node_id < num_nodes; node_id++) {
            const uint32_t* my_neighbors = neighbors + neighbors_offsets[node_id];
            const uint32_t* not_my_neighbors = neighbors + neighbors_offsets[node_id + 1];

            const uint32_t* my_touching = touching + touching_offsets[node_id];
            const uint32_t my_touching_count = (uint32_t)(touching_offsets[node_id + 1] - touching_offsets[node_id]);
            const uint32_t my_inbound_count = inbound_count[node_id];

            const uint32_t my_size = nodes_sizes[node_id];
            const uint32_t my_pins = nodes_pins[node_id];

            // setup the histogram, skipping incompatible neighbors due to size and pins constraints
            histogram_node.clear();
            histogram_score.clear();
            histogram_inbound.clear();
            for (const uint32_t* nb = my_neighbors; nb < not_my_neighbors; nb++) {
                const uint32_t curr_neighbor = *nb;
                if (my_size + nodes_sizes[curr_neighbor] <= max_nodes_per_part && my_pins + nodes_pins[curr_neighbor] <= max_pins_per_part) {
                    bin_of[curr_neighbor] = (uint32_t)histogram_node.size();
                    histogram_node.push_back(curr_neighbor);
                    // add a little bit of symmetric deterministic noise
                    histogram_score.push_back(deterministic_noise<DETERMINISTIC_SCORE_NOISE>(curr_neighbor, node_id));
                    histogram_inbound.push_back(inbound_count[curr_neighbor]);
                }
            }

            // iterate over touching hyperedges
            for (uint32_t hedge_idx = 0u; hedge_idx < my_touching_count; hedge_idx++) {
                const uint32_t actual_hedge_idx = my_touching[hedge_idx];
                const dim_t my_hedge_offset = hedges_offsets[actual_hedge_idx];
                const dim_t my_hedge_size = hedges_offsets[actual_hedge_idx + 1] - my_hedge_offset;
                const uint32_t my_hedge_weight = (uint32_t)(hedge_weights[actual_hedge_idx]*FIXED_POINT_SCALE);
                const uint32_t my_hedge_src_count = srcs_count[actual_hedge_idx];
                const uint32_t* my_hedge = hedges + my_hedge_offset;
                for (uint32_t i = 0; i < my_hedge_size; i++) {
                    const uint32_t hist_idx = bin_of[my_hedge[i]];
                    if (hist_idx != UINT32_MAX) {
                        // normalize hedge weight over size
                        histogram_score[hist_idx] += my_hedge_weight / my_hedge_size;
                        if (i >= my_hedge_src_count && hedge_idx < my_inbound_count) // the pin is a destination and the hedge is an inbound-to-me one
                            histogram_inbound[hist_idx]--;
                    }
                }
            }

            // get the best 'candidates_count' candidates out of the histogram, skipping those that would lead to invalid clusters
            uint32_t best_score[MAX_CANDIDATES];
            uint32_t best_neighbor[MAX_CANDIDATES];
            for (uint32_t i = 0; i < candidates_count; i++) {
                best_score[i] = 0u;
                best_neighbor[i] = UINT32_MAX;
            }
            for (uint32_t nb = 0; nb < histogram_node.size(); nb++) {
                const uint32_t curr_neighbor = histogram_node[nb];
                bin_of[curr_neighbor] = UINT32_MAX; // reset the map for the next node
                if (histogram_inbound[nb] + my_inbound_count > max_inbound_per_part) continue;
                const uint32_t curr_score = histogram_score[nb];
                for (uint32_t i = 0; i < candidates_count; i++) {
                    // tie-breaker: higher id node wins; invariant: partial neighbors order
                    if (curr_score > best_score[i] || curr_score == best_score[i] && curr_neighbor > best_neighbor[i]) {
                        for (uint32_t j = candidates_count - 1; j > i; j--) {
                            best_score[j] = best_score[j - 1];
                            best_neighbor[j] = best_neighbor[j - 1];
                        }
                        best_score[i] = curr_score;
                        best_neighbor[i] = curr_neighbor;
                        break;
                    }
                }
            }

            for (uint32_t i = 0; i < candidates_count; i++) {
                pairs[node_id * candidates_count + i] = best_neighbor[i];
                scores[node_id * candidates_count + i] = best_score[i]; // stay fixed point for now!
            }
        }
    }
}

// create groups of at most two nodes, a matching over the tree of pairs, one round per candidate
// SEQUENTIAL COMPLEXITY: n*log n
// PARALLEL OVER: n (one tree level at a time)
void grouping_kernel(
    const uint32_t* __restrict__ pairs, // pairs[idx * candidates_count + i] is the i-th target idx wants to be grouped with (UINT32_MAX if undefined)
    const uint32_t* __restrict__ scores, // scores[idx * candidates_count + i] is the strenght with which idx wants to be grouped with its i-th target
    const uint32_t num_nodes,
    const uint32_t candidates_count,
    const bool exact, // if true, a node's gain accounts for its whole subtree (maximum weight matching), otherwise only for its score
    slot* __restrict__ group_slots, // group_slots[idx] -> (gain, id) of the best child of idx, (UINT32_MAX, child id) once idx is grouped
    uint32_t* __restrict__ groups // final group id of each node (non-zero based for now)
) {
    /*
    * Logic: dynamic programming over the tree of pairs, visited one level at a time, going up and then down the tree!
    *
    * => big HP: by construction, a node will always propose a pair with score equal or higher than that of pairs formulated
    *            by others towards him. This HP leaves no room for cycles (longer than two - pairs pointing one to the other)!
    *            The "pairs" build a tree with "roots" that are pairs pointing to each other!
    *
    * Tree of round i:
    * - every node not yet grouped points to its i-th candidate, its target, with the candidate's score
    * - the target becomes the node's parent, unless the node is a root
    * - roots: nodes with no target, nodes whose target is locked (grouped in a previous round), and the lowest id node in a
    *   mutual pair (the pair's other node becomes its child)
    * - levels: a node's level is its height (the longest path down to a leaf), children always sit in lower levels than their parent
    *
    * Upward pass (levels from the leaves up):
    * - every node computes its gain, then claims its parent's slot with (gain, id) via an atomic max, the parent's slot keeps
    *   the best child (ties -> highest id)
    * - a node enters the next level when its last child claims it, its slot is then final
    *
    * Downward pass (levels from the roots down):
    * - a node pairs with its parent iff the parent is not paired with its own parent, and the parent's slot holds the node
    * - lock pairs by setting both slot scores to the maximum, the group id is the child's
    *
    * Gain of a node (dynamic programming):
    * - with(v) = score(v) + sum of children's wout -> total score in the subtree of v, if v pairs with its parent
    * - wout(v) = sum of children's wout + max(0, best child gain) -> total score in the subtree of v, if v does not
    * - exact: gain(v) = with(v) - wout(v) = score(v) - max(0, best child gain)
    * - otherwise: gain(v) = score(v)
    * NOTE: in CUDA every node walks up the tree on its own, and its own walk claims its parent with (with, wout) = (score, 0),
    *       while walks from deeper in its subtree claim with - wout = score - (best child gain seen), since the sums of
    *       children's wout cancel out; the atomic max always keeps the claim of the node's own walk, "otherwise" is what CUDA computes
    *
    * Every decision is taken once, with its inputs final: the outcome does not depend on the thread count nor on scheduling.
    */

    std::vector<uint8_t> locked(num_nodes, 0); // locked[idx] -> true once idx got grouped in a previous round
    std::vector<uint32_t> parents(num_nodes); // parents[idx] -> idx's parent in the current round's tree
    std::vector<uint32_t> pending(num_nodes); // pending[idx] -> children of idx that did not yet claim it during the upward pass
    std::vector<uint8_t> paired_up(num_nodes); // paired_up[idx] -> true if idx pairs with its parent in the current round
    std::vector<uint32_t> order(num_nodes); // order[...] -> the current round's nodes, level after level
    std::vector<uint32_t> level_bounds; // level_bounds[l] -> idx in "order" where the l-th level starts

    // initialize everyone as a one-node group
    #pragma omp parallel for schedule(static) if(num_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < num_nodes; node++)
        group_slots[node] = pack_slot(0u, node);

    for (uint32_t i = 0; i < candidates_count; i++) {
        // build the tree
        #pragma omp parallel for schedule(static) if(num_nodes > PARALLEL_GRAIN)
        for (uint32_t node = 0; node < num_nodes; node++) {
            pending[node] = 0u;
            paired_up[node] = 0;
        }

        // STYLE: one node per iteration!
        #pragma omp parallel for schedule(static) if(num_nodes > PARALLEL_GRAIN)
        for (uint32_t node = 0; node < num_nodes; node++) {
            parents[node] = UINT32_MAX;
            if (locked[node]) continue;
            const uint32_t target = pairs[node * candidates_count + i];
            if (target == UINT32_MAX) continue; // root: a node with no target
            if (locked[target]) continue; // root: a node whose target is locked
            if (pairs[target * candidates_count + i] == node && node < target) continue; // root: the lowest id node in a mutual pair
            parents[node] = target;
            atomic_add<uint32_t>(&pending[target], 1u);
        }

        // first level: the leaves
        uint32_t tail = 0; // tail -> first free idx in "order"
        #pragma omp parallel for schedule(static) if(num_nodes > PARALLEL_GRAIN)
        for (uint32_t node = 0; node < num_nodes; node++)
            if (!locked[node] && pending[node] == 0)
                order[atomic_add<uint32_t>(&tail, 1u)] = node;

        // go up the trees
        level_bounds.clear();
        level_bounds.push_back(0u);
        uint32_t level_begin = 0, level_end = tail;
        while (level_begin < level_end) {
            level_bounds.push_back(level_end);
            // STYLE: one node of the level per iteration!
            #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK) if(level_end - level_begin > PARALLEL_GRAIN)
            for (uint32_t idx = level_begin; idx < level_end; idx++) {
                const uint32_t current = order[idx];
                const uint32_t parent = parents[current];
                if (parent == UINT32_MAX) continue;
                const uint32_t score = scores[current * candidates_count + i];
                // NOTE: all of current's children sit in lower levels, its slot is final
                const uint32_t best_child_gain = slot_score(atomic_load<slot>(&group_slots[current]));
                const uint32_t gain = exact ? (score > best_child_gain ? score - best_child_gain : 0u) : score;
                // NOTE: if, by bad luck, the fixed-point gain is the same, the id is used as a tie-breaker
                atomic_max<slot>(&group_slots[parent], pack_slot(gain, current));
                if (atomic_sub<uint32_t>(&pending[parent], 1u) == 1u) // the last child to claim the parent moves it to the next level
                    order[atomic_add<uint32_t>(&tail, 1u)] = parent;
            }
            level_begin = level_end;
            level_end = tail;
        }
        // NOTE: nodes on a cycle longer than two (never seen in practice, it violates the HP above) never reach a level, and
        //       are left alone for this round

        // go down the trees
        for (int32_t level = (int32_t)level_bounds.size() - 2; level >= 0; level--) {
            const uint32_t begin = level_bounds[level], end = level_bounds[level + 1];
            // STYLE: one node of the level per iteration!
            #pragma omp parallel for schedule(static) if(end - begin > PARALLEL_GRAIN)
            for (uint32_t idx = begin; idx < end; idx++) {
                const uint32_t current = order[idx];
                const uint32_t parent = parents[current];
                paired_up[current] = parent != UINT32_MAX && !paired_up[parent] && slot_id(atomic_load<slot>(&group_slots[parent])) == current;
                if (paired_up[current]) {
                    // NOTE: lock groups by setting scores to the maximum, the parent's slot already holds current's id
                    atomic_store<slot>(&group_slots[current], pack_slot(UINT32_MAX, current));
                    atomic_store<slot>(&group_slots[parent], pack_slot(UINT32_MAX, current));
                }
            }
        }

        // anyone not yet selected, reset your slot to enable the next pairing round
        #pragma omp parallel for schedule(static) if(num_nodes > PARALLEL_GRAIN)
        for (uint32_t node = 0; node < num_nodes; node++) {
            if (locked[node]) continue;
            if (slot_score(group_slots[node]) == UINT32_MAX) locked[node] = 1;
            else group_slots[node] = pack_slot(0u, node);
        }
    }

    // write inside "groups" the id in each node's slot, used to identify its group, eventually, zero-base those ids
    #pragma omp parallel for schedule(static) if(num_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < num_nodes; node++)
        groups[node] = slot_id(group_slots[node]);
}
