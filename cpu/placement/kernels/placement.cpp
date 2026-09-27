#include <cmath>
#include <cfloat>
#include <cstdio>
#include <vector>
#include <cassert>
#include <algorithm>

#include "placement.hpp"
#include "utils_plc.hpp"
#include "utils.hpp"

// TOPOLOGY:
// explicitly instantiated topology globals
template<>
Lattice2D topo<Lattice2D>{};
template<>
Torus6D topo<Torus6D>{};
template<>
ArbitraryGraph topo<ArbitraryGraph>{};

// assign to each inverse placement slot the node occupying that place
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
template<Topology T>
void inverse_placement_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    uint32_t* __restrict__ inv_placement
) {
    // STYLE: one node per iteration!
    // NOTE: node idxs are batch-flat everywhere downstream, inv_placement stores them as such too
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < batch_nodes; node++) {
        const uint32_t my_start = node / num_nodes;
        inv_placement[my_start*volume + topo<T>.flattenedIdx(placement[node])] = node;
    }
}

// compute the forces pulling each node in the four cardinal directions
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
template<Topology T>
void forces_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const Coord_t<T>* __restrict__ placement,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    float* __restrict__ forces
) {
    /*
    * Idea:
    * - iterate, for each node, on its touching hedges, and on each node in each hedge, for each node updating all directional forces
    * - let the lanes of a node accumulate the forces, then reduce them
    */

    // STYLE: one node per iteration!
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t node = 0; node < batch_nodes; node++) {
        const uint32_t my_start = node / num_nodes; // multi-start owning this node
        if (!active[my_start]) continue; // this multi-start already converged
        const uint32_t nodes_base = my_start * num_nodes;
        const uint32_t my_node = node - nodes_base; // node idx local to its own multi-start

        const Coord_t<T> my_place = placement[node];
        // NOTE: the hypergraph is shared by every multi-start, index it with the local node idx
        const uint32_t* my_touching = touching + touching_offsets[my_node];
        const uint32_t touching_count = touching_offsets[my_node + 1] - touching_offsets[my_node];

        float base_potential_lanes[WARP_SIZE] = {}; // base_potential_lanes[lane] -> base potential accumulated by that lane
        float forces_lanes[T::neighborsCount()][WARP_SIZE] = {}; // forces_lanes[neigh_idx][lane] -> force towards the neigh_idx-th neighbor accumulated by that lane
        // only neighbors inside the topology get a force, the others are never read
        static_assert(T::neighborsCount() <= 32u, "the neighbors mask holds one bit per neighbor");
        Coord_t<T> neigh_places[T::neighborsCount()]; // neigh_places[neigh_idx] -> place of the neigh_idx-th neighbor
        uint32_t neigh_mask = 0u; // i-th bit set iff the i-th neighbor is inside the topology
        for (uint32_t neigh_idx = 0; neigh_idx < T::neighborsCount(); neigh_idx++) {
            neigh_places[neigh_idx] = topo<T>.neighbor(my_place, neigh_idx);
            neigh_mask |= (uint32_t)topo<T>.contains(neigh_places[neigh_idx]) << neigh_idx;
        }

        forEachTouchingPin(
            hedges, hedges_offsets, hedge_weights, my_touching, touching_count, nodes_base,
            [&](uint32_t lane, uint32_t pin, float my_hedge_weight) {
                if (pin == node) return;
                const Coord_t<T> pin_place = placement[pin];
                const uint32_t distance = topo<T>.distance(my_place, pin_place);
                // logic: base potential = how much distant I am be from my connectees
                //        my_force = how much distant I would be from my connectees if moved
                // NOTE: in CUDA nvcc fuses "acc += my_hedge_weight * distance" into a single FFMA, hence the explicit 'fmaf'
                base_potential_lanes[lane] = std::fmaf(my_hedge_weight, (float)distance, base_potential_lanes[lane]);
                for (uint32_t neigh_idx = 0; neigh_idx < T::neighborsCount(); neigh_idx++)
                    if ((neigh_mask >> neigh_idx) & 1u)
                        forces_lanes[neigh_idx][lane] = std::fmaf(my_hedge_weight, (float)std::max(topo<T>.distance(neigh_places[neigh_idx], pin_place), 1u), forces_lanes[neigh_idx][lane]);
            }
        );

        // reduce across the lanes
        const float my_base_potential = lanesReduceSumLN0<float>(base_potential_lanes);
        // logic: final force = reduction in distance if moved (higher is better)
        for (uint32_t neigh_idx = 0; neigh_idx < T::neighborsCount(); neigh_idx++)
            forces[node*T::neighborsCount() + neigh_idx] = my_base_potential - lanesReduceSumLN0<float>(forces_lanes[neigh_idx]);
    }
}

// compute the tension of each node with its neighbors, thereby proposing swapping pairs
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
template<Topology T>
void tensions_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ inv_placement,
    const float* __restrict__ forces,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    const uint8_t* __restrict__ active,
    const uint32_t candidates_count,
    uint32_t* __restrict__ pairs,
    uint32_t* __restrict__ scores
) {
    // STYLE: one node per iteration!
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < batch_nodes; node++) {
        const uint32_t my_start = node / num_nodes; // multi-start owning this node
        if (!active[my_start]) continue; // this multi-start already converged
        const uint32_t* my_inv_placement = inv_placement + my_start * volume;

        uint32_t my_pairs[T::neighborsCount()];
        float my_scores[T::neighborsCount()];

        const Coord_t<T> my_place = placement[node];

        // compute tensions (aka scores)
        for (uint32_t neigh_idx = 0; neigh_idx < T::neighborsCount(); neigh_idx++) {
            const Coord_t<T> neigh_place = topo<T>.neighbor(my_place, neigh_idx);
            if (topo<T>.contains(neigh_place)) { // valid neighbor
                const uint32_t neighbor = my_inv_placement[topo<T>.flattenedIdx(neigh_place)];
                if (neighbor != UINT32_MAX) { // there is a node placed on the neighboring spot
                    my_pairs[neigh_idx] = neighbor;
                    // tension = sum of opposing forces = my gain for moving towards the neighbor + the neighbor's gain for moving back towards me
                    const uint32_t back_idx = topo<T>.neighborIdx(neigh_place, my_place);
                    my_scores[neigh_idx] = forces[node*T::neighborsCount() + neigh_idx] + forces[neighbor*T::neighborsCount() + back_idx];
                } else { // empty neighboring spot
                    my_pairs[neigh_idx] = UINT32_MAX - neigh_idx - 1; // flag for "empty spot towards the neigh-th neighbor"
                    my_scores[neigh_idx] = forces[node*T::neighborsCount() + neigh_idx];
                }
            } else {
                my_pairs[neigh_idx] = UINT32_MAX;
                my_scores[neigh_idx] = 0.0f;
            }
        }

        // write pairs and scores, from highest to lowest score
        uint32_t* final_pairs = pairs + node * candidates_count;
        uint32_t* final_scores = scores + node * candidates_count;
        for (uint32_t i = 0; i < candidates_count; i++) {
            uint32_t max_pair = UINT32_MAX;
            float max_score = 0.0f;
            uint32_t max_idx = UINT32_MAX;
            for (uint32_t j = 0; j < T::neighborsCount(); j++) {
                if (my_scores[j] > max_score || (my_scores[j] == max_score && my_scores[j] > 0 && my_pairs[j] > max_pair)) {
                    max_pair = my_pairs[j];
                    max_score = my_scores[j];
                    max_idx = j;
                }
            }
            final_pairs[i] = max_pair;
            final_scores[i] = (uint32_t)std::min((uint64_t)(UINT32_MAX - T::neighborsCount() - 1), (uint64_t)(max_score*FORCE_FIXED_POINT_SCALE)); // go to fixed point to later use scores for book-keeping -> negative scores go to 0
            if (max_idx != UINT32_MAX) {
                my_pairs[max_idx] = UINT32_MAX;
                my_scores[max_idx] = 0.0f;
            }
        }
    }
}

// choose the pairs of at most 2 nodes, highest score first, that become candidate for swapping
// SEQUENTIAL COMPLEXITY: n*log n
// PARALLEL OVER: n (one tree level at a time)
template<Topology T>
void exclusive_swaps_kernel(
    const uint32_t* __restrict__ pairs, // pairs[idx * candidates_count + i] -> i-th target idx wants to be swapped with, a node or an empty cell
    const uint32_t* __restrict__ scores, // scores[idx * candidates_count + i] -> the strenght with which idx wants to be swapped with its i-th target
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    const uint32_t candidates_count,
    slot* __restrict__ swap_slots // swap_slots[idx] -> (score, id) of the best child of idx, ('UINT32_MAX - i', target id) once idx is locked
) {
    /*
    * Logic: a matching over the tree of pairs, visited one level at a time, going up and then down the tree!
    * => see the partitioning version for the idea!
    *
    * Tree of round i:
    * - every node not yet locked points to its i-th candidate, its target, with the candidate's score
    * - the target becomes the node's parent, unless the node is a root
    * - roots: nodes with no target, nodes whose target is locked, and nodes that lock in this round before the upward pass
    * - levels: a node's level is its height (the longest path down to a leaf), children always sit in lower levels than their parent
    *
    * Locks (before the upward pass):
    * - a node whose target is an empty cell locks itself (tension is symmetric, nobody else could claim it with a higher score)
    * - a mutually-pointing pair locks together, unless its other node is already locked
    *
    * Upward pass (levels from the leaves up):
    * - every node claims its parent's slot with (score, id) via an atomic max, the parent's slot keeps the best child (ties -> highest id)
    * - a node enters the next level when its last child claims it
    *
    * Downward pass (levels from the roots down):
    * - a node pairs with its parent iff the parent is not paired with its own parent, and the parent's slot holds the node
    * - lock pairs by setting both slots to point one to the other, with 'UINT32_MAX - i' as score
    *
    * Extras:
    * - a node not locked in a round resets its slot's score, but keeps its id, for the next round
    * - at the end, a locked slot's score becomes the pair's score: the maximum between the scores its two nodes gave the pair
    *
    * Every decision is taken once, with its inputs final: the outcome does not depend on the thread count nor on scheduling.
    * NOTE: in CUDA every node walks up the tree on its own, claiming slots as long as it wins them, then back down its path,
    *       pairing nodes; a walk climbing past a node repeats that node's own claims and decisions, hence the same outcome,
    *       without the walks' path buffers and chunks of multi-starts
    */

    const uint32_t batch_nodes = batch_size * num_nodes;
    const uint32_t lock_limit = UINT32_MAX - candidates_count; // slots with a score above this are locked

    std::vector<uint8_t> locked(batch_nodes); // locked[idx] -> true once idx got locked in a previous round
    std::vector<uint32_t> parents(batch_nodes); // parents[idx] -> idx's parent in the current round's tree
    std::vector<uint32_t> pending(batch_nodes); // pending[idx] -> children of idx that did not yet claim it during the upward pass
    std::vector<uint8_t> paired_up(batch_nodes); // paired_up[idx] -> true if idx pairs with its parent in the current round
    std::vector<uint32_t> order(batch_nodes); // order[...] -> the current round's nodes, level after level
    std::vector<uint32_t> level_bounds; // level_bounds[l] -> idx in "order" where the l-th level starts

    // initialize yourself to a one-node swap (no-swap)
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < batch_nodes; node++) {
        locked[node] = 0;
        if (active[node / num_nodes]) swap_slots[node] = pack_slot(0u, node);
    }

    // a node takes part in a round iff its multi-start did not converge and it is not yet locked
    auto walks = [&](const uint32_t node) { return active[node / num_nodes] && !locked[node]; };
    // an empty cell, rather than a node, as target
    auto is_empty = [](const uint32_t target) { return target >= UINT32_MAX - T::neighborsCount() && target < UINT32_MAX; };

    for (uint32_t i = 0; i < candidates_count; i++) {
        // lock nodes targeting an empty cell, and mutually-pointing pairs
        // STYLE: one node per iteration!
        #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
        for (uint32_t node = 0; node < batch_nodes; node++) {
            pending[node] = 0u;
            paired_up[node] = 0;
            if (!walks(node)) continue;
            const uint32_t target = pairs[node * candidates_count + i];
            if (is_empty(target) || target != UINT32_MAX && pairs[target * candidates_count + i] == node && !locked[target])
                swap_slots[node] = pack_slot(UINT32_MAX - i, target);
        }

        // build the tree
        // STYLE: one node per iteration!
        #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
        for (uint32_t node = 0; node < batch_nodes; node++) {
            parents[node] = UINT32_MAX;
            if (!walks(node) || slot_score(swap_slots[node]) > lock_limit) continue;
            const uint32_t target = pairs[node * candidates_count + i];
            if (target == UINT32_MAX) continue; // root: a node with no target
            if (locked[target] || slot_score(swap_slots[target]) > lock_limit) continue; // root: a node whose target is locked
            parents[node] = target;
            atomic_add<uint32_t>(&pending[target], 1u);
        }

        // first level: the leaves
        uint32_t tail = 0; // tail -> first free idx in "order"
        #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
        for (uint32_t node = 0; node < batch_nodes; node++)
            if (walks(node) && slot_score(swap_slots[node]) <= lock_limit && pending[node] == 0)
                order[atomic_add<uint32_t>(&tail, 1u)] = node;

        // go up the trees
        level_bounds.clear();
        level_bounds.push_back(0u);
        uint32_t level_begin = 0, level_end = tail;
        while (level_begin < level_end) {
            level_bounds.push_back(level_end);
            // STYLE: one node of the level per iteration!
            #pragma omp parallel for schedule(static) if(level_end - level_begin > PARALLEL_GRAIN)
            for (uint32_t idx = level_begin; idx < level_end; idx++) {
                const uint32_t current = order[idx];
                const uint32_t parent = parents[current];
                if (parent == UINT32_MAX) continue;
                // NOTE: if, by bad luck, the fixed-point score is the same, the id is used as a tie-breaker
                atomic_max<slot>(&swap_slots[parent], pack_slot(scores[current * candidates_count + i], current));
                if (atomic_sub<uint32_t>(&pending[parent], 1u) == 1u) // the last child to claim the parent moves it to the next level
                    order[atomic_add<uint32_t>(&tail, 1u)] = parent;
            }
            level_begin = level_end;
            level_end = tail;
        }
        // NOTE: nodes on a cycle longer than two never reach a level, and are left alone for this round

        // go down the trees
        for (int32_t level = (int32_t)level_bounds.size() - 2; level >= 0; level--) {
            const uint32_t begin = level_bounds[level], end = level_bounds[level + 1];
            // STYLE: one node of the level per iteration!
            #pragma omp parallel for schedule(static) if(end - begin > PARALLEL_GRAIN)
            for (uint32_t idx = begin; idx < end; idx++) {
                const uint32_t current = order[idx];
                const uint32_t parent = parents[current];
                paired_up[current] = parent != UINT32_MAX && !paired_up[parent] && slot_id(atomic_load<slot>(&swap_slots[parent])) == current;
                if (paired_up[current]) {
                    // mutually-pointing pair, the parent's slot already holds current's id
                    atomic_store<slot>(&swap_slots[current], pack_slot(UINT32_MAX - i, parent));
                    atomic_store<slot>(&swap_slots[parent], pack_slot(UINT32_MAX - i, current));
                }
            }
        }

        // anyone not yet locked, reset your slot's score to enable the next pairing round
        #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
        for (uint32_t node = 0; node < batch_nodes; node++) {
            if (!walks(node)) continue;
            if (slot_score(swap_slots[node]) > lock_limit) locked[node] = 1;
            else swap_slots[node] = pack_slot(0u, slot_id(swap_slots[node]));
        }
    }

    // write inside each locked slot the score of its pair
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < batch_nodes; node++) {
        if (!active[node / num_nodes]) continue; // this multi-start already converged
        const slot my_slot = swap_slots[node];
        if (slot_score(my_slot) <= lock_limit) continue;
        const uint32_t other_node = slot_id(my_slot);
        const uint32_t i_of_pair_formation = UINT32_MAX - slot_score(my_slot); // reconstruct the 'i' of the pair that caused the swap-pair
        uint32_t swap_score = scores[node * candidates_count + i_of_pair_formation];
        if (other_node < UINT32_MAX - T::neighborsCount())
            swap_score = std::max(swap_score, scores[other_node * candidates_count + i_of_pair_formation]);
        swap_slots[node] = pack_slot(swap_score, other_node);
    }
}

// from each node involved in a swap, produce a swap event
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
template<Topology T>
void swap_events_kernel(
    const slot* __restrict__ swap_slots,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    swap* __restrict__ ev_swaps,
    float* __restrict__ ev_scores
) {
    // STYLE: one node per iteration!
    // STYLE: events are not compacted, node "idx" owns event slot "idx"
    // => empty slots carry -FLT_MAX and sort behind every real event of their own multi-start
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t node = 0; node < batch_nodes; node++) {
        ev_swaps[node].lo = UINT32_MAX;
        ev_swaps[node].hi = UINT32_MAX;
        ev_scores[node] = -FLT_MAX;

        if (!active[node / num_nodes]) continue; // this multi-start already converged

        const slot my_swap_slot = swap_slots[node];
        // only the lower-id node spawns an event
        if (slot_id(my_swap_slot) <= node) continue;
        // filter-out no-pair nodes
        if (slot_id(my_swap_slot) == UINT32_MAX) continue;
        if (slot_id(my_swap_slot) < UINT32_MAX - T::neighborsCount()) {
            const slot target_swap_slot = swap_slots[slot_id(my_swap_slot)];
            if (slot_id(target_swap_slot) != node) continue;
            if (slot_score(my_swap_slot) != slot_score(target_swap_slot))
                printf("SCORE SYMMETRY MISMATCH: my id %u, my score %u, tg id %u, tg score %u\n", slot_id(my_swap_slot), slot_score(my_swap_slot), slot_id(target_swap_slot), slot_score(target_swap_slot));
            assert(slot_score(my_swap_slot) == slot_score(target_swap_slot));
        }

        // nodes in a pair, or paired with an empty cell, generate events
        ev_swaps[node].lo = node;
        ev_swaps[node].hi = slot_id(my_swap_slot); // this could be 'UINT32_MAX - 1..neighborsCount' for empty cells!
        ev_scores[node] = ((float)slot_score(my_swap_slot))/FORCE_FIXED_POINT_SCALE;
    }
}

// from each event (now sorted) give the swapped nodes their rank
// SEQUENTIAL COMPLEXITY: n (actually, this should be the # events)
// PARALLEL OVER: n
template<Topology T>
void scatter_ranks_kernel(
    const swap* __restrict__ ev_swaps,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    uint32_t* __restrict__ nodes_rank
) {
    // STYLE: one event per iteration!
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t event = 0; event < batch_nodes; event++) {
        const uint32_t my_start = event / num_nodes;
        if (!active[my_start]) continue; // this multi-start already converged

        const swap my_ev_swap = ev_swaps[event];
        if (my_ev_swap.lo == UINT32_MAX) continue; // empty slot, no event here

        // NOTE: ranks are local to a multi-start, so they can be compared against one another in 'cascade_kernel'
        const uint32_t my_rank = event - my_start * num_nodes;
        nodes_rank[my_ev_swap.lo] = my_rank;
        if (my_ev_swap.hi < UINT32_MAX - T::neighborsCount()) // be wary of empty cells
            nodes_rank[my_ev_swap.hi] = my_rank;
    }
}

// deterministically cancel duplicate claims on the same empty cell, keeping only the best-scoring (lowest event idx) one
// SEQUENTIAL COMPLEXITY: n*neighborsCount (actually, n -> # events)
// PARALLEL OVER: n
template<Topology T>
void resolve_empty_conflicts_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ inv_placement,
    const uint32_t* __restrict__ nodes_rank,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    const uint8_t* __restrict__ active,
    swap* __restrict__ ev_swaps
) {
    // NOTE: events are cancelled in place while others read them, hence the atomic accesses to their higher-id node
    const uint32_t batch_nodes = batch_size * num_nodes;

    // STYLE: one event per iteration!
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t event = 0; event < batch_nodes; event++) {
        const uint32_t my_start = event / num_nodes;
        if (!active[my_start]) continue; // this multi-start already converged
        const uint32_t nodes_base = my_start * num_nodes;
        const uint32_t my_rank = event - nodes_base; // rank local to the multi-start
        const uint32_t* my_inv_placement = inv_placement + my_start * volume;

        const swap my_ev_swap = ev_swaps[event];
        if (my_ev_swap.lo == UINT32_MAX) continue; // empty slot, no event here
        if (my_ev_swap.hi < UINT32_MAX - T::neighborsCount()) continue; // not an empty-cell move

        const uint32_t direction = UINT32_MAX - my_ev_swap.hi - 1;
        const Coord_t<T> target = topo<T>.neighbor(placement[my_ev_swap.lo], direction);

        // scan the target cell's neighbors: only they could possibly be contending for it too
        uint32_t best_event = my_rank;
        for (uint32_t neigh_idx = 0; neigh_idx < T::neighborsCount(); neigh_idx++) {
            const Coord_t<T> nb_coord = topo<T>.neighbor(target, neigh_idx);
            if (!topo<T>.contains(nb_coord)) continue;
            const uint32_t nb_node = my_inv_placement[topo<T>.flattenedIdx(nb_coord)];
            if (nb_node == UINT32_MAX) continue;
            const uint32_t nb_event = nodes_rank[nb_node];
            if (nb_event == UINT32_MAX || nb_event == my_rank) continue;
            const uint32_t nb_ev_swap_lo = ev_swaps[nodes_base + nb_event].lo;
            const uint32_t nb_ev_swap_hi = atomic_load<uint32_t>(&ev_swaps[nodes_base + nb_event].hi);
            if (nb_ev_swap_lo != nb_node || nb_ev_swap_hi < UINT32_MAX - T::neighborsCount()) continue; // not an empty move
            // NOTE: skip events already cancelled (their direction is lost)
            // => an event only loses its cell to a better-scoring one, the best one is never cancelled, and every loser still sees it
            if (nb_ev_swap_hi == UINT32_MAX) continue;
            const uint32_t nb_direction = UINT32_MAX - nb_ev_swap_hi - 1;
            if (topo<T>.neighbor(placement[nb_node], nb_direction) == target)
                best_event = std::min(best_event, nb_event);
        }

        if (best_event != my_rank)
            atomic_store<uint32_t>(&ev_swaps[event].hi, UINT32_MAX); // event cancelled, lost the cell to a better-scoring neighbor
    }
}

// place of a pin, once every event ranked before 'my_rank' is applied
template<Topology T>
inline Coord_t<T> placeInSequence(
    const Coord_t<T>* __restrict__ placement,
    const swap* __restrict__ ev_swaps,
    const uint32_t* __restrict__ nodes_rank,
    const uint32_t nodes_base,
    const uint32_t my_rank,
    const uint32_t pin
) {
    const uint32_t pin_event_idx = nodes_rank[pin];
    if (pin_event_idx >= my_rank) return placement[pin];
    const swap pin_ev_swaps = ev_swaps[nodes_base + pin_event_idx];
    if (pin == pin_ev_swaps.hi) return placement[pin_ev_swaps.lo];
    if (pin_ev_swaps.hi == UINT32_MAX) return placement[pin]; // cancelled empty move, pin never moved
    if (pin_ev_swaps.hi < UINT32_MAX - T::neighborsCount()) return placement[pin_ev_swaps.hi];
    const uint32_t pin_direction = UINT32_MAX - pin_ev_swaps.hi - 1; // swap neighbor idx
    return topo<T>.neighbor(placement[pin], pin_direction);
}

// update forces, and then tensions, pulling each swap-pair one towards the other
// => this is not done "in isolation" anymore, but considering the "sequence" of swaps by score
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
template<Topology T>
void cascade_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const Coord_t<T>* __restrict__ placement,
    const swap* __restrict__ ev_swaps,
    const uint32_t* __restrict__ nodes_rank,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint8_t* __restrict__ active,
    float* __restrict__ scores
) {
    /*
    * Idea:
    * Same as 'forces_kernel', but fused with the tension-computation logic and repeated for both nodes.
    */

    // STYLE: one event per iteration!
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t event = 0; event < batch_nodes; event++) {
        const uint32_t my_start = event / num_nodes; // multi-start owning this event slot
        if (!active[my_start]) continue; // this multi-start already converged
        const uint32_t nodes_base = my_start * num_nodes;
        const uint32_t my_rank = event - nodes_base; // rank local to the multi-start

        const swap my_ev_swaps = ev_swaps[event];

        // empty slot, leave its -FLT_MAX score in place so it stays behind every real event
        if (my_ev_swaps.lo == UINT32_MAX) continue;

        // cancelled by conflict resolution
        if (my_ev_swaps.hi == UINT32_MAX) {
            scores[event] = 0.0f;
            continue;
        }

        // gain of moving 'curr_node' one step towards 'direction', w.r.t. the sequence of events
        auto sequence_force = [&](const uint32_t curr_node, const uint32_t direction) -> float {
            const Coord_t<T> my_place = placement[curr_node];
            const Coord_t<T> neigh_place = topo<T>.neighbor(my_place, direction);
            const uint32_t curr_local = curr_node - nodes_base; // the hypergraph is shared by every multi-start
            const uint32_t* my_touching = touching + touching_offsets[curr_local];
            const uint32_t touching_count = touching_offsets[curr_local + 1] - touching_offsets[curr_local];

            float base_potential_lanes[WARP_SIZE] = {}; // base_potential_lanes[lane] -> base potential accumulated by that lane
            float force_lanes[WARP_SIZE] = {}; // force_lanes[lane] -> force accumulated by that lane

            forEachTouchingPin(
                hedges, hedges_offsets, hedge_weights, my_touching, touching_count, nodes_base,
                [&](uint32_t lane, uint32_t pin, float my_hedge_weight) {
                    if (pin == curr_node) return;
                    // reconstruct the pin's placement w.r.t. the sequence of events
                    const Coord_t<T> pin_place = placeInSequence<T>(placement, ev_swaps, nodes_rank, nodes_base, my_rank, pin);
                    // NOTE: in CUDA nvcc fuses "acc += my_hedge_weight * distance" into a single FFMA, hence the explicit 'fmaf'
                    base_potential_lanes[lane] = std::fmaf(my_hedge_weight, (float)topo<T>.distance(my_place, pin_place), base_potential_lanes[lane]);
                    force_lanes[lane] = std::fmaf(my_hedge_weight, (float)std::max(topo<T>.distance(neigh_place, pin_place), 1u), force_lanes[lane]);
                }
            );

            // reduce across the lanes
            return lanesReduceSumLN0<float>(base_potential_lanes) - lanesReduceSumLN0<float>(force_lanes);
        };

        // LOWER-ID NODE (always valid)
        uint32_t direction; // same as "neigh_idx" -> swap-pair neighbor index
        if (my_ev_swaps.hi >= UINT32_MAX - T::neighborsCount())
            direction = UINT32_MAX - my_ev_swaps.hi - 1;
        else
            direction = topo<T>.neighborIdx(placement[my_ev_swaps.lo], placement[my_ev_swaps.hi]);
        const float first_force = sequence_force(my_ev_swaps.lo, direction);

        // HIGHER-ID NODE (if valid)
        float second_force = 0.0f;
        if (my_ev_swaps.hi < UINT32_MAX - T::neighborsCount())
            second_force = sequence_force(my_ev_swaps.hi, topo<T>.neighborIdx(placement[my_ev_swaps.hi], placement[my_ev_swaps.lo]));

        scores[event] = first_force + second_force;
    }
}

// apply the swaps in each multi-start's improving event prefix
// SEQUENTIAL COMPLEXITY: n (actually, this is the # swaps to apply)
// PARALLEL OVER: n
template<Topology T>
void apply_swaps_kernel(
    const swap* __restrict__ ev_swaps,
    const uint32_t* __restrict__ num_good_swaps, // num_good_swaps[start] -> length of that multi-start's improving event prefix
    const uint32_t num_nodes,
    const uint32_t batch_size,
    const uint32_t volume,
    const uint8_t* __restrict__ active,
    Coord_t<T>* __restrict__ placement,
    uint32_t* __restrict__ inv_placement
) {
    /*
    * Notes:
    * - no need for atomics:
    *   - node<->node swaps are already mutually exclusive
    *   - conflict resolution made empty-cell moves mutually exclusive
    */

    // STYLE: one event per iteration!
    const uint32_t batch_nodes = batch_size * num_nodes;
    #pragma omp parallel for schedule(static) if(batch_nodes > PARALLEL_GRAIN)
    for (uint32_t event = 0; event < batch_nodes; event++) {
        const uint32_t my_start = event / num_nodes;
        if (!active[my_start]) continue; // this multi-start already converged
        if (event - my_start * num_nodes >= num_good_swaps[my_start]) continue; // past the improving prefix
        uint32_t* my_inv_placement = inv_placement + my_start * volume;

        const swap my_swap = ev_swaps[event];
        if (my_swap.hi == UINT32_MAX) continue; // cancelled by 'resolve_empty_conflicts_kernel'

        if (my_swap.hi < UINT32_MAX - T::neighborsCount()) {
            const Coord_t<T> plac_lo = placement[my_swap.lo];
            const Coord_t<T> plac_hi = placement[my_swap.hi];
            placement[my_swap.lo] = plac_hi;
            placement[my_swap.hi] = plac_lo;
            my_inv_placement[topo<T>.flattenedIdx(plac_lo)] = my_swap.hi;
            my_inv_placement[topo<T>.flattenedIdx(plac_hi)] = my_swap.lo;
        } else {
            const Coord_t<T> plac_lo = placement[my_swap.lo];
            const uint32_t direction = UINT32_MAX - my_swap.hi - 1;
            const Coord_t<T> plac_hi = topo<T>.neighbor(plac_lo, direction);
            placement[my_swap.lo] = plac_hi;
            my_inv_placement[topo<T>.flattenedIdx(plac_hi)] = my_swap.lo;
            my_inv_placement[topo<T>.flattenedIdx(plac_lo)] = UINT32_MAX;
        }
    }
}

// same as 'cub::BlockScan<float, PREFIX_GAIN_THREADS>(...).InclusiveSum(in, out, aggregate)', one entry per thread, returns the aggregate
// NOTE: in CUDA a block scan is "raking": the first warp's lanes each reduce a contiguous segment of the threads' entries, scan
//       the lanes' sums with shuffles (Kogge-Stone), then scan their segment again, seeded with the lanes before them; float
//       addition is not associative, so the same segments, shuffles and seeds are replayed here
inline float blockInclusiveSum(const float (&in)[PREFIX_GAIN_THREADS], float (&out)[PREFIX_GAIN_THREADS]) {
    constexpr uint32_t SEGMENT_LENGTH = PREFIX_GAIN_THREADS / WARP_SIZE; // entries per raking lane
    // upsweep: every lane reduces its segment
    float lanes[WARP_SIZE]; // lanes[lane] -> the lane's segment sum, then (after the scan) the lane's inclusive prefix
    for (uint32_t lane = 0; lane < WARP_SIZE; lane++) {
        float partial = in[lane * SEGMENT_LENGTH];
        for (uint32_t k = 1; k < SEGMENT_LENGTH; k++) partial = partial + in[lane * SEGMENT_LENGTH + k];
        lanes[lane] = partial;
    }
    // warp scan: shuffle up, doubling offsets
    for (uint32_t offset = 1; offset < WARP_SIZE; offset <<= 1)
        for (uint32_t lane = WARP_SIZE - 1; lane >= offset; lane--) // backwards, so that every lane reads its peer before the peer updates
            lanes[lane] = lanes[lane - offset] + lanes[lane];
    // downsweep: every lane scans its segment again, seeded with the inclusive prefix of the lane before it
    for (uint32_t lane = 0; lane < WARP_SIZE; lane++) {
        float running = in[lane * SEGMENT_LENGTH];
        if (lane != 0) running = lanes[lane - 1] + running;
        out[lane * SEGMENT_LENGTH] = running;
        for (uint32_t k = 1; k < SEGMENT_LENGTH; k++) {
            running = running + in[lane * SEGMENT_LENGTH + k];
            out[lane * SEGMENT_LENGTH + k] = running;
        }
    }
    return lanes[WARP_SIZE - 1];
}

// per multi-start, scan the in-sequence event gains and keep the highest-gain prefix, retiring converged multi-starts
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: batch_size
void prefix_gain_kernel(
    const float* __restrict__ ev_scores,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    uint32_t* __restrict__ num_good_swaps, // num_good_swaps[start] -> length of that multi-start's improving event prefix
    uint8_t* __restrict__ active
) {
    // STYLE: one multi-start per iteration!
    // NOTE: the scan is per multi-start, and not batch-wide, so that its tiling - and thus its float rounding -
    //       never depends on the batch size: a multi-start must reach the same result whatever it is batched with
    #pragma omp parallel for schedule(dynamic, 1)
    for (uint32_t my_start = 0; my_start < batch_size; my_start++) {
        if (!active[my_start]) continue; // this multi-start already converged

        const float* my_ev_scores = ev_scores + my_start * num_nodes;
        float carry = 0.0f; // gain accumulated over the tiles scanned so far
        int32_t best_key = -1; // best prefix end so far
        float best_value = -FLT_MAX; // gain of the best prefix so far

        for (uint32_t tile = 0; tile < num_nodes; tile += PREFIX_GAIN_THREADS) {
            float tile_scores[PREFIX_GAIN_THREADS], tile_prefix[PREFIX_GAIN_THREADS];
            for (uint32_t t = 0; t < PREFIX_GAIN_THREADS; t++) {
                const uint32_t my_rank = tile + t;
                const float my_score = my_rank < num_nodes ? my_ev_scores[my_rank] : -FLT_MAX;
                // empty slots must neither move the running prefix nor ever win the argmax
                tile_scores[t] = my_score != -FLT_MAX ? my_score : 0.0f;
            }
            const float tile_gain = blockInclusiveSum(tile_scores, tile_prefix);

            // argmax over the tile's events, the first one in case of ties
            int32_t tile_best_key = -1;
            float tile_best_value = -FLT_MAX;
            for (uint32_t t = 0; t < PREFIX_GAIN_THREADS; t++) {
                const uint32_t my_rank = tile + t;
                if (my_rank >= num_nodes || my_ev_scores[my_rank] == -FLT_MAX) continue;
                const float my_prefix = tile_prefix[t] + carry;
                if (tile_best_key < 0 || my_prefix > tile_best_value) {
                    tile_best_key = (int32_t)my_rank;
                    tile_best_value = my_prefix;
                }
            }

            carry += tile_gain;
            if (tile_best_key >= 0 && tile_best_value > best_value) {
                best_key = tile_best_key;
                best_value = tile_best_value;
            }
        }

        // retire this multi-start if no improving prefix is left
        if (best_key < 0 || best_value < FD_MIN_GAIN) {
            num_good_swaps[my_start] = 0u;
            active[my_start] = 0u;
        } else
            num_good_swaps[my_start] = (uint32_t)best_key + 1u;
    }
}

// compute the max src-dst manhattan distance per hedge
// SEQUENTIAL COMPLEXITY: e*d^2
// NOTE: the 'd^2' is only due to the serialization over sources => it is only 'd' when there is one source
// PARALLEL OVER: e
template<Topology T>
void max_src_dst_distance_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    float* __restrict__ result
) {
    /*
    * Idea:
    * - for each hedge, find the max src-dst manhattan distance (proxy for latency)
    * - scale it by the hedge's weight
    * - for each source, go over destinations and find the most distant one
    * - repeat for the next source, accumulate the total distance
    */

    // STYLE: one hedge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
        const uint32_t* srcs_start = hedges + hedges_offsets[hedge_idx];
        const uint32_t* dsts_start = srcs_start + srcs_count[hedge_idx];
        const uint32_t* dsts_end = hedges + hedges_offsets[hedge_idx + 1];

        uint32_t tot_distance = 0u;
        for (const uint32_t* src_ptr = srcs_start; src_ptr < dsts_start; src_ptr++) {
            const Coord_t<T> src_plc = placement[*src_ptr];
            uint32_t max_distance = 0u;
            for (const uint32_t* dst_ptr = dsts_start; dst_ptr < dsts_end; dst_ptr++)
                max_distance = std::max(max_distance, topo<T>.distance(src_plc, placement[*dst_ptr]));
            tot_distance += max_distance;
        }

        result[hedge_idx] = tot_distance * hedge_weights[hedge_idx];
    }
}

// same as 'max_src_dst_distance_kernel', but accumulates the manhattan distance over all src-dst per hedge
// SEQUENTIAL COMPLEXITY: e*d^2
// PARALLEL OVER: e
template<Topology T>
void tot_src_dst_distance_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    float* __restrict__ result
) {
    // STYLE: one hedge per iteration!
    const uint32_t batch_hedges = batch_size * num_hedges;
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t batch_hedge = 0; batch_hedge < batch_hedges; batch_hedge++) {
        // the hypergraph is shared by every multi-start, only the placement it is graded against differs
        const uint32_t my_start = batch_hedge / num_hedges;
        const uint32_t my_hedge = batch_hedge - my_start * num_hedges;
        const Coord_t<T>* my_placement = placement + my_start * num_nodes;

        const uint32_t* srcs_start = hedges + hedges_offsets[my_hedge];
        const uint32_t* dsts_start = srcs_start + srcs_count[my_hedge];
        const uint32_t* dsts_end = hedges + hedges_offsets[my_hedge + 1];

        uint32_t tot_distance = 0u;
        for (const uint32_t* src_ptr = srcs_start; src_ptr < dsts_start; src_ptr++) {
            const Coord_t<T> src_plc = my_placement[*src_ptr];
            for (const uint32_t* dst_ptr = dsts_start; dst_ptr < dsts_end; dst_ptr++)
                tot_distance += topo<T>.distance(src_plc, my_placement[*dst_ptr]);
        }

        result[batch_hedge] = tot_distance * hedge_weights[my_hedge];
    }
}

// compute the min spanning tree weight between pins per hedge
// SEQUENTIAL COMPLEXITY: e*d^2
// PARALLEL OVER: e
template<Topology T>
void min_spanning_tree_weight_kernel(
    const Coord_t<T>* __restrict__ placement,
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    const uint32_t num_nodes,
    const uint32_t batch_size,
    float* __restrict__ result
) {
    /*
    * Spanning tree algorithm (Prim-style):
    * - each hedge is a fully connected and weighted graph
    * - for each pin, track its min current distance from any pin already in the MST and a flag telling if the pin is itself in the MST already
    * - arbitrarily add the first pin to the MST
    *   - for consistency, always add the last pin
    * - pick the pin with smallest distance to the MST, and add it to it
    *   - every other pin updates its min distance from the MST based on the new pin (lower it iff closer to the new pin)
    * - repeat until no pin remains outside the MST
    * - no need to track which edges were used for the MST, we just need to accumulate the total weight used when adding pins to it
    * NOTE: in CUDA a warp's lanes hold up to REG_PINS_CAPACITY pins each in registers, here each thread has a scratch as large as needed
    */

    // STYLE: one hedge per iteration!
    const uint32_t batch_hedges = batch_size * num_hedges;
    #pragma omp parallel
    {
        std::vector<uint32_t> mst_distance; // mst_distance[pin idx] -> min distance of the pin from the MST, UINT32_MAX once in the MST
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t batch_hedge = 0; batch_hedge < batch_hedges; batch_hedge++) {
            // the hypergraph is shared by every multi-start, only the placement it is graded against differs
            const uint32_t my_start = batch_hedge / num_hedges;
            const uint32_t my_hedge = batch_hedge - my_start * num_hedges;
            const Coord_t<T>* my_placement = placement + my_start * num_nodes;

            const uint32_t* hedge_start = hedges + hedges_offsets[my_hedge];
            const uint32_t hedge_size = hedges_offsets[my_hedge + 1] - hedges_offsets[my_hedge] - 1; // pins other than the last one
            if (hedge_size + 1 <= 1) {
                result[batch_hedge] = 0.0f;
                continue;
            }

            // initialize distance to the first MST pin (last one)
            const Coord_t<T> first_mst_pin_plc = my_placement[hedge_start[hedge_size]];
            mst_distance.resize(hedge_size);
            for (uint32_t pin_idx = 0u; pin_idx < hedge_size; pin_idx++)
                mst_distance[pin_idx] = topo<T>.distance(first_mst_pin_plc, my_placement[hedge_start[pin_idx]]);

            uint32_t tot_span = 0u;
            for (uint32_t added = 0u; added < hedge_size; added++) {
                // argmin for next pin to add to the MST
                // NOTE: in CUDA ties go to the highest pin idx, here to the lowest, the MST's total weight is the same
                uint32_t min_pin = 0u;
                for (uint32_t pin_idx = 1u; pin_idx < hedge_size; pin_idx++)
                    if (mst_distance[pin_idx] < mst_distance[min_pin]) min_pin = pin_idx;
                tot_span += mst_distance[min_pin];
                mst_distance[min_pin] = UINT32_MAX;

                // update each pin's min distance from the MST
                const Coord_t<T> new_mst_pin_plc = my_placement[hedge_start[min_pin]];
                for (uint32_t pin_idx = 0u; pin_idx < hedge_size; pin_idx++)
                    if (mst_distance[pin_idx] != UINT32_MAX)
                        mst_distance[pin_idx] = std::min(mst_distance[pin_idx], topo<T>.distance(new_mst_pin_plc, my_placement[hedge_start[pin_idx]]));
            }

            result[batch_hedge] = tot_span * hedge_weights[my_hedge];
        }
    }
}

// TEMPLATE INSTANTIATIONS

#define INSTANTIATE_PLACEMENT_KERNELS(T) \
    template void inverse_placement_kernel<T>( \
        const Coord_t<T>*, uint32_t, uint32_t, uint32_t, uint32_t*); \
     \
    template void forces_kernel<T>( \
        const uint32_t*, const dim_t*, \
        const uint32_t*, const dim_t*, \
        const float*, const Coord_t<T>*, uint32_t, uint32_t, const uint8_t*, float*); \
     \
    template void tensions_kernel<T>( \
        const Coord_t<T>*, const uint32_t*, const float*, \
        uint32_t, uint32_t, uint32_t, const uint8_t*, uint32_t, uint32_t*, uint32_t*); \
     \
    template void exclusive_swaps_kernel<T>( \
        const uint32_t*, const uint32_t*, uint32_t, uint32_t, const uint8_t*, uint32_t, slot*); \
     \
    template void swap_events_kernel<T>( \
        const slot*, uint32_t, uint32_t, const uint8_t*, swap*, float*); \
     \
    template void scatter_ranks_kernel<T>( \
        const swap*, uint32_t, uint32_t, const uint8_t*, uint32_t*); \
     \
    template void resolve_empty_conflicts_kernel<T>( \
        const Coord_t<T>*, const uint32_t*, const uint32_t*, \
        uint32_t, uint32_t, uint32_t, const uint8_t*, swap*); \
     \
    template void cascade_kernel<T>( \
        const uint32_t*, const dim_t*, \
        const uint32_t*, const dim_t*, \
        const float*, const Coord_t<T>*, \
        const swap*, const uint32_t*, uint32_t, uint32_t, const uint8_t*, float*); \
     \
    template void apply_swaps_kernel<T>( \
        const swap*, const uint32_t*, uint32_t, uint32_t, uint32_t, const uint8_t*, \
        Coord_t<T>*, uint32_t*); \
     \
    template void max_src_dst_distance_kernel<T>( \
        const Coord_t<T>*, const uint32_t*, const dim_t*, \
        const uint32_t*, const float*, uint32_t, float*); \
     \
    template void tot_src_dst_distance_kernel<T>( \
        const Coord_t<T>*, const uint32_t*, const dim_t*, \
        const uint32_t*, const float*, uint32_t, uint32_t, uint32_t, float*); \
     \
    template void min_spanning_tree_weight_kernel<T>( \
        const Coord_t<T>*, const uint32_t*, const dim_t*, \
        const float*, uint32_t, uint32_t, uint32_t, float*);

INSTANTIATE_PLACEMENT_KERNELS(Lattice2D)
INSTANTIATE_PLACEMENT_KERNELS(Torus6D)
INSTANTIATE_PLACEMENT_KERNELS(ArbitraryGraph)

#undef INSTANTIATE_PLACEMENT_KERNELS
