#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "../utils.cuh"
#include "utils.cuh"

namespace cg = cooperative_groups;

// DEVICE CONSTANTS:
__constant__ uint32_t max_width;
__constant__ uint32_t max_height;

// assign to each inverse placement slot the node occupying that place
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
__global__
void inverse_placement_kernel(
    const coords* __restrict__ placement,
    const uint32_t num_nodes,
    uint32_t* __restrict__ inv_placement
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    const coords my_place = placement[tid];
    inv_placement[my_place.y * max_width + my_place.x] = tid;
}

// compute the forces pulling each node in the four cardinal directions
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
// SHUFFLES OVER: d (hedges)
__global__
void forces_kernel(
    const uint32_t* __restrict__ hedges,
    const uint32_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const uint32_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const coords* __restrict__ placement,
    const uint32_t num_nodes,
    float* __restrict__ forces
) {
    // STYLE: one node per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_nodes) return;

    /*
    * Idea:
    * - iterate, for each node, on its touching hedges, and on each node in each hedge, for each node updating all 4 forces
    * - let a warp handle a node, reduce among its threads the four forces
    */
    
    const coords my_place = placement[warp_id];
    const uint32_t* my_touching = touching + touching_offsets[warp_id];
    const uint32_t* not_my_touching = touching + touching_offsets[warp_id + 1];
    //uint32_t touching_count = touching_offsets[warp_id + 1] - touching_offsets[warp_id];
    
    float my_base_potential = 0.0f;
    float my_forces[4];
    thr_init<float>(my_forces, 4, 0.0f);
    
    // scan touching hyperedges
    // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        // NOTE: this is not a warp-sync kernel, so using shuffles here to share data looses time, it's better to exploit caches with redundant reads!
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            if (my_hedge < not_my_hedge) {
                const uint32_t pin = *my_hedge;
                if (pin == warp_id) continue;
                const coords pin_place = placement[pin];
                const uint32_t distance = manhattan(my_place, pin_place);
                my_base_potential += my_hedge_weight * distance;
                // logic: base potential = how much distant I am be from my connectees
                //        my_force = how much distant I would be from my connectees if moved
                my_forces[LEFT] += my_hedge_weight * max(distance + (pin_place.x >= my_place.x) * 2 - 1, 1); // doing (cnd)*2-1 maps the condition's evaluation from 1/0 --to--> 1/-1
                my_forces[RIGHT] += my_hedge_weight * max(distance + (pin_place.x <= my_place.x) * 2 - 1, 1); // if pin is to my left, add 1
                my_forces[UP] += my_hedge_weight * max(distance + (pin_place.y >= my_place.y) * 2 - 1, 1); // if pin is below me, add 1
                my_forces[DOWN] += my_hedge_weight * max(distance + (pin_place.y <= my_place.y) * 2 - 1, 1); // if pin is above me, add 1
            }
        }
    }
    
    // reduce across the warp
    my_base_potential = warpReduceSumLN0<float>(my_base_potential);
    for (uint32_t f = 0; f < 4; f++)
        my_forces[f] = warpReduceSumLN0<float>(my_forces[f]);
    
    if (lane_id == 0) {
        // logic: final force = reduction in distance if moved (higher is better)
        forces[warp_id*4 + LEFT] = my_base_potential - my_forces[LEFT];
        forces[warp_id*4 + RIGHT] = my_base_potential - my_forces[RIGHT];
        forces[warp_id*4 + UP] = my_base_potential - my_forces[UP];
        forces[warp_id*4 + DOWN] = my_base_potential - my_forces[DOWN];
    }
}

// compute the tension of each node along the 4 cardinal directions, thereby proposing swapping pairs
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
__global__
void tensions_kernel(
    const coords* __restrict__ placement,
    const uint32_t* __restrict__ inv_placement,
    const float* __restrict__ forces,
    const uint32_t num_nodes,
    uint32_t* __restrict__ pairs,
    uint32_t* __restrict__ scores
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    uint32_t my_pairs[4];
    float my_scores[4];

    const coords my_place = placement[tid];
    
    if (my_place.x > 0) {
        const uint32_t neighbor = inv_placement[my_place.y * max_width + my_place.x - 1];
        if (neighbor != UINT32_MAX) {
            my_pairs[LEFT] = neighbor;
            my_scores[LEFT] = forces[tid*4 + LEFT] + forces[neighbor*4 + RIGHT]; // tension = sum of forces
        } else {
            my_pairs[LEFT] = UINT32_MAX - LEFT - 1; // flag for "empty spot"
            my_scores[LEFT] = forces[tid*4 + LEFT];
        }
    } else
        my_scores[LEFT] = 0.0f;

    if (my_place.x < max_width - 1) {
        const uint32_t neighbor = inv_placement[my_place.y * max_width + my_place.x + 1];
        if (neighbor != UINT32_MAX) {
            my_pairs[RIGHT] = neighbor;
            my_scores[RIGHT] = forces[tid*4 + RIGHT] + forces[neighbor*4 + LEFT]; // tension = sum of forces
        } else {
            my_pairs[RIGHT] = UINT32_MAX - RIGHT - 1; // flag for "empty spot"
            my_scores[RIGHT] = forces[tid*4 + RIGHT];
        }
    } else
        my_scores[RIGHT] = 0.0f;

    if (my_place.y > 0) {
        const uint32_t neighbor = inv_placement[(my_place.y - 1) * max_width + my_place.x];
        if (neighbor != UINT32_MAX) {
            my_pairs[UP] = neighbor;
            my_scores[UP] = forces[tid*4 + UP] + forces[neighbor*4 + DOWN]; // tension = sum of forces
        } else {
            my_pairs[UP] = UINT32_MAX - UP - 1; // flag for "empty spot"
            my_scores[UP] = forces[tid*4 + UP];
        }
    } else
        my_scores[UP] = 0.0f;

    if (my_place.y < max_height - 1) {
        const uint32_t neighbor = inv_placement[(my_place.y + 1) * max_width + my_place.x];
        if (neighbor != UINT32_MAX) {
            my_pairs[DOWN] = neighbor;
            my_scores[DOWN] = forces[tid*4 + DOWN] + forces[neighbor*4 + UP]; // tension = sum of forces
        } else {
            my_pairs[DOWN] = UINT32_MAX - DOWN - 1; // flag for "empty spot"
            my_scores[DOWN] = forces[tid*4 + DOWN];
        }
    } else
        my_scores[DOWN] = 0.0f;

    // write pairs and scores, from highest to lowest score
    uint32_t* final_pairs = pairs + tid * MAX_CANDIDATE_MOVES;
    uint32_t* final_scores = scores + tid * MAX_CANDIDATE_MOVES;
    for (uint32_t i = 0; i < MAX_CANDIDATE_MOVES; i++) {
        uint32_t max_pair = UINT32_MAX;
        float max_score = 0.0f;
        uint32_t max_idx = 0;
        for (uint32_t j = 0; j < 4; j++) {
            if (my_scores[j] > max_score) {
                max_pair = my_pairs[j];
                max_score = my_scores[j];
                max_idx = j;
            }
        }
        final_pairs[i] = max_pair;
        final_scores[i] = (uint32_t)(max_score*FORCE_FIXED_POINT_SCALE); // go to fixed point to later use scores for book-keeping -> negative scores go to 0
        my_pairs[max_idx] = UINT32_MAX;
        my_scores[max_idx] = 0.0f;
    }
}

// choose the pairs of at most 2 nodes, highest score first, that become candidate for swapping
// SEQUENTIAL COMPLEXITY: n*log n
// PARALLEL OVER: n
__global__
void exclusive_swaps_kernel(
    const uint32_t* __restrict__ pairs, // pairs[idx] is the partner idx wants to be swapped with
    const uint32_t* __restrict__ scores, // scores[idx] is the strenght with which idx wants to be swapped with pairs[idx]
    const uint32_t num_nodes,
    const uint32_t num_repeats,
    slot* __restrict__ swap_slots, // initialized with -1 on the id
    uint32_t* __restrict__ swap_flags // initialized to 0, set to 1 for the lower-id node of a swap-pair
) {
    // SETUP FOR GLOBAL SYNC
    cg::grid_group grid = cg::this_grid();

    // STYLE: 'num_repeats' node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tcount = gridDim.x * blockDim.x;

    /*
    * Logic: a walk up (and down) the tree for grouping!
    * => see the partitioning version for the idea!
    *
    * Extras:
    * - scores are set to 'UINT32_MAX - i' to lock nodes, where 'i' is the index of the candidate pair that caused the locking
    * - ultimately, slots of a pair must point one to the other (not share the same id)
    * - at the end, the lowest-id node of each pair sets a flag, that will be used to generate a swap-event from each pair
    *
    * Note: tension is SYMMETRIC!
    */

    int32_t path_length[MAX_REPEATS];
    uint32_t path[PATH_SIZE];
    const uint32_t actual_path_size = PATH_SIZE / num_repeats;
    uint32_t completed_repeats = 0u;

    for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
        const uint32_t curr_tid = tid + repeat * tcount;
        // initialize yourself to a one-node swap (no-swap), unless someone already claimed you
        if (curr_tid < num_nodes) atomicCAS(&reinterpret_cast<unsigned long long*>(swap_slots)[curr_tid], pack_slot(0u, UINT32_MAX), pack_slot(0u, curr_tid));
    }

    for (uint32_t i = 0; i < MAX_CANDIDATE_MOVES; i++) {
        // repeat for every node that you need to handle
        for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
            const uint32_t curr_tid = tid + repeat * tcount;

            // if you are not a valid node
            if (curr_tid >= num_nodes) break;
            // if you formed a pair, stop
            if (completed_repeats & (1u << repeat)) continue;

            uint32_t* curr_path = path + actual_path_size * repeat;
            int32_t curr_path_length = 0;

            uint32_t current = curr_tid; // current node in the tree of pairs
            uint32_t target = pairs[curr_tid * MAX_CANDIDATE_MOVES + i]; // target node of "current"
            uint32_t score = scores[curr_tid * MAX_CANDIDATE_MOVES + i]; // score with which "current" points to "target"

            // go up the tree
            bool outcome = false;
            if (target >= UINT32_MAX - 4 && target < UINT32_MAX) {
                // the target is an empty cell (counts as a root, lock immediately -> nobody else could think of claiming you if you chose this path, because tension is symmetric)
                set_slot(swap_slots, current, target, UINT32_MAX - i);
            } else if (target != UINT32_MAX) {
                outcome = true;
                uint32_t target_target = pairs[target * MAX_CANDIDATE_MOVES + i]; // target node of "target"
                while (current != target_target) {
                    // NOTE: if, by bad luck, the fixed-point score is the same, the id is used as a tie-breaker
                    outcome = atomic_max_on_slot(swap_slots, target, current, score);
                    if (!outcome) break;
                    assert(curr_path_length < actual_path_size);
                    curr_path[curr_path_length++] = target;
                    current = target;
                    target = target_target;
                    if (target >= UINT32_MAX - 4) break; // alternative root: a node with no target
                    target_target = pairs[target_target * MAX_CANDIDATE_MOVES + i]; // if this goes to -1, stop after the next iteration, as to still handle the current "target" that will be the "root"
                    score = scores[current * MAX_CANDIDATE_MOVES + i];
                }
            }
            // handle "root(s)" as a pair of nodes pointing to each other
            if (outcome && target < UINT32_MAX - 4 && swap_slots[target].score <= UINT32_MAX - MAX_CANDIDATE_MOVES) {
                set_slot(swap_slots, current, target, UINT32_MAX - i); // mutually-pointing pair
                set_slot(swap_slots, target, current, UINT32_MAX - i);
            }

            path_length[repeat] = curr_path_length;
        }

        // global synch
        grid.sync();

        // repeat for every node that you need to handle
        for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
            const uint32_t curr_tid = tid + repeat * tcount;

            // if you are not a valid node
            if (curr_tid >= num_nodes) break;
            // if you formed a pair, stop
            if (completed_repeats & (1u << repeat)) continue;

            uint32_t* curr_path = path + actual_path_size * repeat;
            int32_t curr_path_length = path_length[repeat];
            path_length[repeat] = 0;

            // go down the tree
            // it the root had no target, path[path_length - 1] is the root, if the root was a mutual-poiting pair, path[path_length - 1] is the first encountered node of the pair
            uint32_t current, target;
            for (curr_path_length = curr_path_length - 2; curr_path_length >= 0; curr_path_length--) {
                target = curr_path[curr_path_length + 1]; // this is "who current would have liked to be with", the node one step back up the ladder towards root
                current = curr_path[curr_path_length];
                if (swap_slots[target].id == current) {
                    swap_slots[current].id = target; // mutually-pointing pair
                    swap_slots[current].score = UINT32_MAX - i;
                    swap_slots[target].score = UINT32_MAX - i;
                    curr_path_length--;
                }
            }
            // the path did not include the "curr_tid" node, handle it here iff you did not happen to set-for-swap path[0] with path[1] during the last iteration above (leading to path_length == -2)
            if (curr_path_length == -1) {
                target = curr_path[0];
                current = curr_tid;
                if (swap_slots[target].id == current) {
                    swap_slots[current].id = target; // mutually-pointing pair
                    swap_slots[current].score = UINT32_MAX - i;
                    swap_slots[target].score = UINT32_MAX - i;
                }
            }
        }
        
        // global synch
        grid.sync();

        for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
            const uint32_t curr_tid = tid + repeat * tcount;
            if (curr_tid >= num_nodes) break;
            // if you formed a pair, stop
            if (completed_repeats & (1u << repeat)) continue;
            if (swap_slots[curr_tid].score > UINT32_MAX - MAX_CANDIDATE_MOVES) {
                completed_repeats |= (1u << repeat);
                continue;
            }
            swap_slots[curr_tid].score = 0;
        }

        // global synch
        grid.sync();
    }

    // write inside "swaps" the id of the node you are thus entitled to swap with
    for (uint32_t repeat = 0; repeat < num_repeats; repeat++) {
        const uint32_t curr_tid = tid + repeat * tcount;
        if (curr_tid >= num_nodes) break;
        // lowest-id node sets the create-event flag
        if (swap_slots[curr_tid].score > UINT32_MAX - MAX_CANDIDATE_MOVES) {
            const uint32_t other_tid = swap_slots[curr_tid].id;
            if (other_tid > curr_tid) swap_flags[curr_tid] = 1;
            // TODO: we could encode in the score "who locked who" in order not to need the maximum!
            // TODO: only the lower-id node actually needs to right score to generate the event!
            const uint32_t i_of_pair_formation = UINT32_MAX - swap_slots[curr_tid].score; // reconstruct the 'i' of the pair that caused the swap-pair
            uint32_t swap_score = scores[curr_tid * MAX_CANDIDATE_MOVES + i_of_pair_formation];
            if (other_tid < UINT32_MAX - 4)
                swap_score = max(swap_score, scores[other_tid * MAX_CANDIDATE_MOVES + i_of_pair_formation]);
            swap_slots[curr_tid].score = swap_score;
        }
    }
}

// from each node involved in a swap, produce a swap event
// SEQUENTIAL COMPLEXITY: n
// PARALLEL OVER: n
__global__
void swap_events_kernel(
    const slot* __restrict__ swap_slots,
    const uint32_t* __restrict__ swap_flags,
    const uint32_t num_nodes,
    swap* __restrict__ ev_swaps,
    float* __restrict__ ev_scores
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    const slot my_swap_slot = swap_slots[tid];
    // only the lower-id node spawns an event
    if (my_swap_slot.id <= tid) return;
    // filter-out no-pair nodes
    if (my_swap_slot.id == UINT32_MAX) return;
    if (my_swap_slot.id < UINT32_MAX - 4) {
        const slot target_swap_slot = swap_slots[my_swap_slot.id];
        if (target_swap_slot.id != tid) return;
        if (my_swap_slot.score != target_swap_slot.score)
            printf("SCORE SYMMETRY MISMATCH: my id %d, my score %d, tg id %d, tg score %d\n", my_swap_slot.id, my_swap_slot.score, target_swap_slot.id, target_swap_slot.score);
        assert(my_swap_slot.score == target_swap_slot.score);
    }

    // nodes in a pair, or paired with an empty cell, generate events
    const uint32_t ev_idx = swap_flags[tid];
    swap* my_ev_swap = ev_swaps + ev_idx;
    float* my_ev_score = ev_scores + ev_idx;
    (*my_ev_swap).lo = tid;
    (*my_ev_swap).hi = my_swap_slot.id; // this could be UINT32_MAX - 1..4 for empty cells!
    *my_ev_score = ((float)my_swap_slot.score)/FORCE_FIXED_POINT_SCALE;
}

// from each event (now sorted) give the swapped nodes their rank
// SEQUENTIAL COMPLEXITY: n (actually, this should be the # events)
// PARALLEL OVER: n
__global__
void scatter_ranks_kernel(
    const swap* __restrict__ ev_swaps,
    const uint32_t num_events,
    uint32_t* __restrict__ nodes_rank
) {
    // STYLE: one event per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    const swap my_ev_swap = ev_swaps[tid];
    nodes_rank[my_ev_swap.lo] = tid;
    if (my_ev_swap.hi < UINT32_MAX - 4) // be wary of empty cells
        nodes_rank[my_ev_swap.hi] = tid;
}

// update forces, and then tensions, pulling each swap-pair one towards the other
// => this is not done "in isolation" anymore, but considering the "sequence" of swaps by score
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
// SHUFFLES OVER: d (hedges)
__global__
void cascade_kernel(
    const uint32_t* __restrict__ hedges,
    const uint32_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const uint32_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const coords* __restrict__ placement,
    const swap* __restrict__ ev_swaps,
    const uint32_t* __restrict__ nodes_rank,
    const uint32_t num_events,
    float* __restrict__ scores
) {
    // STYLE: one event per warp!
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    // global across blocks - coincides with the node to handle
    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= num_events) return;

    /*
    * Idea:
    * Same as 'forces_kernel', but fused with the tension-computation logic and repeated for both nodes.
    */

    const swap my_ev_swaps = ev_swaps[warp_id];
    // NOTE: current rank == tid

    float first_force, second_force;


    // LOWER-ID NODE (always valid)

    uint32_t curr_node = my_ev_swaps.lo;
    coords my_place = placement[curr_node];

    uint32_t direction;
    if (my_ev_swaps.hi >= UINT32_MAX - 4) {
        direction = UINT32_MAX - my_ev_swaps.hi - 1;
    } else {
        const coords other_place = placement[my_ev_swaps.hi];
        if (my_place.x == other_place.x + 1) direction = LEFT;
        else if (my_place.x == other_place.x - 1) direction = RIGHT;
        else if (my_place.y == other_place.y + 1) direction = UP;
        else if (my_place.y == other_place.y - 1) direction = DOWN;
    }

    const uint32_t* my_touching = touching + touching_offsets[curr_node];
    const uint32_t* not_my_touching = touching + touching_offsets[curr_node + 1];

    float base_potential = 0.0f;
    float force = 0.0f;

    // scan touching hyperedges
    // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
    for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
        const uint32_t actual_hedge_idx = *hedge_idx;
        // NOTE: this is not a warp-sync kernel, so using shuffles here to share data looses time, it's better to exploit caches with redundant reads!
        const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
        my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
        const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
        const float my_hedge_weight = hedge_weights[actual_hedge_idx];
        for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
            if (my_hedge < not_my_hedge) {
                const uint32_t pin = *my_hedge;
                if (pin == curr_node) continue;
                coords pin_place;
                uint32_t pin_event_idx = nodes_rank[pin];
                // reconstruct the pin's placement w.r.t. the sequence of events
                if (pin_event_idx < warp_id) {
                    const swap pin_ev_swaps = ev_swaps[pin_event_idx];
                    if (pin == pin_ev_swaps.hi) pin_place = placement[pin_ev_swaps.lo];
                    else {
                        if (pin_ev_swaps.hi < UINT32_MAX - 4) pin_place = placement[pin_ev_swaps.hi];
                        else {
                            uint32_t pin_direction = UINT32_MAX - pin_ev_swaps.hi - 1;
                            pin_place = placement[pin];
                            if (pin_direction == LEFT) pin_place.x -= 1;
                            else if (pin_direction == RIGHT) pin_place.x += 1;
                            else if (pin_direction == UP) pin_place.y -= 1;
                            else if (pin_direction == DOWN) pin_place.y += 1;
                        }
                    }
                } else
                    pin_place = placement[pin];
                const uint32_t distance = manhattan(my_place, pin_place);
                base_potential += my_hedge_weight * distance;
                if (direction == LEFT) force += my_hedge_weight * max(distance + (pin_place.x >= my_place.x) * 2 - 1, 1);
                else if (direction == RIGHT) force += my_hedge_weight * max(distance + (pin_place.x <= my_place.x) * 2 - 1, 1);
                else if (direction == UP) force += my_hedge_weight * max(distance + (pin_place.y >= my_place.y) * 2 - 1, 1);
                else if (direction == DOWN) force += my_hedge_weight * max(distance + (pin_place.y <= my_place.y) * 2 - 1, 1);
            }
        }
    }

    // reduce across the warp
    base_potential = warpReduceSumLN0<float>(base_potential);
    force = warpReduceSumLN0<float>(force);
    first_force = base_potential - force;


    // HIGHER-ID NODE (if valid)

    if (my_ev_swaps.hi < UINT32_MAX - 4) {
        curr_node = my_ev_swaps.hi;
        my_place = placement[curr_node];
        
        if (direction == LEFT) direction = RIGHT;
        else if (direction == RIGHT) direction = LEFT;
        else if (direction == UP) direction = DOWN;
        else if (direction == DOWN) direction = UP;

        const uint32_t* my_touching = touching + touching_offsets[curr_node];
        const uint32_t* not_my_touching = touching + touching_offsets[curr_node + 1];
        
        base_potential = 0.0f;
        force = 0.0f;
        
        // scan touching hyperedges
        // TODO: could optimize by having threads that don't have anything left in the current hedge already wrap over to the next one
        for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
            const uint32_t actual_hedge_idx = *hedge_idx;
            // NOTE: this is not a warp-sync kernel, so using shuffles here to share data looses time, it's better to exploit caches with redundant reads!
            const uint32_t* my_hedge = hedges + hedges_offsets[actual_hedge_idx];
            my_hedge += lane_id; // each thread in the warp reads one every WARP_SIZE pins
            const uint32_t* not_my_hedge = hedges + hedges_offsets[actual_hedge_idx + 1];
            const float my_hedge_weight = hedge_weights[actual_hedge_idx];
            for (; my_hedge < not_my_hedge; my_hedge += WARP_SIZE) {
                if (my_hedge < not_my_hedge) {
                    const uint32_t pin = *my_hedge;
                    if (pin == curr_node) continue;
                    coords pin_place;
                    uint32_t pin_event_idx = nodes_rank[pin];
                    // reconstruct the pin's placement w.r.t. the sequence of events
                    if (pin_event_idx < warp_id) {
                        const swap pin_ev_swaps = ev_swaps[pin_event_idx];
                        if (pin == pin_ev_swaps.hi) pin_place = placement[pin_ev_swaps.lo];
                        else {
                            if (pin_ev_swaps.hi < UINT32_MAX - 4) pin_place = placement[pin_ev_swaps.hi];
                            else {
                                uint32_t pin_direction = UINT32_MAX - pin_ev_swaps.hi - 1;
                                pin_place = placement[pin];
                                if (pin_direction == LEFT) pin_place.x -= 1;
                                else if (pin_direction == RIGHT) pin_place.x += 1;
                                else if (pin_direction == UP) pin_place.y -= 1;
                                else if (pin_direction == DOWN) pin_place.y += 1;
                            }
                        }
                    } else
                        pin_place = placement[pin];
                    const uint32_t distance = manhattan(my_place, pin_place);
                    base_potential += my_hedge_weight * distance;
                    if (direction == LEFT) force += my_hedge_weight * max(distance + (pin_place.x >= my_place.x) * 2 - 1, 1);
                    else if (direction == RIGHT) force += my_hedge_weight * max(distance + (pin_place.x <= my_place.x) * 2 - 1, 1);
                    else if (direction == UP) force += my_hedge_weight * max(distance + (pin_place.y >= my_place.y) * 2 - 1, 1);
                    else if (direction == DOWN) force += my_hedge_weight * max(distance + (pin_place.y <= my_place.y) * 2 - 1, 1);
                }
            }
        }

        // reduce across the warp
        base_potential = warpReduceSumLN0<float>(base_potential);
        force = warpReduceSumLN0<float>(force);
        second_force = base_potential - force;
    } else
        second_force = 0.0f;
    
    if (lane_id == 0)
        scores[warp_id] = first_force + second_force;
}

// from each node involved in a swap, produce a swap event
// SEQUENTIAL COMPLEXITY: n (actually, this is the # swaps to apply)
// PARALLEL OVER: n
__global__
void apply_swaps_kernel(
    const swap* __restrict__ ev_swaps,
    const uint32_t num_good_swaps,
    coords* __restrict__ placement,
    uint32_t* __restrict__ inv_placement
) {
    // STYLE: one node per thread!
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_good_swaps) return;

    /*
    * Notes:
    * - no need for atomics, swaps are already mutually exclusive
    * - exceptional case, if two nodes aiming for an empty cell, wanted the same empty cell, give it to the lowest-id one!
    *   The other one can try again on the next round!
    *   => For now, this is not deterministic, not even lowest-id, literally first-come-first-served!
    * 
    * TODO: the solution in the "exceptional case" isn't great, because it screws up the "in sequence" gain calculation...
    *       Must find a way to preclude moves into the same cell earlier than here...
    * HOW: re-design the whole "pairing" and "slots" mechanism during the tension, walks, and events generation kernels:
    * - the tensions kernel proposes, as candidates, not the id of the node with which to swap, but the coordinates the node wants to occupy
    * - slots are one per cell, not one per node, everything else is the same, but now empty cells are contended too like every other
    *   - all the walks logic is the same, empty cells are simply passive actors, behaving just like roots that have no target of their own
    * - even the flags scan happens the same way, here we could keep one flag per node, only set by the lowest-id node, no issue
    * - events generation is again one thread per node, pass through the node's placement to get to its slot tho
    */

    const swap my_swap = ev_swaps[tid];
    if (my_swap.hi < UINT32_MAX - 4) {
        const coords plac_lo = placement[my_swap.lo];
        const coords plac_hi = placement[my_swap.hi];
        placement[my_swap.lo] = plac_hi;
        placement[my_swap.hi] = plac_lo;
        inv_placement[plac_lo.y * max_width + plac_lo.x] = my_swap.hi;
        inv_placement[plac_hi.y * max_width + plac_hi.x] = my_swap.lo;
    } else {
        const coords plac_lo = placement[my_swap.lo];
        coords plac_hi = plac_lo;
        uint32_t direction = UINT32_MAX - my_swap.hi - 1;
        if (direction == LEFT) plac_hi.x -= 1;
        else if (direction == RIGHT) plac_hi.x += 1;
        else if (direction == UP) plac_hi.y -= 1;
        else if (direction == DOWN) plac_hi.y += 1;
        const uint32_t prev = atomicCAS(&inv_placement[plac_hi.y * max_width + plac_hi.x], UINT32_MAX, my_swap.lo);
        if (prev == UINT32_MAX) {
            placement[my_swap.lo] = plac_hi;
            inv_placement[plac_lo.y * max_width + plac_lo.x] = UINT32_MAX;
        }
    }
}