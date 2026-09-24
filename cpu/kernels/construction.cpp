#include <vector>
#include <algorithm>
#include <functional>

#include "construction.hpp"
#include "utils.hpp"

// NOTE: in CUDA deduplication goes through a shared memory hash-set backed by a global memory one; here each thread keeps
//       a "seen" array, where seen[element] is the last item (node, group, hedge) that saw the element, so that the array
//       never needs clearing between items

// count how many hedges are inbound and how many touch each node
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void touching_count_kernel(
    const uint32_t* __restrict__ hedges, // stores srcs first, then dsts
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t num_hedges,
    dim_t* __restrict__ touching_offsets, // initialized at 0s
    uint32_t* __restrict__ inbound_count // initialized at 0s
) {
    // STYLE: one hedge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
        const uint32_t* hedge = hedges + hedges_offsets[hedge_idx];
        const uint32_t hedge_size = (uint32_t)(hedges_offsets[hedge_idx + 1] - hedges_offsets[hedge_idx]);
        const uint32_t hedge_srcs_count = srcs_count[hedge_idx];
        for (uint32_t pin_idx = 0; pin_idx < hedge_size; pin_idx++) {
            const uint32_t pin = hedge[pin_idx];
            atomic_add<dim_t>(&touching_offsets[pin + 1], 1ull); // leave the first entry to be 0 (offset of the first set)
            if (pin_idx >= hedge_srcs_count) // the pin is a dst
                atomic_add<uint32_t>(&inbound_count[pin], 1u);
        }
    }
}

// write inbound and outbound sets
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void touching_build_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_hedges,
    uint32_t* __restrict__ touching,
    uint32_t* __restrict__ inserted_inbound, // initialized at 0s
    uint32_t* __restrict__ inserted_outbound // initialized from inbound_count
) {
    // STYLE: one hedge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
        const uint32_t* hedge = hedges + hedges_offsets[hedge_idx];
        const uint32_t hedge_size = (uint32_t)(hedges_offsets[hedge_idx + 1] - hedges_offsets[hedge_idx]);
        const uint32_t hedge_srcs_count = srcs_count[hedge_idx];
        for (uint32_t pin_idx = 0; pin_idx < hedge_size; pin_idx++) {
            const uint32_t pin = hedge[pin_idx];
            uint32_t *pin_touching = touching + touching_offsets[pin];
            uint32_t insert_idx;
            if (pin_idx >= hedge_srcs_count) // the pin is a dst
                insert_idx = atomic_add<uint32_t>(&inserted_inbound[pin], 1u);
            else
                insert_idx = atomic_add<uint32_t>(&inserted_outbound[pin], 1u);
            pin_touching[insert_idx] = hedge_idx;
        }
    }
}

// sort each node's inbound and outbound sets by hedge id
// => inbound sets must be sorted, outbound ones are sorted too, to undo the order imposed by the atomics above
// SEQUENTIAL COMPLEXITY: n*h*log(h)
// PARALLEL OVER: n
void touching_sort_kernel(
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t num_nodes,
    uint32_t* __restrict__ touching
) {
    // STYLE: one node per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t node_id = 0; node_id < num_nodes; node_id++) {
        uint32_t *my_touching = touching + touching_offsets[node_id];
        uint32_t *not_my_touching = touching + touching_offsets[node_id + 1];
        std::sort(my_touching, my_touching + inbound_count[node_id]);
        std::sort(my_touching + inbound_count[node_id], not_my_touching);
    }
}

// compute the distinct neighbors of each node
// SEQUENTIAL COMPLEXITY: n*h*d
// PARALLEL OVER: n
void neighborhoods_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_nodes,
    csr_builder &neighbors // neighbors.open(node idx) -> where to append the node's distinct neighbors
) {
    #pragma omp parallel
    {
        std::vector<uint32_t> seen(num_nodes, UINT32_MAX); // seen[node] -> last node whose neighborhood included it

        // STYLE: one node per iteration!
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t node_id = 0; node_id < num_nodes; node_id++) {
            std::vector<uint32_t> &my_neighbors = neighbors.open(node_id);
            const uint32_t* my_touching = touching + touching_offsets[node_id];
            const uint32_t* not_my_touching = touching + touching_offsets[node_id + 1];
            for (const uint32_t* hedge_idx = my_touching; hedge_idx < not_my_touching; hedge_idx++) {
                const uint32_t* my_hedge = hedges + hedges_offsets[*hedge_idx];
                const uint32_t* not_my_hedge = hedges + hedges_offsets[*hedge_idx + 1];
                for (const uint32_t* pin = my_hedge; pin < not_my_hedge; pin++) {
                    const uint32_t neighbor = *pin;
                    if (neighbor == node_id || seen[neighbor] == node_id) continue;
                    seen[neighbor] = node_id;
                    my_neighbors.push_back(neighbor);
                }
            }
            neighbors.close(node_id);
        }
    }
}

// compute the distinct neighbors of each group from the neighbors of its nodes
// SEQUENTIAL COMPLEXITY: n*d*h (in reality there are <<d*h neighbors per node)
// PARALLEL OVER: n (groups)
void apply_coarsening_neighbors_kernel(
    const uint32_t* __restrict__ neighbors,
    const dim_t* __restrict__ neighbors_offsets,
    const uint32_t* __restrict__ groups, // groups[node id] -> node's group id
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    csr_builder &coarse_neighbors // coarse_neighbors.open(group idx) -> where to append the group's distinct neighbors
) {
    /*
    * Must ensure that:
    * - a node itself NEVER appears among its own neighbors
    */

    #pragma omp parallel
    {
        std::vector<uint32_t> seen(num_groups, UINT32_MAX); // seen[group] -> last group whose neighborhood included it

        // STYLE: one group (new node) per iteration!
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t group_id = 0; group_id < num_groups; group_id++) {
            std::vector<uint32_t> &my_neighbors = coarse_neighbors.open(group_id);
            // for every original node in the group, go over its neighbors
            for (dim_t u = ungroups_offsets[group_id]; u < ungroups_offsets[group_id + 1]; u++) {
                const uint32_t node = ungroups[u];
                for (dim_t i = neighbors_offsets[node]; i < neighbors_offsets[node + 1]; i++) {
                    const uint32_t new_neighbor = groups[neighbors[i]]; // translate to group id
                    if (new_neighbor == group_id || seen[new_neighbor] == group_id) continue;
                    seen[new_neighbor] = group_id;
                    my_neighbors.push_back(new_neighbor);
                }
            }
            coarse_neighbors.close(group_id);
        }
    }
}

// write the new distinct pins of hedges, sources first, then destinations (sorted in descending order)
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void apply_coarsening_hedges_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t* __restrict__ srcs_count,
    const uint32_t* __restrict__ groups, // groups[node idx] -> new group/node id
    const uint32_t num_hedges,
    const uint32_t num_groups,
    csr_builder &coarse_hedges, // coarse_hedges.open(hedge idx) -> where to append the hedge's distinct groups
    uint32_t* __restrict__ coarse_srcs_count
) {
    /*
    * Must ensure that:
    * - there are no self-cycles (remove the >>src<< to break them)
    * - the same node never appears twice in the same hedge
    *   => exploited by the coarsening of touching sets
    * - the source remains the first nodes in each coarse hedge
    */

    #pragma omp parallel
    {
        std::vector<uint32_t> seen(num_groups, UINT32_MAX); // seen[group] -> last hedge whose pins included it
        std::vector<uint32_t> new_dsts; // distinct destinations of the current hedge

        // STYLE: one hedge per iteration!
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
            const uint32_t *hedge_start = hedges + hedges_offsets[hedge_idx], *hedge_end = hedges + hedges_offsets[hedge_idx + 1];
            const uint32_t *hedge_srcs_end = hedge_start + srcs_count[hedge_idx];
            std::vector<uint32_t> &new_pins = coarse_hedges.open(hedge_idx);
            new_dsts.clear();
            // go over the hedge's destinations first, they take priority over sources
            for (const uint32_t* curr = hedge_srcs_end; curr < hedge_end; curr++) {
                const uint32_t pin = groups[*curr]; // read and map pin to its new id
                if (seen[pin] == hedge_idx) continue;
                seen[pin] = hedge_idx;
                new_dsts.push_back(pin);
            }
            // go over the hedge's sources
            uint32_t new_srcs_count = 0;
            for (const uint32_t* curr = hedge_start; curr < hedge_srcs_end; curr++) {
                const uint32_t pin = groups[*curr];
                if (seen[pin] == hedge_idx) continue;
                seen[pin] = hedge_idx;
                new_pins.push_back(pin);
                new_srcs_count++;
            }
            // sort destinations (descending)
            std::sort(new_dsts.begin(), new_dsts.end(), std::greater<uint32_t>());
            new_pins.insert(new_pins.end(), new_dsts.begin(), new_dsts.end());
            coarse_srcs_count[hedge_idx] = new_srcs_count;
            coarse_hedges.close(hedge_idx);
        }
    }
}

// write the distinct inbound (sorted), then outbound, touching hedges of each group
// SEQUENTIAL COMPLEXITY: n*h
// PARALLEL OVER: n (groups)
void apply_coarsening_touching_kernel(
    const uint32_t* __restrict__ touching,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t* __restrict__ inbound_count,
    const uint32_t* __restrict__ ungroups, // ungroups[ungroups_offsets[group id] + i] -> i-th original node in the group
    const dim_t* __restrict__ ungroups_offsets, // group (new node) id -> offset in ungroups where to find its original nodes
    const uint32_t num_groups,
    const uint32_t num_hedges,
    csr_builder &coarse_touching, // coarse_touching.open(group idx) -> where to append the group's distinct touching hedges
    uint32_t* __restrict__ coarse_inbound_count
) {
    /*
    * Must ensure that:
    * - inbound hedges are at the start of the set
    * - inbound hedges are sorted by id
    *
    * Important: here hedges that are both inbound and outbound are lost on one side, the outbound one specifically,
    *            just like the coarse hedges that break self-cycles by dropping the source.
    *
    * NOTE: in CUDA touching hedges are counted from the coarse hedges, here from the nodes in each group, since a group
    *       touches a coarse hedge iff one of its nodes touches the original hedge
    * NOTE: in CUDA outbound hedges end up in hash-set order, here in the order they are first seen; the order of touching
    *       hedges is the order of float sums over them during refinement, hence a possible source of divergence
    */

    #pragma omp parallel
    {
        std::vector<uint32_t> seen(num_hedges, UINT32_MAX); // seen[hedge] -> last group whose touching set included it

        // STYLE: one group (new node) per iteration!
        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for (uint32_t group_id = 0; group_id < num_groups; group_id++) {
            std::vector<uint32_t> &new_touching = coarse_touching.open(group_id);
            const dim_t inbound_start = new_touching.size();
            // for every original node in the group, go over its inbound set
            for (dim_t u = ungroups_offsets[group_id]; u < ungroups_offsets[group_id + 1]; u++) {
                const uint32_t node = ungroups[u];
                const uint32_t* my_inbound = touching + touching_offsets[node];
                for (uint32_t i = 0; i < inbound_count[node]; i++) {
                    const uint32_t hedge_idx = my_inbound[i];
                    if (seen[hedge_idx] == group_id) continue;
                    seen[hedge_idx] = group_id;
                    new_touching.push_back(hedge_idx);
                }
            }
            std::sort(new_touching.begin() + inbound_start, new_touching.end());
            coarse_inbound_count[group_id] = (uint32_t)(new_touching.size() - inbound_start);
            // for every original node in the group, go over its outbound set
            for (dim_t u = ungroups_offsets[group_id]; u < ungroups_offsets[group_id + 1]; u++) {
                const uint32_t node = ungroups[u];
                const uint32_t* my_outbound = touching + touching_offsets[node] + inbound_count[node];
                const uint32_t* not_my_outbound = touching + touching_offsets[node + 1];
                for (const uint32_t* hedge_idx = my_outbound; hedge_idx < not_my_outbound; hedge_idx++) {
                    if (seen[*hedge_idx] == group_id) continue; // dedupe among outbound, and against inbound hedges
                    seen[*hedge_idx] = group_id;
                    new_touching.push_back(*hedge_idx);
                }
            }
            coarse_touching.close(group_id);
        }
    }
}

// write to each node the partition of its group
// PARALLEL OVER: n
void apply_uncoarsening_partitions(
    const uint32_t* __restrict__ groups, // groups[node id] -> node's group
    const uint32_t* __restrict__ coarse_partitions, // coarse_partitions[group id] -> group's partition
    const uint32_t num_nodes,
    uint32_t* __restrict__ partitions // partitions[node id] -> group's partition
) {
    // STYLE: one node per iteration!
    #pragma omp parallel for schedule(static) if(num_nodes > PARALLEL_GRAIN)
    for (uint32_t node_id = 0; node_id < num_nodes; node_id++)
        partitions[node_id] = coarse_partitions[groups[node_id]];
}
