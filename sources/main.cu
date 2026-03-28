#include <tuple>
#include <string>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <iostream>
#include <optional>
#include <filesystem>

#include <cuda_runtime.h>

#include "thruster.cuh"

#include <cub/cub.cuh>

#include "hgraph.hpp"
#include "constr.hpp"
#include "runconfig.hpp"

#include "utils.cuh"
#include "defines.cuh"
#include "constants.cuh"
#include "data_types.cuh"
#include "coarsening.cuh"
#include "construction.cuh"
#include "refinement.cuh"
#include "init_part.cuh"


using namespace hgraph;
using namespace constraints;


int main(int argc, char** argv) {
    if (argc == 1) {
        printHelp();
        return 0;
    }

    // parse CLI args
    runconfig cfg = parseArgs(argc, argv);

    // load hypergraph
    HyperGraph hg = loadHgraph(cfg);

    // setup constraints
    Constraints constr = setupConstr(cfg, hg);

    // print statistics
    std::cout << "Loaded hypergraph:\n";
    std::cout << "  Nodes:      " << hg.nodes() << "\n";
    std::cout << "  Hyperedges: " << hg.hedges().size() << "\n";
    std::cout << "  Total pins: " << hg.hedgesFlat().size() << "\n";
    std::cout << "  Total connections weight: " << std::fixed << std::setprecision(3) << hg.connectivity() << "\n";

    std::cout << "Using constraints \"" << constr.name() << "\":\n";
    std::cout << "  Nodes per partition:         " << constr.nodesPerPart() << "\n";
    std::cout << "  Inbound hedge per partition: " << constr.inboundPerPart() << "\n";
    std::cout << "  Maximum partitions:          " << constr.maxParts() << "\n";

    std::cout << "Using settings:\n";
    std::cout << "  Candidates count:            " << cfg.candidates_count << "\n";
    std::cout << "  Refinement repetitions:      " << cfg.refine_repeats << "\n";
    std::cout << "  Oversized multiplier factor: " << std::fixed << std::setprecision(2) << cfg.oversized_multiplier << "\n";

    std::cout << "CUDA device:\n";
    
    // get device properties
    int device_cnt;
    cudaGetDeviceCount(&device_cnt);
    std::cout << "  Found " << device_cnt << " devices: using device " << DEVICE_ID << "\n";
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, DEVICE_ID);
    std::cout << "  Device name: " << props.name << "\n";
    std::cout << "  Available VRAM: " << std::fixed << std::setprecision(1) << (float)(props.totalGlobalMem) / (1 << 30) << " GB\n";
    std::cout << "  Shared mem. per block: " << std::fixed << std::setprecision(1) << (float)(props.sharedMemPerBlock) / (1 << 10) << " KB\n";
    std::cout << "  Max. grid size: " << props.maxGridSize[0] << " x " << props.maxGridSize[1] << " x " << props.maxGridSize[2] << "\n";
    std::cout << "  Max. block size: " << props.maxThreadsDim[0] << " x " << props.maxThreadsDim[1] << " x " << props.maxThreadsDim[2] << "\n";

    std::cout << "Preparing hypergraph data...\n";

    //if (!cfg.device_touching_construction)
    //    hg.buildIncidenceSets();

    if (!constr.checkFit(hg, false, true))
        std::cerr << "WARNING, the hypergraph did not pass the fit check on the given constraints (NOTE: this test admits false negatives) !!\n";

    /*
    * Note: by design, only inbound hedges can be constrained (because their deduplication takes priority over outbound), therefore to support other constraints there are two options:
    * - to constrain outbound hedges, simply swap inbound and outbound hedges
    * - to constrain incident (touching) hedges, make them all inbound (no src)
    * 
    * Important:
    * - no cycles admitted
    * - during execution, hedges and incidence sets will diverge:
    *   - hedges remove duplicates (cycles) between sources and destinations, from the destinations (sources preserved)
    *   - incidence sets (touching) remove duplicates between inbound and outbound, from the outbound (inbound preserved -> for constraint checks)
    * 
    * Chore: for hedges and incidence sets alike, the half that "keeps the element" in case of a duplicate between srcs-dsts or in-out should always be the first half of the
    *        contigous segment. Unfortunately, right now hedges are the other way around...
    */

    uint32_t num_hedges = static_cast<uint32_t>(hg.hedges().size());
    std::vector<dim_t> hedges_offsets; // hedge idx -> hedge start index in the contiguous hedges array
    std::vector<uint32_t> srcs_count;
    hedges_offsets.reserve(num_hedges + 1);
    srcs_count.reserve(num_hedges);
    
    // prepare hedge offsets
    // HP: no duplicates per hedge, no self-cycles (keep the src only, arg=false -> still consider the hedge among the src's inbounds, arg=true -> update the inbound set to match)
    hg.deduplicateHyperedges(2, false); // remove the srcs
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedges_offsets.push_back(static_cast<dim_t>(hg.hedges()[i].offset()));
        srcs_count.push_back(hg.hedges()[i].src_count());
    }
    hedges_offsets.push_back(hg.hedgesFlat().size());
    
    std::vector<uint32_t> touching_hedges;
    std::vector<dim_t> touching_hedges_offsets;
    std::vector<uint32_t> inbound_count;
    touching_hedges.reserve(hg.hedgesFlat().size());
    touching_hedges_offsets.reserve(hg.nodes() + 1);
    inbound_count.reserve(hg.nodes());

    // prepare touching sets
    // HP: no duplicates in either set, eventually duplicates in outbound w.r.t. inbounds will also be lost,
    //     inbounds must come first and their part must be sorted by id (ascending)
    for (uint32_t n = 0; n < hg.nodes(); ++n) {
        auto curr_size = touching_hedges.size();
        touching_hedges_offsets.push_back(curr_size);
        // NOTE: must put in inbounds first!
        for (uint32_t h : hg.inboundSortedIds(n))
            touching_hedges.push_back(h);
        inbound_count.push_back(touching_hedges.size() - curr_size);
        for (uint32_t h : hg.outboundSortedIds(n))
            touching_hedges.push_back(h);
    }
    touching_hedges_offsets.push_back(touching_hedges.size());
    uint32_t touching_hedges_size = touching_hedges.size();
    
    // prepare hyperedge weights
    std::vector<float> hedge_weights(num_hedges);
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedge_weights[i] = hg.hedges()[i].weight();
    }
    
    // total number of distinct nodes (for output indexing)
    const uint32_t num_nodes = hg.nodes(); // nodes count used when allocating outputs

    // estimated max hedge and neighbors count
    dim_t max_hedge_size = std::transform_reduce(std::next(hedges_offsets.begin()), hedges_offsets.end(), hedges_offsets.begin(), dim_t{0}, [](dim_t a, dim_t b) { return std::max(a, b); }, [](dim_t next, dim_t curr) { return next - curr; });
    dim_t max_neighbors = hg.sampleMaxNeighborhoodSize(2400); // TODO: is 240 enough here?
    std::cout << "Max hedges estimate set to " << max_hedge_size << ", neighbors estimate set to " << max_neighbors << "\n";

    // constraints
    const uint32_t h_max_nodes_per_part = constr.nodesPerPart();
    const uint32_t h_max_inbound_per_part = constr.inboundPerPart();
    const uint32_t max_parts = constr.maxParts(); // not needed in kernels
    const uint32_t target_parts = min(max_parts, (num_nodes + h_max_nodes_per_part - 1) / h_max_nodes_per_part);
    assert(h_max_nodes_per_part <= INT32_MAX);
    assert(h_max_inbound_per_part <= INT32_MAX);
    assert(max_parts <= INT32_MAX);

    std::cout << "Starting timer...\n";
    auto time_start = std::chrono::high_resolution_clock::now();
    cudaEvent_t d_time_start, d_time_stop;
    CUDA_CHECK(cudaEventCreate(&d_time_start));
    CUDA_CHECK(cudaEventCreate(&d_time_stop));
    CUDA_CHECK(cudaEventRecord(d_time_start));

    std::cout << "Setting up GPU memory...\n";

    // ============================
    // === CUDA STUFF GOES HERE ===
    
    // device streams
    // TODO: use these!!
    cudaStream_t compute_stream = nullptr;
    cudaStream_t transfer_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
    int least_priority = 0;
    int greatest_priority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    // give higher priority access to memory bandwidth to the compute kernel
    CUDA_CHECK(cudaStreamCreateWithPriority(&compute_stream, cudaStreamNonBlocking, greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&transfer_stream, cudaStreamNonBlocking, least_priority));
    
    // device pointers
    uint32_t *d_hedges = nullptr; // hedges[hedges_offsets[hedge idx]] -> contigous array of pins of hedge (stored as src+destinations, with the srcs first)
    dim_t *d_hedges_offsets = nullptr; // hedges_offsets[hedge idx] -> hedge start idx in d_hedges
    uint32_t *d_srcs_count = nullptr; // srcs_count[hedge idx] -> number of sources of hedge idx
    uint32_t *d_neighbors = nullptr; // neighbors[neighbors_offsets[node idx]] -> contigous set/array of neighbors of node (its neighborhood)
    dim_t *d_neighbors_offsets = nullptr; // neighbors_offsets[node idx] -> neighbors set start idx in d_neighbors
    uint32_t *d_touching = nullptr; // touching[touching_offsets[node idx]] -> contigous inbound+outbout set/array (first inbound, then outbound) of node
    dim_t *d_touching_offsets = nullptr; // touching_offsets[node idx] -> touching set start idx in d_touching
    uint32_t *d_inbound_count = nullptr; // inbound_count[node idx] -> how many hedge of touching[node idx] are inbound (inbound hedges are before inbound_count[node idx], then outbound)
    float *d_hedge_weights = nullptr; // hedge_weights[hedge idx] -> weight
    uint32_t *d_pairs = nullptr; // partitions[node idx] -> best neighbor of node idx
    float *d_f_scores = nullptr; // connection strength for each pair, used during refinement
    uint32_t *d_u_scores = nullptr; // fixed point version of the above, used for the candidates and grouping kernels
    slot *d_slots = nullptr; // slot to finalize node pairs during grouping (true dtype: "slot")
    dp_score *d_dp_scores = nullptr; // dynamic programming score for each node in the tree assuming it connected (with) or not (w/out) to its target
    uint32_t *d_nodes_sizes = nullptr; // nodes_size[node idx] -> how many pins the node counts as towards the partition size limit
    uint32_t *d_partitions_sizes = nullptr; // partitions_sizes[idx] -> how many nodes (by total size) are in the partition
    uint32_t *d_pins_per_partitions = nullptr; // matrix<num_hedges x num_partitions>, pins_per_partitions[hedge idx * num_partitions + partition idx] -> count of pins of "hedge" in that "partition"
    uint32_t *d_partitions_inbound_sizes = nullptr; // partitions_inbound_sizes[partition] -> distinct inbound hedges count for "partition"

    // allocate device memory
    CUDA_CHECK(cudaMalloc(&d_hedges, hg.hedgesFlat().size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hedges_offsets, (num_hedges + 1) * sizeof(dim_t)));
    CUDA_CHECK(cudaMalloc(&d_srcs_count, num_hedges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_touching, touching_hedges.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_touching_offsets, (num_nodes + 1) * sizeof(dim_t)));
    CUDA_CHECK(cudaMalloc(&d_inbound_count, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hedge_weights, num_hedges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pairs, num_nodes * sizeof(uint32_t) * cfg.candidates_count));
    CUDA_CHECK(cudaMalloc(&d_f_scores, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_scores, num_nodes * sizeof(uint32_t) * cfg.candidates_count));
    CUDA_CHECK(cudaMalloc(&d_slots, num_nodes * sizeof(slot)));
    CUDA_CHECK(cudaMalloc(&d_dp_scores, num_nodes * sizeof(dp_score)));
    CUDA_CHECK(cudaMalloc(&d_nodes_sizes, num_nodes * sizeof(uint32_t)));

    // copy to device
    CUDA_CHECK(cudaMemcpy(d_hedges, hg.hedgesFlat().data(), hg.hedgesFlat().size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedges_offsets, hedges_offsets.data(), (num_hedges + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_srcs_count, srcs_count.data(), num_hedges * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching, touching_hedges.data(), touching_hedges.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching_offsets, touching_hedges_offsets.data(), (num_nodes + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hedge_weights, hedge_weights.data(), num_hedges * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inbound_count, inbound_count.data(), num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice));
    std::vector<dim_t>().swap(hedges_offsets);
    std::vector<uint32_t>().swap(touching_hedges);
    std::vector<uint32_t>().swap(srcs_count);
    std::vector<dim_t>().swap(touching_hedges_offsets);
    std::vector<uint32_t>().swap(inbound_count);

    // initialize
    thrust::device_ptr<uint32_t> t_nodes_sizes(d_nodes_sizes);
    thrust::fill(t_nodes_sizes, t_nodes_sizes + num_nodes, 1u); // each initial node counts as 1 (NOTE: can be tuned to give some nodes more "space")

    // copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(max_nodes_per_part, &h_max_nodes_per_part, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(max_inbound_per_part, &h_max_inbound_per_part, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    // wrap up memory duties with a sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // prepare neighborhoods
    std::tie(max_neighbors, d_neighbors, d_neighbors_offsets) = buildNeighbors(
        cfg,
        d_hedges,
        d_hedges_offsets,
        d_touching,
        d_touching_offsets,
        num_nodes,
        max_neighbors,
        d_neighbors,
        d_neighbors_offsets
    );


    // returns the number of partitions and the pointer to the final partitions device buffer
    std::function<std::tuple<uint32_t, uint32_t*>(const uint32_t, const uint32_t, uint32_t*&, dim_t*&, uint32_t*&, dim_t, uint32_t*&, dim_t*&, dim_t, uint32_t*&, uint32_t*&)> coarsen_refine_uncoarsen = [&](
        const uint32_t level_idx,
        const uint32_t curr_num_nodes,
        uint32_t*& d_hedges,
        dim_t*& d_hedges_offsets,
        uint32_t*& d_srcs_count,
        const dim_t hedges_size,
        uint32_t*& d_touching,
        dim_t*& d_touching_offsets,
        const dim_t touching_size,
        uint32_t*& d_inbound_count,
        uint32_t*& d_nodes_sizes
    ) { // this is a lambda
        std::cout << "Coarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

        /*
        * Flow:
        * 1) coarsen
        *   - propose valid candidate node pairs
        *   - group nodes w.r.t. strongest pairs
        *   => if groups are less than the threshold
        *       -> return them as the initial partitions (INCC - inbound constrained case)
        *       -> run the initial partitioning routine (KWAY - k-way balanced case)
        *   - coarsen all data structures
        * 2) recursive call to the next coarsening level
        *   - returns the coarse partitions
        * 3) uncoarsen
        *   - uncoarsen partitions
        *   - revert to using pre-coarsening data structures (free coarse ones)
        * 4) refinement
        *   - compute pins per partition
        *   - propose refinement moves in isolation and rank them
        *   - compute per-move gain as if applied in sequence
        *   - compute per-move validity via a prefix sum of the # of invalid partitions
        *     when applying the sequence of size, hedge, and inbound set events up to its rank
        *   - apply the highest-gain valid subsequence of moves
        *   => return final partitioning to the outer level
        */

        /*
        * Buffers allocated on (and local to) each level:
        * - d_groups
        * - d_ungroups, d_ungroups_offsets
        * - all event buffers for constraint checks
        * Buffers constructed anew before (and passed as args to) each level:
        * - d_hedges, d_hedges_offsets, d_srcs_count
        * - d_touching, d_touching_offsets, d_inbound_count
        * - d_nodes_sizes / d_groups_sizes
        * Buffers (constructed by and) returned from each level:
        * - d_partitions
        * Buffers updated (globally) in-place after each level:
        * - d_pairs
        * - d_u_scores, d_f_scores
        * - d_slots
        * - d_ranks
        * - d_neighbors, d_neighbors_offsets
        * - d_partitions_sizes
        * - d_pins_per_partitions
        * - d_partitions_inbound_sizes
        * Untouched buffers:
        * - d_hedge_weights
        *
        * TODO: could remove some (or even all) the synchronizes
        */

        /*
        * Constraint checks:
        * - coarsening only proposes pairs that, individually, would not violated constraints if grouped
        * - grouping more than two nodes checks if the larger group still fits constraints (TODO)
        * - refinement proposes moves that do not violated constraints if applied in isolation
        * - refinement selects the best subsequence of moves that ends on a valid state
        */

        // ======================================
        // k-way base case, build inital partitioning
        // => condition: passed the nodes threshold
        if (cfg.mode == Mode::KWAY && curr_num_nodes < KWAY_INIT_UPPER_THREASHOLD) {
            /*auto [d_init_partitions, d_init_partitions_sizes] = initial_partitioning(
                curr_num_nodes,
                num_hedges,
                d_hedges,
                d_hedges_offsets,
                d_hedge_weights,
                hedges_size,
                d_touching,
                d_touching_offsets,
                d_nodes_sizes,
                max_parts,
                h_max_nodes_per_part // -> rely on 'max_inbound_per_part' on the device for this
            );*/
            auto [d_init_partitions, d_init_partitions_sizes] = initial_partitioning_kahypar(
                curr_num_nodes,
                num_hedges,
                d_hedges,
                d_hedges_offsets,
                d_hedge_weights,
                d_touching_offsets,
                hedges_size,
                d_nodes_sizes,
                cfg.kway,
                cfg.epsi,
                h_max_nodes_per_part
            );
            d_partitions_sizes = d_init_partitions_sizes;
            CUDA_CHECK(cudaMalloc(&d_pins_per_partitions, num_hedges * max_parts * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_partitions_inbound_sizes, max_parts * sizeof(uint32_t))); // TODO: remove, not needed in KWAY mode

            return std::make_tuple(max_parts, d_init_partitions);
        }
        // ======================================

        // each node picks its candidate(s)
        candidatesProposal(
            cfg,
            d_hedges,
            d_hedges_offsets,
            d_srcs_count,
            d_neighbors,
            d_neighbors_offsets,
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            d_hedge_weights,
            d_nodes_sizes,
            curr_num_nodes,
            d_pairs,
            d_u_scores
        );

        // =============================
        // print some temporary results
        #if VERBOSE
        logCandidates(
            cfg,
            d_pairs,
            d_u_scores,
            curr_num_nodes
        );
        #endif
        // =============================

        // matching over the candidates graph
        uint32_t new_num_nodes; // new number of nodes after this coarsening round
        uint32_t *d_groups = nullptr; // groups[node idx] -> node's group id (zero-based)
        uint32_t *d_groups_sizes = nullptr; // group_sizes[group id] = sum of sizes of all nodes in that group
        uint32_t *d_ungroups = nullptr; // ungroups[ungroups_offsets[group id] + i] -> the group's i-th node (its original idx)
        dim_t *d_ungroups_offsets = nullptr; // ungroups_offsets[node idx] -> node's group id (zero-based)
        // TODO: make groupNodes only build d_ungroups while sorting node ids by their group id, build the offsets array later, as it is only needed if you do not reach a base case
        std::tie(new_num_nodes, d_groups, d_groups_sizes, d_ungroups, d_ungroups_offsets) = groupNodes(
            cfg,
            props,
            d_inbound_count,
            d_pairs,
            d_u_scores,
            d_nodes_sizes,
            curr_num_nodes,
            h_max_nodes_per_part,
            h_max_inbound_per_part,
            d_slots,
            d_dp_scores
        );

        // ======================================
        // k-way base case, build inital partitioning
        // => condition: too little shrinking to justify further coarsening
        if (cfg.mode == Mode::KWAY && ((float)new_num_nodes / (float)curr_num_nodes > KWAY_INIT_SHRINK_RATIO_LIMIT || new_num_nodes < KWAY_INIT_LOWER_THREASHOLD)) {
            /*auto [d_init_partitions, d_init_partitions_sizes] = initial_partitioning(
                curr_num_nodes,
                num_hedges,
                d_hedges,
                d_hedges_offsets,
                d_hedge_weights,
                hedges_size,
                d_touching,
                d_touching_offsets,
                d_nodes_sizes,
                max_parts,
                h_max_nodes_per_part // -> rely on 'max_inbound_per_part' on the device for this
            );*/
            auto [d_init_partitions, d_init_partitions_sizes] = initial_partitioning_kahypar(
                curr_num_nodes,
                num_hedges,
                d_hedges,
                d_hedges_offsets,
                d_hedge_weights,
                d_touching_offsets,
                hedges_size,
                d_nodes_sizes,
                cfg.kway,
                cfg.epsi,
                h_max_nodes_per_part
            );
            d_partitions_sizes = d_init_partitions_sizes;
            CUDA_CHECK(cudaMalloc(&d_pins_per_partitions, num_hedges * max_parts * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_partitions_inbound_sizes, max_parts * sizeof(uint32_t))); // TODO: remove, not needed in KWAY mode
            CUDA_CHECK(cudaFree(d_groups));
            CUDA_CHECK(cudaFree(d_ungroups));
            CUDA_CHECK(cudaFree(d_ungroups_offsets));
            CUDA_CHECK(cudaFree(d_groups_sizes));

            return std::make_tuple(max_parts, d_init_partitions);
        }
        // ======================================

        // ======================================
        // base case, return inital partitioning
        // TODO: could increase the threshold and instead of "become the partitions" run a host-side robust partitioning algorithm
        //       => what this does now is equivalent to using the coarsening algorithm also as the algorithm to perform the initial partitioning
        // NOTE: with the current setup, we stop as soon as we clear "max_parts" and let refinement eventually empty some if they are too many
        //       => and alternative solution could be to always wait for "new_num_nodes == curr_num_nodes" and then enforce "max_parts" to spot failures
        if (new_num_nodes <= target_parts || new_num_nodes == curr_num_nodes) {
            // HERE we repurpose the coarsening routine as the routine for initial partitions:
            // - num_partitions = new_num_nodes
            // - partitions = groups

            // NOTE: d_partitions eventually will coincide with the innermost group each node was part of + refinement moves
            //       => the innermost nodes (groups) count is also the number of partitions

            // NOTE: just like groups, partitions need to ordered, as they be used as indices; however, partitions are few, and if one becomes
            //       empty we can just discard its index and leave a few empty spots in the data structures, it's cheaper to compress at the end

            // neighbors are no longer needed after coarsening is done
            CUDA_CHECK(cudaFree(d_neighbors));
            CUDA_CHECK(cudaFree(d_neighbors_offsets));

            // prepare initial partition sizes
            // NOTE: current groups become the partitions, and so group sizes become partition sizes
            d_partitions_sizes = d_groups_sizes;

            // NOTE: the inbound counters per partition are just the transposed of pins per partition! No need to compute them separately!
            CUDA_CHECK(cudaMalloc(&d_pins_per_partitions, num_hedges * new_num_nodes * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_partitions_inbound_sizes, new_num_nodes * sizeof(uint32_t)));

            // base case, reached the target number of partitions
            if (new_num_nodes <= target_parts) {
                std::cout << "Minimal initial partitioning built at level " << level_idx << ", remaining nodes=" << curr_num_nodes << ", number of partitions=" << new_num_nodes << "\n";
            } else if (new_num_nodes <= max_parts) {
                std::cout << "Initial partitioning built at level " << level_idx << ", remaining nodes=" << curr_num_nodes << ", number of partitions=" << new_num_nodes << "\n";
                std::cerr << "WARNING: the partitioning is valid, but didn't reach the minimal number of partitions (" << target_parts << ")...\n";
            } else { // base case, failure to coarsen further
                std::cerr << "FAILED TO COARSEN FURTHER at level " << level_idx << ", remaining nodes=" << curr_num_nodes << " number of partitions=" << new_num_nodes << " max allowed partitions=" << max_parts << "\n";
                std::cerr << "WARNING: falling back to returning current groups as individual partitions...\n";
            }

            CUDA_CHECK(cudaFree(d_ungroups));
            CUDA_CHECK(cudaFree(d_ungroups_offsets));

            return std::make_tuple(new_num_nodes, d_groups);
        }
        // ======================================

        // =============================
        // print some temporary results
        #if VERBOSE
        logGroups(
            cfg,
            d_pairs,
            d_groups,
            d_groups_sizes,
            curr_num_nodes,
            new_num_nodes,
            h_max_nodes_per_part
        );
        #endif
        // =============================

        // update the maximum hedges and neighbors estimate by scaling it by new_num_nodes/curr_num_nodes
        float scale = (float)new_num_nodes / curr_num_nodes;
        max_hedge_size = std::ceil(max_hedge_size * scale);
        max_neighbors = std::ceil(max_neighbors * scale);
        std::cout << "Max hedges estimate updated to " << max_hedge_size << ", neighbors estimate updated to " << max_neighbors << "\n";

        // prepare coarse neighbors buffers
        // NOTE: overwrites previous neighbors
        std::tie(max_neighbors, d_neighbors, d_neighbors_offsets) = coarsenNeighbors(
            cfg,
            d_groups,
            d_ungroups,
            d_ungroups_offsets,
            curr_num_nodes,
            new_num_nodes,
            max_neighbors,
            d_neighbors,
            d_neighbors_offsets
        );

        // prepare coarse hedges buffers
        dim_t new_hedges_size;
        uint32_t *d_coarse_hedges = nullptr;
        dim_t *d_coarse_hedges_offsets = nullptr;
        uint32_t* d_coarse_srcs_count = nullptr;
        std::tie(new_hedges_size, max_hedge_size, d_coarse_hedges, d_coarse_hedges_offsets, d_coarse_srcs_count) = coarsenHedges(
            cfg,
            d_hedges,
            d_hedges_offsets,
            d_srcs_count,
            d_groups,
            num_hedges,
            max_hedge_size
        );

        // prepare coarse touching buffers
        dim_t new_touching_size;
        uint32_t *d_coarse_touching = nullptr;
        dim_t *d_coarse_touching_offsets = nullptr;
        uint32_t *d_coarse_inbound_count = nullptr;
        std::tie(new_touching_size, d_coarse_touching, d_coarse_touching_offsets, d_coarse_inbound_count) = coarsenTouching(
            cfg,
            d_coarse_hedges,
            d_coarse_hedges_offsets,
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            num_hedges
        );
        
        // spill non-coarse data structures to host
        std::vector<uint32_t> h_hedges;
        std::vector<dim_t> h_hedges_offsets;
        std::vector<uint32_t> h_srcs_count;
        std::vector<uint32_t> h_touching;
        std::vector<dim_t> h_touching_offsets;
        std::vector<uint32_t> h_inbound_count;
        if (level_idx < SAVE_MEMORY_UP_TO_LEVEL) {
            // TODO: make these async, move everything out of the default stream and use a "compute" and a "transfer" stream
            // TODO: spill inbound counts and src counts too!
            h_hedges.resize(hedges_size);
            h_hedges_offsets.resize(num_hedges + 1);
            h_srcs_count.resize(num_hedges);
            h_touching.resize(touching_size);
            h_touching_offsets.resize(curr_num_nodes + 1);
            h_inbound_count.resize(curr_num_nodes);
            std::cout << "Spilling " << std::fixed << std::setprecision(3) << (float)((hedges_size + touching_size) * sizeof(uint32_t) + (num_hedges + 1 + curr_num_nodes + 1) * sizeof(dim_t)) / (1 << 30) << " GB from device to host at level " << level_idx << " ...\n";
            CUDA_CHECK(cudaMemcpy(h_hedges.data(), d_hedges, hedges_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_hedges_offsets.data(), d_hedges_offsets, (num_hedges + 1) * sizeof(dim_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_srcs_count.data(), d_srcs_count, num_hedges * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_touching.data(), d_touching, touching_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_touching_offsets.data(), d_touching_offsets, (curr_num_nodes + 1) * sizeof(dim_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_inbound_count.data(), d_inbound_count, curr_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_hedges));
            CUDA_CHECK(cudaFree(d_hedges_offsets));
            CUDA_CHECK(cudaFree(d_srcs_count));
            CUDA_CHECK(cudaFree(d_touching));
            CUDA_CHECK(cudaFree(d_touching_offsets));
            CUDA_CHECK(cudaFree(d_inbound_count));
        }
        
        // ======================================
        // recursive call, go down one more level
        auto [num_partitions, d_coarse_partitions] = coarsen_refine_uncoarsen(
            level_idx + 1,
            new_num_nodes,
            d_coarse_hedges,
            d_coarse_hedges_offsets,
            d_coarse_srcs_count,
            new_hedges_size,
            d_coarse_touching,
            d_coarse_touching_offsets,
            new_touching_size,
            d_coarse_inbound_count,
            d_groups_sizes
        );
        // ======================================

        std::cout << "Uncoarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

        // un-spill non-coarse data structures to device
        if (level_idx < SAVE_MEMORY_UP_TO_LEVEL) {
            std::cout << "Unspilling " << std::fixed << std::setprecision(3) << (float)((hedges_size + touching_size) * sizeof(uint32_t) + (num_hedges + 1 + curr_num_nodes + 1) * sizeof(dim_t)) / (1 << 30) << " GB from host to device at level " << level_idx << " ...\n";
            CUDA_CHECK(cudaMalloc(&d_hedges, hedges_size * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_hedges_offsets, (num_hedges + 1) * sizeof(dim_t)));
            CUDA_CHECK(cudaMalloc(&d_srcs_count, num_hedges * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_touching, touching_size * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_touching_offsets, (curr_num_nodes + 1) * sizeof(dim_t)));
            CUDA_CHECK(cudaMalloc(&d_inbound_count, curr_num_nodes * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_hedges, h_hedges.data(), hedges_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_hedges_offsets, h_hedges_offsets.data(), (num_hedges + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_srcs_count, h_srcs_count.data(), num_hedges * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_touching, h_touching.data(), touching_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_touching_offsets, h_touching_offsets.data(), (curr_num_nodes + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_inbound_count, h_inbound_count.data(), curr_num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice));
            std::vector<uint32_t>().swap(h_hedges);
            std::vector<dim_t>().swap(h_hedges_offsets);
            std::vector<uint32_t>().swap(h_srcs_count);
            std::vector<uint32_t>().swap(h_touching);
            std::vector<dim_t>().swap(h_touching_offsets);
            std::vector<uint32_t>().swap(h_inbound_count);
        }

        // prepare this level's uncoarsened partitions
        uint32_t *d_partitions = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partitions, curr_num_nodes * sizeof(uint32_t)));

        {
            // launch configuration - uncoarsening kernel (partitions)
            // uncoarsen d_coarse_partitions into d_partitions
            int threads_per_block = 128; // 128/32 -> 4 warps per block
            int warps_per_block = threads_per_block / WARP_SIZE;
            int num_warps_needed = curr_num_nodes ; // 1 warp per node
            int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
            // launch - uncoarsening kernel (partitions)
            std::cout << "Running uncoarsening kernel (partitions) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
            apply_uncoarsening_partitions<<<blocks, threads_per_block>>>(
                d_groups,
                d_coarse_partitions,
                curr_num_nodes,
                d_partitions
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // cleanup groups
        CUDA_CHECK(cudaFree(d_groups));
        CUDA_CHECK(cudaFree(d_ungroups));
        CUDA_CHECK(cudaFree(d_ungroups_offsets));
        CUDA_CHECK(cudaFree(d_coarse_hedges));
        CUDA_CHECK(cudaFree(d_coarse_hedges_offsets));
        CUDA_CHECK(cudaFree(d_coarse_srcs_count));
        CUDA_CHECK(cudaFree(d_coarse_touching));
        CUDA_CHECK(cudaFree(d_coarse_touching_offsets));
        CUDA_CHECK(cudaFree(d_coarse_inbound_count));
        CUDA_CHECK(cudaFree(d_groups_sizes));
        CUDA_CHECK(cudaFree(d_coarse_partitions)); // allocated at the next inner level, freed here!

        // =============================
        // print some temporary results
        #if VERBOSE
        logPartitions(
            d_partitions,
            d_partitions_sizes,
            curr_num_nodes,
            num_partitions,
            h_max_nodes_per_part
        );
        #endif
        // =============================

        // fiduccia-mattheyses refinement (with events ^-^)
        refinementRepeats(
            cfg,
            d_hedges,
            d_hedges_offsets,
            d_srcs_count,
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            d_hedge_weights,
            d_nodes_sizes,
            level_idx,
            curr_num_nodes,
            num_hedges,
            num_partitions,
            touching_size,
            d_pairs,
            d_f_scores,
            d_partitions,
            d_partitions_sizes,
            d_pins_per_partitions,
            d_partitions_inbound_sizes
        );

        return std::make_tuple(num_partitions, d_partitions);
    };


    // START: the multi-level recursive refinement routine, down we go!
    auto [num_partitions, d_partitions] = coarsen_refine_uncoarsen(
        0, // first level
        num_nodes,
        d_hedges,
        d_hedges_offsets,
        d_srcs_count,
        hg.hedgesFlat().size(),
        d_touching,
        d_touching_offsets,
        touching_hedges_size,
        d_inbound_count,
        d_nodes_sizes
    );

    // final partitions rework: merge small ones and make partition ids zero-based
    thrust::device_ptr<uint32_t> t_partitions(d_partitions);
    thrust::device_ptr<uint32_t> t_partitions_sizes(d_partitions_sizes);
    thrust::device_ptr<uint32_t> t_partitions_inbound_sizes(d_partitions_inbound_sizes);

    if (cfg.mode == Mode::INCC) {
        // greedily merge small partitions
        // => checking inbound constraints w/out deduping, with a straight sum, to make it fast
        thrust::device_vector<uint32_t> t_part_index(num_partitions);
        thrust::sequence(t_part_index.begin(), t_part_index.end());
        // extract small partitions (size < K)
        thrust::device_vector<uint32_t> t_small_parts(num_partitions);
        auto small_end = thrust::copy_if(t_part_index.begin(), t_part_index.end(), t_small_parts.begin(), [=] __host__ __device__ (uint32_t p) { return t_partitions_sizes[p] < SMALL_PART_MERGE_SIZE_THRESHOLD; });
        t_small_parts.resize(small_end - t_small_parts.begin());
        uint32_t smallest_part_size = thrust::reduce(t_partitions_sizes, t_partitions_sizes + num_partitions, UINT32_MAX, thrust::minimum<uint32_t>());
        std::cout << "Smallest partition size: " << smallest_part_size << "\n";
        if (!t_small_parts.empty()) {
            std::cout << "Partitions compression over " << t_small_parts.size() << " partitions ...\n";
            // stable sort small partitions with key (size, inbound, id)
            thrust::stable_sort(t_small_parts.begin(), t_small_parts.end(), [=] __host__ __device__ (uint32_t a, uint32_t b) { uint32_t sa = t_partitions_sizes[a]; uint32_t sb = t_partitions_sizes[b]; if (sa != sb) return sa < sb; uint32_t ia = t_partitions_inbound_sizes[a]; uint32_t ib = t_partitions_inbound_sizes[b]; if (ia != ib) return ia < ib; return a < b; });
            // greedy grouping scan for constraints
            thrust::device_vector<constraints_state> t_constraints_states(t_small_parts.size());
            thrust::transform(t_small_parts.begin(), t_small_parts.end(), t_constraints_states.begin(), [=] __host__ __device__ (uint32_t p) { return constraints_state{ t_partitions_sizes[p], t_partitions_inbound_sizes[p], 0u }; });
            thrust::inclusive_scan(t_constraints_states.begin(), t_constraints_states.end(), t_constraints_states.begin(), [=] __host__ __device__ (const constraints_state& a, const constraints_state& b) { if (a.s + b.s <= h_max_nodes_per_part && a.i + b.i <= h_max_inbound_per_part) return constraints_state{ a.s + b.s, a.i + b.i, a.g }; return constraints_state{ b.s, b.i, a.g + 1 }; });
            // get the id of each node of a group
            thrust::device_vector<uint32_t> t_groups(t_constraints_states.size());
            thrust::transform(t_constraints_states.begin(), t_constraints_states.end(), t_groups.begin(), [] __host__ __device__ (const constraints_state& s) { return s.g; });
            // map groups to a representative partition id (lowest id in the group); groups are already contiguous, a single reduce-by-key is enough
            thrust::device_vector<uint32_t> t_rep_ids(t_groups.size());
            auto rep_end = thrust::reduce_by_key(t_groups.begin(), t_groups.end(), t_small_parts.begin(), thrust::make_discard_iterator(), t_rep_ids.begin(), thrust::equal_to<uint32_t>(), thrust::minimum<uint32_t>());
            t_rep_ids.resize(rep_end.second - t_rep_ids.begin());
            // build the map from partition id to the representative node
            thrust::device_vector<uint32_t> pid_map(num_partitions);
            thrust::sequence(pid_map.begin(), pid_map.end());
            thrust::device_vector<uint32_t> new_pids(t_small_parts.size());
            thrust::gather(t_groups.begin(), t_groups.end(), t_rep_ids.begin(), new_pids.begin());
            thrust::scatter(new_pids.begin(), new_pids.end(), t_small_parts.begin(), pid_map.begin());
            // update partitions
            uint32_t* pid_map_ptr = thrust::raw_pointer_cast(pid_map.data());
            thrust::transform(t_partitions, t_partitions + num_nodes, t_partitions, [pid_map_ptr] __host__ __device__ (uint32_t p) { return pid_map_ptr[p]; });
        } else
            std::cout << "Partitions compression not performed ...\n";
    }

    // make d_partitions zero-based again, if we emptied some partitions... (same logic as that used for d_groups)
    thrust::device_vector<uint32_t> t_indices(num_nodes);
    thrust::sequence(t_indices.begin(), t_indices.end());
    thrust::sort_by_key(t_partitions, t_partitions + num_nodes, t_indices.begin());
    thrust::device_vector<uint32_t> t_headflags(num_nodes);
    t_headflags[0] = 0;
    thrust::transform(t_partitions + 1, t_partitions + num_nodes, t_partitions, t_headflags.begin() + 1, [] __device__ (uint32_t curr, uint32_t prev) { return curr != prev ? 1u : 0u; });
    thrust::inclusive_scan(t_headflags.begin(), t_headflags.end(), t_headflags.begin());
    const uint32_t new_num_partitions = t_headflags.back() + 1;
    thrust::scatter(t_headflags.begin(), t_headflags.end(), t_indices.begin(), t_partitions);

    // copy back results
    std::vector<uint32_t> partitions(num_nodes);
    CUDA_CHECK(cudaMemcpy(partitions.data(), d_partitions, num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // =============================
    // print some example outputs
    #if VERBOSE
    std::set<uint32_t> part_count;
    std::cout << "Final partitioning results:\n";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        uint32_t part = partitions[i];
        part_count.insert(part);
        if (i < std::min<uint32_t>(num_nodes, VERBOSE_LENGTH)) {
            if (part == UINT32_MAX) std::cout << "node " << i << " -> part=none";
            else std::cout << "node " << i << " ->" << " part=" << part;
            std::cout << ((i + 1) % 4 == 0 ? "\n" : "\t");
        }
    }
    std::cout << "Partitions count: " << part_count.size() << " (plus " << num_partitions - part_count.size() << " empty ones)" << "\n";
    if (new_num_partitions != part_count.size())
        std::cerr << "WARNING, distinct partitions count (" << part_count.size() << ") does not match the computed number of partitions when zero-ing their ids (" << new_num_partitions << ") !!\n";
    std::set<uint32_t>().swap(part_count);
    #endif
    // =============================

    // cleanup device memory
    CUDA_CHECK(cudaFree(d_hedges));
    CUDA_CHECK(cudaFree(d_hedges_offsets));
    CUDA_CHECK(cudaFree(d_srcs_count));
    //CUDA_CHECK(cudaFree(d_neighbors)); // should have already been freed at the innermost recursion level
    //CUDA_CHECK(cudaFree(d_neighbors_offsets));
    CUDA_CHECK(cudaFree(d_touching));
    CUDA_CHECK(cudaFree(d_touching_offsets));
    CUDA_CHECK(cudaFree(d_inbound_count));
    CUDA_CHECK(cudaFree(d_hedge_weights));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_f_scores));
    CUDA_CHECK(cudaFree(d_u_scores));
    CUDA_CHECK(cudaFree(d_slots));
    CUDA_CHECK(cudaFree(d_dp_scores));
    CUDA_CHECK(cudaFree(d_nodes_sizes));
    CUDA_CHECK(cudaFree(d_partitions));
    CUDA_CHECK(cudaFree(d_partitions_sizes));
    CUDA_CHECK(cudaFree(d_pins_per_partitions));
    CUDA_CHECK(cudaFree(d_partitions_inbound_sizes));

    // final sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaStreamDestroy(transfer_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));

    CUDA_CHECK(cudaEventRecord(d_time_stop));
    CUDA_CHECK(cudaEventSynchronize(d_time_stop));
    float d_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d_total_ms, d_time_start, d_time_stop));
    CUDA_CHECK(cudaEventDestroy(d_time_start));
    CUDA_CHECK(cudaEventDestroy(d_time_stop));
    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << "Stopping timer...\n";

    // === CUDA STUFF ENDS HERE ===
    // ============================

    std::cout << "CUDA section: complete; proceeding with partitioning results validation and evalution...\n";

    double total_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
    std::cout << "Total device execution time: " << std::fixed << std::setprecision(3) << d_total_ms << " ms\n";
    std::cout << "Total host execution time: " << std::fixed << std::setprecision(3) << total_ms << " ms\n";

    /*DEBUG: check the metrics calculation!
    float dbg_conn = hg.connectivityFromPart(partitions);
    float dbg_cutn = hg.cutnetFromPart(partitions);
    */

    if (constr.checkPartitionValidity(hg, partitions, true)) {
        // log metrics
        auto partitioned_hg = hg.getPartitionsHypergraph(partitions, 2, true); // remove the destination if self-cycles happen
        auto hedge_overlap = constr.hedgeOverlap(hg, partitions);
        std::cout << "Partitioned hypergraph:\n";
        std::cout << "  Nodes:         " << partitioned_hg.nodes() << "\n";
        std::cout << "  Hyperedges:    " << partitioned_hg.hedges().size() << "\n";
        std::cout << "  Total pins:    " << partitioned_hg.hedgesFlat().size() << "\n";
        std::cout << "  Cut-net:       " << partitioned_hg.cutnet() << "\n";
        std::cout << "  Connectivity:  " << partitioned_hg.connectivity() << "\n";
        std::cout << "  SOED:          " << hg.soedFromPart(partitions) << "\n";
        std::cout << "  Hedge overlap: " << std::fixed << std::setprecision(3) << hedge_overlap.ar_mean << " ar. mean, " << hedge_overlap.geo_mean << " geo. mean\n";
        //if (dbg_conn != partitioned_hg.connectivity()) std::cerr << "ERROR, incorrect metric calculation for connectivity: " << dbg_conn << " vs " << partitioned_hg.connectivity() << " !!\n";
        //if (dbg_cutn != partitioned_hg.cutnet()) std::cerr << "ERROR, incorrect metric calculation for cut-net: " << dbg_cutn << " vs " << partitioned_hg.cutnet() << " !!\n";
        
        // save results
        saveResult(cfg, partitioned_hg, partitions);
    } else {
        std::cerr << "WARNING, invalid partitining !!\n";
        return 1;
    }

    return 0;
}
