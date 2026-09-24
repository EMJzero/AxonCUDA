#include <set>
#include <tuple>
#include <chrono>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <iostream>
#include <optional>
#include <functional>
#include <filesystem>

#include <omp.h>

#include "hgraph.hpp"
#include "constr.hpp"
#include "runconfig.hpp"

#include "utils.hpp"
#include "prims.hpp"
#include "eval_instr.hpp"
#include "defines.hpp"
#include "constants.hpp"
#include "data_types.hpp"
#include "coarsening.hpp"
#include "chaining.hpp"
#include "construction.hpp"
#include "refinement.hpp"
#include "init_part.hpp"
#include "postprocess.hpp"

using namespace hgraph;
using namespace constraints;
using namespace config;


// validate the final partitioning, then log its quality metrics and save it to file
// returns false if the partitioning violates the constraints
static bool evaluateAndSaveResults(
    runconfig &cfg,
    const Constraints &constr,
    const HyperGraph &hg,
    const std::vector<uint32_t> &partitions
) {
    if (!constr.checkPartitionValidity(hg, partitions, cfg.verbose_errs_and_warns))
        return false;

    // log metrics
    DBG(cfg) std::cout << "Preparing partitioned hypergraph and computing quality metrics...\n";
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

    // save results
    saveResult(cfg, partitioned_hg, partitions);
    return true;
}

// first "model name" in /proc/cpuinfo
static std::string cpuModelName() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line))
        if (line.rfind("model name", 0) == 0)
            return line.substr(line.find(':') + 2);
    return "unknown";
}


int main(int argc, char** argv) {
    if (argc == 1) {
        printHelp();
        return 0;
    }

    // parse CLI args
    runconfig cfg = parseArgs(argc, argv);
    omp_set_num_threads((int)cfg.threads);

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
    std::cout << "  Inbound pins per partition:  " << constr.pinsPerPart() << "\n";
    std::cout << "  Maximum partitions:          " << constr.maxParts() << "\n";

    // quick path: a single partition was requested ('-k 1'), every node trivially belongs to partition 0
    if (cfg.mode == Mode::KWAY && cfg.kway == 1) {
        INFO(cfg) std::cout << "Single partition requested ('-k 1'), skipping the partitioning routine...\n";
        // HP: no duplicates per hedge, no self-cycles (keep the src only) -> same treatment as the regular path, so that metrics are comparable
        hg.deduplicateHyperedges(2, false); // remove the srcs
        std::vector<uint32_t> partitions(hg.nodes(), 0u);
        if (!evaluateAndSaveResults(cfg, constr, hg, partitions)) {
            std::cerr << "ERROR, invalid partitining !!\n";
            return 1;
        }
        return 0;
    }

    std::cout << "Using settings:\n";
    std::cout << "  Candidates count:            " << cfg.candidates_count << "\n";
    std::cout << "  Refinement repetitions:      " << cfg.refine_repeats << "\n";
    std::cout << "  Pins per partition mode:     " << (cfg.ppp_mode == PinsPerPartMode::DENSE ? "dense" : (cfg.ppp_mode == PinsPerPartMode::SPARSE ? "sparse" : "auto")) << "\n";
    std::cout << "  Flags: " << (cfg.parallel_touching_construction ? "ptc " : "") << (cfg.initial_partitions_merge ? "ipm " : "") << (cfg.exact_matching ? "xdp " : "") << "\n";

    std::cout << "CPU:\n";
    std::cout << "  Model name:             " << cpuModelName() << "\n";
    std::cout << "  Hardware threads:       " << omp_get_num_procs() << "\n";
    std::cout << "  OpenMP threads:         " << cfg.threads << "\n";

    INFO(cfg) std::cout << "Preparing hypergraph data...\n";

    // build incidence sets while preparing the hypergraph, unless explicitly required to do it in parallel
    if (!cfg.parallel_touching_construction)
        hg.buildIncidenceSets();

    if (!cfg.parallel_touching_construction && cfg.verbose_errs_and_warns && !constr.checkFit(hg, false, cfg.verbose_errs_and_warns))
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
    */

    const uint32_t num_hedges = static_cast<uint32_t>(hg.hedges().size());

    // HP: no duplicates per hedge, no self-cycles (keep the src only, arg=false -> still consider the hedge among the src's inbounds, arg=true -> update the inbound set to match)
    hg.deduplicateHyperedges(2, false); // remove the srcs

    // total number of distinct nodes (for output indexing)
    const uint32_t num_nodes = hg.nodes(); // nodes count used when allocating outputs

    // constraints
    max_nodes_per_part = constr.nodesPerPart();
    max_inbound_per_part = constr.inboundPerPart();
    max_pins_per_part = constr.pinsPerPart();
    const uint32_t max_parts = constr.maxParts();
    const uint32_t target_parts = std::min(max_parts, (num_nodes + max_nodes_per_part - 1) / max_nodes_per_part);
    assert(max_nodes_per_part <= INT32_MAX);
    assert(max_inbound_per_part <= INT32_MAX);
    assert(max_pins_per_part <= INT32_MAX);
    assert(max_parts <= INT32_MAX);

    INFO(cfg) std::cout << "Starting timer...\n";
    auto time_start = std::chrono::high_resolution_clock::now();

    INFO(cfg) std::cout << "Setting up memory...\n";

    // ============================
    // === CORE STUFF GOES HERE ===

    const dim_t hedges_size = hg.hedgesFlat().size();
    buffer<uint32_t> hedges(hedges_size); // hedges[hedges_offsets[hedge idx]] -> contigous array of pins of hedge (stored as src+destinations, with the srcs first)
    buffer<dim_t> hedges_offsets((dim_t)num_hedges + 1); // hedges_offsets[hedge idx] -> hedge start idx in hedges
    buffer<uint32_t> srcs_count(num_hedges); // srcs_count[hedge idx] -> number of sources of hedge idx
    buffer<uint32_t> touching; // touching[touching_offsets[node idx]] -> contigous inbound+outbout set/array (first inbound, then outbound) of node
    buffer<dim_t> touching_offsets; // touching_offsets[node idx] -> touching set start idx in touching
    buffer<uint32_t> inbound_count; // inbound_count[node idx] -> how many hedge of touching[node idx] are inbound (inbound hedges are before inbound_count[node idx], then outbound)
    buffer<float> hedge_weights(num_hedges); // hedge_weights[hedge idx] -> weight
    buffer<uint32_t> pairs((dim_t)num_nodes * cfg.candidates_count); // pairs[node idx * candidates_count + i] -> i-th best neighbor of node idx
    buffer<float> f_scores(num_nodes); // connection strength for each pair, used during refinement
    buffer<uint32_t> u_scores((dim_t)num_nodes * cfg.candidates_count); // fixed point version of the above, used for the candidates and grouping kernels
    buffer<slot> slots(num_nodes); // slot to finalize node pairs during grouping
    buffer<uint32_t> nodes_sizes(num_nodes); // nodes_size[node idx] -> how many nodes the node counts as towards the partition size limit
    buffer<uint32_t> nodes_pins(num_nodes); // nodes_pins[node idx] -> how many (inbound) pins the node counts as towards the partition pins limit
    buffer<uint32_t> partitions_sizes; // partitions_sizes[idx] -> how many nodes (by total size) are in the partition
    buffer<uint32_t> partitions_inbound_sizes; // partitions_inbound_sizes[partition] -> distinct inbound hedges count for "partition"
    buffer<uint32_t> partitions_pins; // partitions_pins[idx] -> how many inbound pins (by total count) are in the partition

    // copy over the hypergraph (the first write also places it)
    par_copy<uint32_t>(hedges.data(), hg.hedgesFlat().data(), hedges_size);
    #pragma omp parallel for schedule(static) if(num_hedges > PARALLEL_GRAIN)
    for (uint32_t i = 0; i < num_hedges; ++i) {
        hedges_offsets[i] = static_cast<dim_t>(hg.hedges()[i].offset());
        srcs_count[i] = hg.hedges()[i].src_count();
        hedge_weights[i] = hg.hedges()[i].weight();
    }
    hedges_offsets[num_hedges] = hedges_size;

    // initialize
    par_fill<uint32_t>(nodes_sizes.data(), num_nodes, 1u); // each initial node counts as 1 (NOTE: can be tuned to give some nodes more "space")

    // prepare touching sets
    EVP_PUSH("setup_touching");
    dim_t touching_hedges_size;
    if (cfg.parallel_touching_construction) {
        std::tie(touching_hedges_size, touching, touching_offsets, inbound_count) = buildTouching(
            cfg,
            hedges.data(),
            hedges_offsets.data(),
            srcs_count.data(),
            num_nodes,
            num_hedges
        );
        assert(touching_hedges_size == hedges_size); // touching sets are the reverse map of hedge pins, hence they must be the same size
    } else {
        std::tie(touching_hedges_size, touching, touching_offsets, inbound_count) = buildTouchingHost(
            cfg,
            hg
        );
    }
    // |
    // initialize inbound pin counts per node pins from inbound set cardinality
    par_copy<uint32_t>(nodes_pins.data(), inbound_count.data(), num_nodes);
    EVP_POP(); // setup_touching

    INFO(cfg) std::cout << "Starting core timer...\n";
    auto time_core_start = std::chrono::high_resolution_clock::now();

    // prepare neighborhoods
    EVP_PUSH("construct_neighbors");
    dim_t neighbors_size;
    buffer<uint32_t> neighbors; // neighbors[neighbors_offsets[node idx]] -> contigous set/array of neighbors of node (its neighborhood)
    buffer<dim_t> neighbors_offsets; // neighbors_offsets[node idx] -> neighbors set start idx in neighbors
    std::tie(neighbors_size, neighbors, neighbors_offsets) = buildNeighbors(
        cfg,
        hedges.data(),
        hedges_offsets.data(),
        touching.data(),
        touching_offsets.data(),
        num_nodes
    );
    EVP_POP(); // construct_neighbors


    // returns the number of partitions and the final partitions buffer
    std::function<std::tuple<uint32_t, buffer<uint32_t>>(const uint32_t, const uint32_t, const buffer<uint32_t>&, const buffer<dim_t>&, const buffer<uint32_t>&, const dim_t, const buffer<uint32_t>&, const buffer<dim_t>&, const dim_t, const buffer<uint32_t>&, const buffer<uint32_t>&, const buffer<uint32_t>&)> coarsen_refine_uncoarsen = [&](
        const uint32_t level_idx,
        const uint32_t curr_num_nodes,
        const buffer<uint32_t>& hedges,
        const buffer<dim_t>& hedges_offsets,
        const buffer<uint32_t>& srcs_count,
        const dim_t hedges_size,
        const buffer<uint32_t>& touching,
        const buffer<dim_t>& touching_offsets,
        const dim_t touching_size,
        const buffer<uint32_t>& inbound_count,
        const buffer<uint32_t>& nodes_sizes,
        const buffer<uint32_t>& nodes_pins
    ) { // this is a lambda
        INFO(cfg) std::cout << "Coarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

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

        // ======================================
        // k-way base case, build inital partitioning
        // => condition: passed the nodes threshold
        if (cfg.mode == Mode::KWAY && curr_num_nodes < KWAY_INIT_UPPER_THREASHOLD) {
            auto [init_partitions, init_partitions_sizes] = initial_partitioning_kahypar(
                cfg,
                curr_num_nodes,
                num_hedges,
                hedges.data(),
                hedges_offsets.data(),
                hedge_weights.data(),
                touching_offsets.data(),
                hedges_size,
                nodes_sizes.data(),
                cfg.kway,
                cfg.epsi
            );
            partitions_sizes = std::move(init_partitions_sizes);
            partitions_inbound_sizes = buffer<uint32_t>(max_parts); // written by the refinement, never binding in k-way mode
            partitions_pins = buffer<uint32_t>(max_parts);
            // k-way leaves the inbound pins constraint maxed out, so this accumulator is never enforced against, but the
            // refinement still reads it as an event's base value: zero it rather than feeding the events kernel garbage
            par_fill<uint32_t>(partitions_pins.data(), max_parts, 0u);

            // neighbors are no longer needed after coarsening is done
            neighbors.release();
            neighbors_offsets.release();

            return std::make_tuple(max_parts, std::move(init_partitions));
        }
        // ======================================

        EVP_PUSH(std::string("coarsen L") + std::to_string(level_idx));

        // each node picks its candidate(s)
        candidatesProposal(
            cfg,
            hedges.data(),
            hedges_offsets.data(),
            srcs_count.data(),
            neighbors.data(),
            neighbors_offsets.data(),
            touching.data(),
            touching_offsets.data(),
            inbound_count.data(),
            hedge_weights.data(),
            nodes_sizes.data(),
            nodes_pins.data(),
            curr_num_nodes,
            pairs.data(),
            u_scores.data()
        );

        // =============================
        // print some temporary results
        LOG(cfg) {
            logCandidates(
                cfg,
                pairs.data(),
                u_scores.data(),
                curr_num_nodes
            );
        }
        // =============================

        // matching over the candidates graph
        uint32_t new_num_nodes; // new number of nodes after this coarsening round
        buffer<uint32_t> groups; // groups[node idx] -> node's group id (zero-based)
        buffer<uint32_t> groups_sizes; // group_sizes[group id] = sum of sizes of all nodes in that group
        buffer<uint32_t> groups_pins; // group_pins[group id] = sum of pins of all nodes in that group
        buffer<uint32_t> ungroups; // ungroups[ungroups_offsets[group id] + i] -> the group's i-th node (its original idx)
        buffer<dim_t> ungroups_offsets; // ungroups_offsets[group id] -> offset of the group's first node in ungroups
        std::tie(new_num_nodes, groups, groups_sizes, groups_pins, ungroups, ungroups_offsets) = groupNodes(
            cfg,
            inbound_count.data(),
            pairs.data(),
            u_scores.data(),
            nodes_sizes.data(),
            nodes_pins.data(),
            curr_num_nodes,
            slots.data()
        );

        // ======================================
        // k-way base case, build inital partitioning
        // => condition: too little shrinking to justify further coarsening
        if (cfg.mode == Mode::KWAY && ((float)new_num_nodes / (float)curr_num_nodes > KWAY_INIT_SHRINK_RATIO_LIMIT || new_num_nodes < KWAY_INIT_LOWER_THREASHOLD)) {
            auto [init_partitions, init_partitions_sizes] = initial_partitioning_kahypar(
                cfg,
                curr_num_nodes,
                num_hedges,
                hedges.data(),
                hedges_offsets.data(),
                hedge_weights.data(),
                touching_offsets.data(),
                hedges_size,
                nodes_sizes.data(),
                cfg.kway,
                cfg.epsi
            );
            partitions_sizes = std::move(init_partitions_sizes);
            partitions_inbound_sizes = buffer<uint32_t>(max_parts); // written by the refinement, never binding in k-way mode
            partitions_pins = buffer<uint32_t>(max_parts);
            // k-way leaves the inbound pins constraint maxed out, so this accumulator is never enforced against, but the
            // refinement still reads it as an event's base value: zero it rather than feeding the events kernel garbage
            par_fill<uint32_t>(partitions_pins.data(), max_parts, 0u);

            // neighbors are no longer needed after coarsening is done
            neighbors.release();
            neighbors_offsets.release();

            EVP_POP(); // coarsen L (k-way shrink base case)
            return std::make_tuple(max_parts, std::move(init_partitions));
        }
        // ======================================

        // ======================================
        // base case, return inital partitioning
        // NOTE: with the current setup, we stop as soon as we clear "max_parts" and let refinement eventually empty some if they are too many
        // NOTE: if not enough coarse nodes are formed after after a while, stop anyway
        float coarsening_ratio = (float)new_num_nodes / curr_num_nodes;
        if (
            new_num_nodes <= target_parts
            || new_num_nodes == curr_num_nodes
            || (level_idx >= NUMBER_OF_LEVELS_WITH_NO_SHRINK_LIMIT && coarsening_ratio > SHRINK_RATIO_LIMIT)
        ) {
            // try to further merge partitions (groups) so long as it is allowed by constraints
            // => here the goal is just to reduce partitions count, no concern about mergees connection strenght
            if (cfg.initial_partitions_merge) {
                new_num_nodes = greedyMergeGroups(
                    cfg,
                    nodes_sizes.data(),
                    nodes_pins.data(),
                    inbound_count.data(),
                    ungroups.data(),
                    ungroups_offsets.data(),
                    curr_num_nodes,
                    new_num_nodes,
                    groups.data(),
                    groups_sizes.data(),
                    groups_pins.data()
                );
            }

            // HERE we repurpose the coarsening routine as the routine for initial partitions:
            // - num_partitions = new_num_nodes
            // - partitions = groups

            // base case, reached the target number of partitions
            if (new_num_nodes <= target_parts) {
                INFO(cfg) std::cout << "Minimal initial partitioning built at level " << level_idx << ", remaining nodes=" << curr_num_nodes << ", number of partitions=" << new_num_nodes << "\n";
            } else if (new_num_nodes <= max_parts) { // still valid but not minimal partitions count
                INFO(cfg) std::cout << "Initial partitioning built at level " << level_idx << ", remaining nodes=" << curr_num_nodes << ", number of partitions=" << new_num_nodes << "\n";
                if (new_num_nodes == curr_num_nodes) { // impossible to coarsen further
                    ERR(cfg) std::cerr << "WARNING: could not coarsen any further, the partitioning is valid, but didn't reach the minimal number of partitions (" << target_parts << ") !!\n";
                } else if (coarsening_ratio > SHRINK_RATIO_LIMIT) { // too little pairs created, too expensive to coarsen further
                    ERR(cfg) std::cerr << "WARNING: coarsening rate too low (" << std::fixed << std::setprecision(2) << 1/coarsening_ratio << " < " << 1/SHRINK_RATIO_LIMIT << ") the partitioning is valid, but didn't reach the minimal number of partitions (" << target_parts << ") !!\n";
                }
            } else { // base case, failure to coarsen further
                ERR(cfg) std::cerr << "FAILED TO COARSEN FURTHER at level " << level_idx << ", remaining nodes=" << curr_num_nodes << " number of partitions=" << new_num_nodes << " max allowed partitions=" << max_parts << "\n";
                if (new_num_nodes == curr_num_nodes) { // impossible to coarsen further
                    ERR(cfg) std::cerr << "  Reason: no coarsening pairs were formed\n";
                } else if (coarsening_ratio > SHRINK_RATIO_LIMIT) { // too little pairs created, too expensive to coarsen further
                    ERR(cfg) std::cerr << "  Reason: coarsening rate too low (" << std::fixed << std::setprecision(2) << 1/coarsening_ratio << " < " << 1/SHRINK_RATIO_LIMIT << ")\n";
                }
                ERR(cfg) std::cerr << "WARNING: falling back to returning current groups as individual partitions !!\n";
            }

            // neighbors are no longer needed after coarsening is done
            neighbors.release();
            neighbors_offsets.release();

            // prepare initial partition sizes
            // NOTE: current groups become the partitions, and so group sizes become partition sizes
            partitions_sizes = std::move(groups_sizes);
            // NOTE: same for group pins, they become the partitions' pin counts
            partitions_pins = std::move(groups_pins);

            // perpare inbound set size counts per partition (written by "refinementRepeats")
            partitions_inbound_sizes = buffer<uint32_t>(new_num_nodes);

            EVP_POP(); // coarsen L (initial-partitioning base case)
            return std::make_tuple(new_num_nodes, std::move(groups));
        }
        // ======================================

        // =============================
        // print some temporary results
        LOG(cfg) logGroups(
            cfg,
            pairs.data(),
            groups.data(),
            groups_sizes.data(),
            groups_pins.data(),
            curr_num_nodes,
            new_num_nodes
        );
        // =============================

        // prepare coarse neighbors buffers
        // NOTE: overwrites previous neighbors
        std::tie(neighbors_size, neighbors, neighbors_offsets) = coarsenNeighbors(
            cfg,
            neighbors.data(),
            neighbors_offsets.data(),
            groups.data(),
            ungroups.data(),
            ungroups_offsets.data(),
            new_num_nodes
        );

        // prepare coarse hedges buffers
        dim_t new_hedges_size;
        buffer<uint32_t> coarse_hedges;
        buffer<dim_t> coarse_hedges_offsets;
        buffer<uint32_t> coarse_srcs_count;
        std::tie(new_hedges_size, coarse_hedges, coarse_hedges_offsets, coarse_srcs_count) = coarsenHedges(
            cfg,
            hedges.data(),
            hedges_offsets.data(),
            srcs_count.data(),
            groups.data(),
            num_hedges,
            new_num_nodes
        );

        // prepare coarse touching buffers
        dim_t new_touching_size;
        buffer<uint32_t> coarse_touching;
        buffer<dim_t> coarse_touching_offsets;
        buffer<uint32_t> coarse_inbound_count;
        std::tie(new_touching_size, coarse_touching, coarse_touching_offsets, coarse_inbound_count) = coarsenTouching(
            cfg,
            touching.data(),
            touching_offsets.data(),
            inbound_count.data(),
            ungroups.data(),
            ungroups_offsets.data(),
            new_num_nodes,
            num_hedges
        );

        EVP_POP(); // coarsen L{level_idx}

        // ======================================
        // recursive call, go down one more level
        auto [num_partitions, coarse_partitions] = coarsen_refine_uncoarsen(
            level_idx + 1,
            new_num_nodes,
            coarse_hedges,
            coarse_hedges_offsets,
            coarse_srcs_count,
            new_hedges_size,
            coarse_touching,
            coarse_touching_offsets,
            new_touching_size,
            coarse_inbound_count,
            groups_sizes,
            groups_pins
        );
        // ======================================

        EVP_PUSH(std::string("uncoarsen_refine L") + std::to_string(level_idx));

        INFO(cfg) std::cout << "Uncoarsening level " << level_idx << ", remaining nodes=" << curr_num_nodes << "\n";

        // prepare this level's uncoarsened partitions
        buffer<uint32_t> partitions(curr_num_nodes);

        // uncoarsen coarse_partitions into partitions
        LAUNCH(cfg) RUN << "uncoarsening kernel (partitions) (threads=" << cfg.threads << ") ...\n";
        apply_uncoarsening_partitions(
            groups.data(),
            coarse_partitions.data(),
            curr_num_nodes,
            partitions.data()
        );

        // cleanup groups
        groups.release();
        ungroups.release();
        ungroups_offsets.release();
        coarse_hedges.release();
        coarse_hedges_offsets.release();
        coarse_srcs_count.release();
        coarse_touching.release();
        coarse_touching_offsets.release();
        coarse_inbound_count.release();
        groups_sizes.release();
        groups_pins.release();
        coarse_partitions.release(); // allocated at the next inner level, freed here!

        // =============================
        // print some temporary results
        LOG(cfg) {
            logPartitions(
                partitions.data(),
                partitions_sizes.data(),
                partitions_inbound_sizes.data(),
                partitions_pins.data(),
                curr_num_nodes,
                num_partitions
            );
        }
        // =============================

        // fiduccia-mattheyses refinement (with events ^-^)
        refinementRepeats(
            cfg,
            hedges.data(),
            hedges_offsets.data(),
            srcs_count.data(),
            touching.data(),
            touching_offsets.data(),
            inbound_count.data(),
            hedge_weights.data(),
            nodes_sizes.data(),
            nodes_pins.data(),
            level_idx,
            curr_num_nodes,
            num_hedges,
            num_partitions,
            touching_size,
            level_idx == 0, // on the final level -> return true inbound set sizes
            pairs.data(),
            f_scores.data(),
            partitions.data(),
            partitions_sizes.data(),
            partitions_inbound_sizes.data(),
            partitions_pins.data()
        );

        EVP_POP(); // uncoarsen_refine L{level_idx}
        return std::make_tuple(num_partitions, std::move(partitions));
    }; // coarsen_refine_uncoarsen end


    // START: the multi-level recursive refinement routine, down we go!
    auto [num_partitions, final_partitions] = coarsen_refine_uncoarsen(
        0, // first level
        num_nodes,
        hedges,
        hedges_offsets,
        srcs_count,
        hedges_size,
        touching,
        touching_offsets,
        touching_hedges_size,
        inbound_count,
        nodes_sizes,
        nodes_pins
    );


    if (cfg.mode == Mode::INCC) {
        // final partitions rework: merge small ones and make partition ids zero-based
        mergeSmallPartitions(
            cfg,
            partitions_sizes.data(),
            partitions_inbound_sizes.data(),
            partitions_pins.data(),
            num_nodes,
            num_partitions,
            final_partitions.data()
        );
    }

    // make partitions zero-based again, if we emptied some partitions... (same logic as that used for groups)
    const uint32_t new_num_partitions = zeroBaseIds(num_nodes, num_partitions, final_partitions.data());

    // copy back results
    std::vector<uint32_t> partitions(final_partitions.data(), final_partitions.data() + num_nodes);

    // =============================
    // print some example outputs
    LOG(cfg) {
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
    }
    // =============================

    auto time_end = std::chrono::high_resolution_clock::now();
    INFO(cfg) std::cout << "Stopping timer...\n";

    // === CORE STUFF ENDS HERE ===
    // ============================

    INFO(cfg) std::cout << "Core section: complete; proceeding with partitioning results validation and evalution...\n";

    double total_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
    double core_ms = std::chrono::duration<double, std::milli>(time_end - time_core_start).count();
    // core: excluding the initialization of hedges and incidence sets in memory
    std::cout << "Total core execution time: " << std::fixed << std::setprecision(3) << core_ms << " ms\n";
    std::cout << "Total execution time: " << std::fixed << std::setprecision(3) << total_ms << " ms\n";
    INFO(cfg) EVP_REPORT();

    EVP_PUSH("postproc_host");
    bool ok = evaluateAndSaveResults(cfg, constr, hg, partitions);
    EVP_POP(); // postproc_host
    if (!ok) {
        std::cerr << "ERROR, invalid partitining !!\n";
        return 1;
    }

    return 0;
}
