#pragma once
#include <string>
#include <vector>
#include <cstdint>

#include "constr.hpp"

namespace hgraph {
    class HyperGraph;
}

namespace config {

    enum class Mode {
        INCC, // incidence constraints
        KWAY  // k-way balanced
    };

    enum class ConstrType {
        KWAY, // k-way balanced constraints
        NAME, // named constraints configuration (incidence constraints)
        MANL, // manual constraints configuration (incidence constraints)
        NONE  // no constraints provided -> use default
    };

    struct runconfig {
        std::string load_path; // path to the hgraph to load 'n' partition
        std::string save_path; // path where to save the partitioned hgraph
        std::string part_path; // path where to save the explicit partitioning
        Mode mode; // problem type to solve (size and incidence constrained VS k-way balanced)
        ConstrType constr_type; // variables used for defining a constraints
        constraints::ConstraintsConfig constr_config; // constraints set to use
        uint32_t kway; // "k" for k-way partitioning
        float epsi; // "epsilon" for k-way partitioning
        float oversized_multiplier; // multiplicative factor for oversized deduplication buffers
        uint32_t candidates_count; // number of candidates to propose per node during coarsening
        uint32_t refine_repeats; // number of repetitions for the refinement routine per level
        bool device_touching_construction; // whether to construct touching/incidence sets on the device or the host
    };

    void printHelp();

    runconfig parseArgs(int argc, char** argv);

    hgraph::HyperGraph loadHgraph(runconfig cfg);

    constraints::Constraints setupConstr(runconfig cfg, hgraph::HyperGraph hg);

    void saveResult(runconfig cfg, hgraph::HyperGraph partitioned_hg, std::vector<uint32_t> partitions);
}