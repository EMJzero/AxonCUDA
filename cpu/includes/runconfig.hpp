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

    enum class PinsPerPartMode {
        AUTO,  // dense matrix if it fits in (a fraction of) the available RAM, sparse otherwise
        DENSE, // always the dense matrix
        SPARSE // always the sparse bitmap-indexed representation
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
        uint32_t candidates_count; // number of candidates to propose per node during coarsening
        uint32_t refine_repeats; // number of repetitions for the refinement routine per level
        uint32_t threads; // number of OpenMP threads to use
        PinsPerPartMode ppp_mode; // pins per partition representation used by the refinement
        bool parallel_touching_construction; // whether to construct touching/incidence sets in parallel or sequentially while loading
        bool initial_partitions_merge; // whether to greedily merge initial partitions to minimize their number
        bool no_pins_constraint; // whether to disable the pins per partition constraint by maxing it out
        bool exact_matching; // whether a node's grouping gain accounts for its whole subtree (exact maximum weight matching) or only for its candidate score
        bool verbose_logs; // whether to log what is happening inside the algorithms
        bool verbose_info; // whether to log the step/phase where the program is at
        bool verbose_errs_and_warns; // whether to log errs and warnings
        bool verbose_kernel_launches; // whether to log every parallel loop or not
        bool debug; // whether to run extra debug checks
    };

    void printHelp();

    runconfig parseArgs(int argc, char** argv);

    hgraph::HyperGraph loadHgraph(runconfig &cfg);

    constraints::Constraints setupConstr(runconfig &cfg, hgraph::HyperGraph hg);

    void saveResult(runconfig &cfg, hgraph::HyperGraph partitioned_hg, std::vector<uint32_t> partitions);
}
