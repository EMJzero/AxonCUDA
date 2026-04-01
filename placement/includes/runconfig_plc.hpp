#pragma once
#include <string>
#include <vector>
#include <cstdint>

#include "nmhardware.hpp"

namespace hgraph {
    class HyperGraph;
}

namespace hwgeom {
    struct Coord2D;
}

namespace hwmodel {
    class HardwareModel;
}

namespace config_plc {

    struct runconfig {
        std::string load_path; // path to the hgraph to load 'n' partition
        std::string save_path; // path where to save the placement data
        std::string constraints; // name the constraints set to use
        uint32_t labelprop_repeats; // number of labelprop rounds performed at each level of recursive bisection in the parallel initial placement
        uint32_t fd_iterations; // number of force-directed refinement iterations to perform
        uint32_t candidates_count; // number of candidate swaps proposed per node during force-directed refinement
        bool feedforward_order; // if true, use the greedy sequential feedforward initial partitioning (runs on the host !!)
        bool device_touching_construction; // whether to construct touching/incidence sets on the device or the host
        uint64_t seed; // seed for the multi-start and recursive bisection methods
    };

    void printHelp();

    runconfig parseArgs(int argc, char** argv);

    hgraph::HyperGraph loadHgraph(runconfig cfg);

    hwmodel::HardwareModel setupNMH(runconfig cfg);

    void saveResult(runconfig cfg, std::vector<hwgeom::Coord2D> h_placement);
}