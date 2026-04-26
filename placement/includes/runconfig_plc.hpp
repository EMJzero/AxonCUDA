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

    enum class SpaceFillingCurve {
        HILB, // Hilbert curve
        SNAK, // S-like snake curve
        ZORD, // Z-order curve
        QUAD // quadtree-style layout
    };

    static constexpr std::pair<SpaceFillingCurve, const char*> SPACE_FILLING_CURVE_NAMES[] = {
        { SpaceFillingCurve::HILB, "hilb" },
        { SpaceFillingCurve::SNAK, "snak" },
        { SpaceFillingCurve::ZORD, "zord" },
        { SpaceFillingCurve::QUAD, "quad" }
    };

    struct runconfig {
        std::string load_path; // path to the hgraph to load 'n' partition
        std::string save_path; // path where to save the placement data
        std::string constraints; // name the constraints set to use
        uint32_t labelprop_repeats; // number of labelprop rounds performed at each level of recursive bisection in the parallel initial placement
        uint32_t fd_iterations; // number of force-directed refinement iterations to perform
        uint32_t candidates_count; // number of candidate swaps proposed per node during force-directed refinement
        uint32_t multi_start_override; // imposes the number of multi-start attempts at placement
        uint32_t num_host_threads; // imposes the number of host threads and GPU streams to spawn to handle the multi-start attempts
        SpaceFillingCurve space_filling_curve; // space filling curve to use for the 1D-to-2D locality-preserving mapping
        bool feedforward_order; // if true, use the greedy sequential feedforward initial partitioning (runs on the host !!)
        bool device_touching_construction; // whether to construct touching/incidence sets on the device or the host
        uint64_t seed; // seed for the multi-start and recursive bisection methods
        bool verbose_logs; // whether to log what is happening inside the algorithms
        bool verbose_info; // whether to log the step/phase where the program is at
        bool verbose_errs_and_warns; // whether to log errs and warnings
        bool verbose_kernel_launches; // whether to log every kernel launch or not
    };

    void printHelp();

    runconfig parseArgs(int argc, char** argv);

    hgraph::HyperGraph loadHgraph(runconfig &cfg);

    hwmodel::HardwareModel setupNMH(runconfig &cfg);

    void saveResult(runconfig &cfg, std::vector<hwgeom::Coord2D> h_placement);

    const char* SFCtoString(SpaceFillingCurve curve);

    bool parseSFC(const std::string& name, SpaceFillingCurve& curve);
}
