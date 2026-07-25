#pragma once
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

#include "topology.hpp"
#include "nmhardware.hpp"

namespace hgraph {
    class HyperGraph;
}

namespace config_plc {

    // TODO: change from fixed-dimensionality topologies to topology+N in the CLI arguments

    enum class TargetTopology {
        LATTICE2D, // 2D lattice -> intrinsic dim. = 2, distance func. = Manhattan
        TORUS6D, // 6D torus -> intrinsic dim. = 6, distance func. = wrapping Manhattan
        HYPERCUBE, // N-D hypercube -> intrinsic dim. = 1, distance func. = Hamming
        HYPERX3D // 3D HyperX -> intrinsic dim. = 3, distance func. = # not-equal dim. (any-radix Hamming)
    };

    static constexpr std::pair<TargetTopology, const char*> TOPOLOGY_NAMES[] = {
        { TargetTopology::LATTICE2D, "lat2d" },
        { TargetTopology::TORUS6D, "tor6d" },
        { TargetTopology::HYPERCUBE, "hcube" },
        { TargetTopology::HYPERX3D, "hx3d" }
    };

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

    using TopologySupport = uint32_t; // flag bits -> the i-th bit is set if the i-th topology in TargetTopology's order is supported

    static constexpr TopologySupport topologySupport(TargetTopology topology) noexcept {
        return TopologySupport{1} << static_cast<uint32_t>(topology);
    }

    static constexpr std::pair<SpaceFillingCurve, TopologySupport> CURVE_TOPOLOGY_SUPPORT[] = {
        { SpaceFillingCurve::HILB, topologySupport(TargetTopology::LATTICE2D) | topologySupport(TargetTopology::TORUS6D) },
        { SpaceFillingCurve::SNAK, topologySupport(TargetTopology::LATTICE2D) | topologySupport(TargetTopology::TORUS6D) },
        { SpaceFillingCurve::ZORD, topologySupport(TargetTopology::LATTICE2D) | topologySupport(TargetTopology::TORUS6D) },
        { SpaceFillingCurve::QUAD, topologySupport(TargetTopology::LATTICE2D) | topologySupport(TargetTopology::TORUS6D) }
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
        TargetTopology topology; // target graph topology, how places are interconnected
        SpaceFillingCurve space_filling_curve; // space filling curve to use for the 1D-to-(N)D locality-preserving mapping
        bool feedforward_order; // if true, use the greedy sequential feedforward initial partitioning (runs on the host !!)
        bool unicast_metrics; // if true, compute and log the unicast-based placement quality metrics
        bool multicast_metrics; // if true, compute and log the multicast-based (Steiner tree involved!!) placement quality metrics
        bool device_touching_construction; // whether to construct touching/incidence sets on the device or the host
        uint64_t seed; // seed for the multi-start and recursive bisection methods
        bool verbose_logs; // whether to log what is happening inside the algorithms
        bool verbose_info; // whether to log the step/phase where the program is at
        bool verbose_errs_and_warns; // whether to log errs and warnings
        bool verbose_kernel_launches; // whether to log every kernel launch or not
        bool debug; // whether to run extra debug synchronization/checks
    };

    void printHelp();

    runconfig parseArgs(int argc, char** argv);

    hgraph::HyperGraph loadHgraph(runconfig &cfg);

    template<topology::Topology T>
    hwmodel::HardwareModel<T> setupNMH(runconfig &cfg);

    template<topology::Topology T>
    void saveResult(runconfig &cfg, std::vector<topology::Coord_t<T>> h_placement);

    const char* topologyToString(TargetTopology topology);

    bool parseTopology(const std::string& name, TargetTopology& topology);

    const char* SFCtoString(SpaceFillingCurve curve);

    bool parseSFC(const std::string& name, SpaceFillingCurve& curve);

    bool validateTopologySFC(TargetTopology topology, SpaceFillingCurve curve);
}
