#pragma once
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

#include "curves.hpp"
#include "topology.hpp"
#include "nmhardware.hpp"

namespace hgraph {
    class HyperGraph;
}

namespace config_plc {

    struct runconfig {
        std::string load_path; // path to the hgraph to load 'n' partition
        std::string graph_path; // path to the target topology graph to load (mandatory when topology == ARBITRARY)
        std::string save_path; // path where to save the placement data
        std::string constraints; // name the constraints set to use
        uint32_t labelprop_repeats; // number of labelprop rounds performed at each level of recursive bisection in the parallel initial placement
        uint32_t fd_iterations; // number of force-directed refinement iterations to perform
        uint32_t candidates_count; // number of candidate swaps proposed per node during force-directed refinement
        uint32_t multi_start_override; // imposes the number of multi-start attempts at placement
        uint32_t batch_size; // imposes how many multi-start attempts are refined together, by one call of every kernel
        uint32_t threads; // number of OpenMP threads to use
        topology::TargetTopology topology; // target graph topology, how places are interconnected
        curve::SpaceFillingCurve space_filling_curve; // space filling curve to use for the 1D-to-(N)D locality-preserving mapping
        bool feedforward_order; // if true, use the greedy sequential feedforward initial partitioning (runs sequentially !!)
        // ROUTING POLICIES: which cost models to evaluate the placement under (see -rp)
        bool unicast_metrics; // if true, compute and log the unicast placement quality metrics
        bool xy_multicast_metrics; // if true, compute and log the XY-multicast placement quality metrics
        bool steiner_multicast_metrics; // if true, compute and log the Steiner-multicast placement quality metrics (very slow!!)
        bool parallel_touching_construction; // whether to construct touching/incidence sets in parallel or sequentially while loading
        uint64_t seed; // seed for the multi-start and recursive bisection methods
        bool verbose_logs; // whether to log what is happening inside the algorithms
        bool verbose_info; // whether to log the step/phase where the program is at
        bool verbose_errs_and_warns; // whether to log errs and warnings
        bool verbose_kernel_launches; // whether to log every parallel loop or not
        bool debug; // whether to run extra debug checks
    };

    void printHelp();

    runconfig parseArgs(int argc, char** argv);

    hgraph::HyperGraph loadHgraph(runconfig &cfg);

    hgraph::HyperGraph loadTopologyGraph(runconfig &cfg);

    template<topology::Topology T>
    hwmodel::HardwareModel<T> setupNMH(runconfig &cfg);

    template<topology::Topology T>
    void saveResult(runconfig &cfg, std::vector<topology::Coord_t<T>> placement);

    const char* topologyToString(topology::TargetTopology topology);

    bool parseTopology(const std::string& name, topology::TargetTopology& topology);

    const char* SFCtoString(curve::SpaceFillingCurve curve);

    bool parseSFC(const std::string& name, curve::SpaceFillingCurve& curve);

    // parse a comma-separated list of routing policy names (unicast, xy, steiner) into the three flags
    bool parseRoutingPolicies(const std::string& list, bool& unicast, bool& xy_multicast, bool& steiner_multicast);

    // render the enabled routing policies back as a comma-separated list
    std::string routingPoliciesToString(bool unicast, bool xy_multicast, bool steiner_multicast);

    bool validateTopologySFC(topology::TargetTopology topology, curve::SpaceFillingCurve curve);
}
