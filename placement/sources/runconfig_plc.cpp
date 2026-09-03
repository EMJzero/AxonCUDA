#include <string>
#include <limits>
#include <cfloat>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <utility>
#include <iostream>
#include <filesystem>
#include <unordered_map>

#include "hgraph.hpp"
#include "topology.hpp"
#include "nmhardware.hpp"

#include "defines_plc.cuh"

#include "runconfig_plc.hpp"

using namespace hgraph;
using namespace hwmodel;
using namespace topology;

namespace config_plc {

    void printHelp() {
        std::cout <<
            "Usage:\n"
            "  prog -r <input_file> [-s <output_file>]\n"
            "  prog -h\n\n"
            "Options:\n"
            "  -r <file>   Reload hypergraph from file\n"
            "  -g <file>   Load the target topology graph from file (mandatory when -t arb is used)\n"
            "  -s <file>   Save placement data to file\n"
            "  -c <name>   Constraints set to use (valid ones: truenorth, loihi, loihi64, loihi84, loihi1024 - default is loihi64)\n"
            "  -lpr <num>  Set the number of label propagation repeats during recursive bisection initial partitioning\n"
            "  -fdi <num>  Set the number of force-directed refinement iterations\n"
            "  -cnc <num>  Set the count of candidate swaps proposed per node during force-directed refinement\n"
            "  -mso <num>  Overrides the number of multi-start attempts (default is chosen to maximally occupy the GPU)\n"
            "  -thr <num>  Overrides the number of threads and streams to spawn (default equals multi-start attempts)\n"
            "  -t <name>   Set the topology of the target graph where to place hypergraph nodes, valid names are:\n"
            "      - lat2d: 2D lattice (default)    - tor6d: 6D torus\n"
            "      - hcube: N-D hypercube           - hx3d: 3D HyperX\n"
            "      - arb: arbitrary weighted graph (requires -g)\n"
            "  -sfc <name> Determines the 1D-to-2D mapping used to preserve locality after ordering nodes, valid names are:\n"
            "      - hilb: Hilbert space-filling curve (default)    - snak: square-ish serpentine sweep\n"
            "      - zord: Z-order curve                            - quad: cyclic quadtree pattern\n"
            "      - flat: identity 1D mapping (default, and only supported option, for arb)\n"
            "    => sfc x topology support lists:\n"
            "      - hilb, snak, zord, quad: lattice, torus\n"
            "      - flat: arbitrary\n"
            "  -ff         Replaces the 1D ordering heuristic with host-side sequential feedforward ordering\n"
            "  -noum       Disables the evaluation and logging of unicast-based placement quality metrics\n"
            "  -nomm       Disables the evaluation and logging of multicast-based placement quality metrics (which can be very slow)\n"
            "  -dtc        When set, construct touching sets on the device, rather than on the host\n"
            "  -seed <num> Set the algorithm's seed to <num> (default: " << SEED << ") (ignored when '-ff' is passed)\n"
            "  -v <lvl>    Set the verbosity level: 0 results only, 1 steps and phases, 2 kernel launches, 3 algorithm outputs, 4 debug \n"
            "  -h          Show this help\n";
    }

    runconfig parseArgs(int argc, char** argv) {
        // defaults
        std::string load_path;
        std::string graph_path;
        std::string save_path;
        std::string constraints;
        uint32_t labelprop_repeats = LABELPROP_REPEATS;
        uint32_t fd_iterations = FD_ITERATIONS;
        uint32_t candidates_count = CANDIDATE_MOVES;
        uint32_t multi_start_override = MULTISTART_ATTEMPTS;
        uint32_t num_host_threads = NUM_HOST_THREADS;
        TargetTopology topology = TargetTopology::LATTICE2D;
        SpaceFillingCurve space_filling_curve = SpaceFillingCurve::HILB;
        bool space_filling_curve_explicit = false;
        bool feedforward_order = false; // NB: runs sequentially on the HOST!
        bool unicast_metrics = true;
        bool multicast_metrics = true;
        bool device_touching_construction = false;
        uint64_t seed = SEED;
        bool verbose_logs = VERBOSE_LOGS;
        bool verbose_info = VERBOSE_INFO;
        bool verbose_errs_and_warns = VERBOSE_ERRS;
        bool verbose_kernel_launches = VERBOSE_LAUNCHES;
        bool debug = DEBUG_ON;

        // CLI handling
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-h") { printHelp(); std::exit(0); }
            else if (arg == "-r") {
                if (i + 1 >= argc) { std::cerr << "Error: -r requires a file path\n"; std::exit(1); }
                load_path = argv[++i];
            } else if (arg == "-g") {
                if (i + 1 >= argc) { std::cerr << "Error: -g requires a file path\n"; std::exit(1); }
                graph_path = argv[++i];
            } else if (arg == "-s") {
                if (i + 1 >= argc) { std::cerr << "Error: -s requires a file path\n"; std::exit(1); }
                save_path = argv[++i];
            } else if (arg == "-c") {
                if (i + 1 >= argc) { std::cerr << "Error: -c requires a config name\n"; std::exit(1); }
                constraints = argv[++i];
            } else if (arg == "-lpr") {
                if (i + 1 >= argc) { std::cerr << "Error: -lpr requires a positive integer value\n"; std::exit(1); }
                labelprop_repeats = std::stoul(argv[++i]);
            } else if (arg == "-fdi") {
                if (i + 1 >= argc) { std::cerr << "Error: -fdi requires a positive integer value\n"; std::exit(1); }
                fd_iterations = std::stoul(argv[++i]);
            } else if (arg == "-mso") {
                if (i + 1 >= argc) { std::cerr << "Error: -mso requires a positive integer value\n"; std::exit(1); }
                multi_start_override = std::stoul(argv[++i]);
                if (multi_start_override == 0) { std::cerr << "Error: -mso must greater than zero\n"; std::exit(1); }
            } else if (arg == "-thr") {
                if (i + 1 >= argc) { std::cerr << "Error: -thr requires a positive integer value\n"; std::exit(1); }
                num_host_threads = std::stoul(argv[++i]);
                if (num_host_threads == 0) { std::cerr << "Error: -thr must greater than zero\n"; std::exit(1); }
            } else if (arg == "-cnc") {
                if (i + 1 >= argc) { std::cerr << "Error: -cnc requires a positive integer value\n"; std::exit(1); }
                candidates_count = std::stoul(argv[++i]);
                if (candidates_count == 0) { std::cerr << "Error: -cnc must be greater than zero\n"; std::exit(1); }
            } else if (arg == "-t") {
                if (i + 1 >= argc) { std::cerr << "Error: -t requires a topology name\n"; std::exit(1); }
                std::string topo_name = argv[++i];
                if (!parseTopology(topo_name, topology)) { std::cerr << "Error: -t requested an invalid topology name\n"; std::exit(1); }
            } else if (arg == "-sfc") {
                if (i + 1 >= argc) { std::cerr << "Error: -sfc requires a curve name\n"; std::exit(1); }
                std::string curve_name = argv[++i];
                if (!parseSFC(curve_name, space_filling_curve)) { std::cerr << "Error: -sfc requested an invalid curve name\n"; std::exit(1); }
                space_filling_curve_explicit = true;
            } else if (arg == "-ff") {
                feedforward_order = true;
            } else if (arg == "-noum") {
                unicast_metrics = false;
            } else if (arg == "-nomm") {
                multicast_metrics = false;
            } else if (arg == "-dtc") {
                device_touching_construction = true;
            } else if (arg == "-seed") {
                if (i + 1 >= argc) { std::cerr << "Error: -seed requires a positive integer value\n"; std::exit(1); }
                seed = std::stoull(argv[++i]);
            } else if (arg == "-v") {
                if (i + 1 >= argc) { std::cerr << "Error: -v requires a positive value between 0 and 4\n"; std::exit(1); }
                int verbosity = std::stoul(argv[++i]);
                if (verbosity < 0 || verbosity > 4) { std::cerr << "Error: -v must be between 0 and 4 (extremes included)\n"; std::exit(1); }
                verbose_logs = verbosity > 2;
                verbose_info = verbosity > 0;
                verbose_errs_and_warns = verbosity > 0;
                verbose_kernel_launches = verbosity > 1;
                debug = verbosity > 3;
                if (verbosity > 2) std::cerr << "WARNING: verbosity 3 and 4 can hinder performance, especially on the host side !!\n";
            } else { std::cerr << "Unknown option: " << arg << "\n"; std::exit(1); }
        }

        if (topology == TargetTopology::ARBITRARY && graph_path.empty()) {
            std::cerr << "Error: topology 'arb' requires a target topology graph file (-g)\n";
            std::exit(1);
        }

        if (!space_filling_curve_explicit && topology == TargetTopology::ARBITRARY) {
            // arbitrary topologies have no embeddable multi-dimensional locality curve -> default to the identity mapping
            space_filling_curve = SpaceFillingCurve::FLAT;
        }

        if (!validateTopologySFC(topology, space_filling_curve)) {
            std::cerr << "Error: space filling curve '" << SFCtoString(space_filling_curve) << "' does not support topology '" << topologyToString(topology) << "'\n";
            std::exit(1);
        }

        return {
            load_path,
            graph_path,
            save_path,
            constraints,
            labelprop_repeats,
            fd_iterations,
            candidates_count,
            multi_start_override,
            num_host_threads,
            topology,
            space_filling_curve,
            feedforward_order,
            unicast_metrics,
            multicast_metrics,
            device_touching_construction,
            seed,
            verbose_logs,
            verbose_info,
            verbose_errs_and_warns,
            verbose_kernel_launches,
            debug
        };
    }

    // shared by loadHgraph (-r) and loadTopologyGraph (-g): both are plain HyperGraph files, dispatched by extension
    static HyperGraph loadHyperGraphFile(const std::string& path, bool verbose) {
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("Failed to load hypergraph, the provided path is not a file.");
        std::filesystem::path file_path(path);
        std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(path)) / (1 << 20) << " MB\n";
        if (file_path.extension() == ".hgr") {
            std::cout << "Loading hypergraph from: " << path << " (hMETIS format) ...\n";
            return HyperGraph::loadhMETIS(path, verbose);
        } else if (file_path.extension() == ".snn") {
            std::cout << "Loading hypergraph from: " << path << " (SNN format) ...\n";
            return HyperGraph::loadSNN(path, verbose);
        } else if (file_path.extension() == ".axh") {
            std::cout << "Loading hypergraph from: " << path << " (AXH format) ...\n";
            return HyperGraph::loadAXH(path, verbose);
        } else {
            throw std::runtime_error("Failed to load hypergraph, unsupported file format (supported: '.hgr', '.snn', '.axh').");
        }
    }

    HyperGraph loadHgraph(runconfig &cfg) {
        HyperGraph hg(0, {}, {}); // placeholder -> overwritten if "-r" is given

        if (!cfg.load_path.empty()) {
            try {
                hg = loadHyperGraphFile(cfg.load_path, cfg.verbose_errs_and_warns);
            } catch (const std::exception& e) {
                std::cerr << "Error loading file: " << e.what() << "\n";
                std::exit(1);
            }
        } else {
            std::cerr << "WARNING, no hypergraph provided (-r), performing a dry-run !!\n";
        }

        return hg;
    }

    HyperGraph loadTopologyGraph(runconfig &cfg) {
        try {
            return loadHyperGraphFile(cfg.graph_path, cfg.verbose_errs_and_warns);
        } catch (const std::exception& e) {
            std::cerr << "Error loading topology graph file: " << e.what() << "\n";
            std::exit(1);
        }
    }

    template<Topology T>
    HardwareModel<T> setupNMH(runconfig &cfg) {
        using Model = HardwareModel<T>;

        if constexpr (requires (const HyperGraph& hg) { { T::fromGraph(hg) } -> std::same_as<T>; }) {
            // topologies built from an explicit graph file (-g), e.g. Arbitrary: costs are still picked by
            // name (-c), reusing the same named profiles as the grid-based topologies, minus their grid extent
            static const std::unordered_map<std::string, HardwareModelConfig> profiles {
                { "loihi", { "Loihi", 1024, 4096, 1.7, 3.5, 2.1, 5.3 } },
                { "loihi64", { "Loihi Large", 1024, 4096, 1.7, 3.5, 2.1, 5.3 } },
                { "loihi84", { "Loihi Jin 84", 4096, 1024*64, 1.0, 0.1, 1.0, 0.01 } },
                { "loihi1024", { "Loihi Jin 1024", 4096, 1024*64, 1.0, 0.1, 1.0, 0.01 } },
                { "truenorth", { "TrueNorth", 256, 256, 1.7, 3.5, 2.1, 5.3 } }
            };
            auto cfg_it = profiles.find(cfg.constraints);
            HardwareModelConfig hwcfg;
            if (cfg_it == profiles.end()) {
                std::cerr << "WARNING, no valid constraints provided (-c), using loihi64 !!\n";
                hwcfg = profiles.at("loihi64");
            } else {
                hwcfg = cfg_it->second;
            }
            HyperGraph topology_graph = loadTopologyGraph(cfg);
            try {
                return Model::createFromGraph(topology_graph, hwcfg);
            } catch (const std::exception& e) {
                std::cerr << "Error building topology from graph: " << e.what() << "\n";
                std::exit(1);
            }
        } else {
            std::unordered_map<std::string, Model (*)()> configurations {
                { "loihi", Model::createLoihi },
                { "loihi64", Model::createLoihiLarge },
                { "loihi84", Model::createLoihiJin84 },
                { "loihi1024", Model::createLoihiJin1024 },
                { "truenorth", Model::createTrueNorth }
            };
            auto hw_it = configurations.find(cfg.constraints);
            if (hw_it == configurations.end()) {
                std::cerr << "WARNING, no valid constraints provided (-c), using loihi64 !!\n";
                return Model::createLoihiLarge();
            }
            return hw_it->second();
        }
    }

    template<Topology T>
    void saveResult(runconfig &cfg, std::vector<Coord_t<T>> h_placement) {
        // save hypergraph
        if (!cfg.save_path.empty()) {
            if (cfg.load_path.empty()) {
                std::cerr << "Error: -s used without loading a hypergraph first.\n";
                std::exit(1);
            }
            try {
                T::Coord::toFile(h_placement, cfg.save_path);
                std::cout << "Placement data saved to " << cfg.save_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error saving file: " << e.what() << "\n";
                std::exit(1);
            }
        }
    }

    const char* topologyToString(TargetTopology topology) {
        for (const auto& [value, name] : TOPOLOGY_NAMES) {
            if (value == topology)
                return name;
        }
        return "unknown";
    }

    bool parseTopology(const std::string& name, TargetTopology& topology) {
        for (const auto& [value, text] : TOPOLOGY_NAMES) {
            if (name == text) {
                topology = value;
                return true;
            }
        }
        return false;
    }

    const char* SFCtoString(SpaceFillingCurve curve) {
        for (const auto& [value, name] : SPACE_FILLING_CURVE_NAMES) {
            if (value == curve)
                return name;
        }
        return "unknown";
    }

    bool parseSFC(const std::string& name, SpaceFillingCurve& curve) {
        for (const auto& [value, text] : SPACE_FILLING_CURVE_NAMES) {
            if (name == text) {
                curve = value;
                return true;
            }
        }
        return false;
    }

    bool validateTopologySFC(TargetTopology topology, SpaceFillingCurve curve) {
        for (const auto& [c, supported_topologies] : CURVE_TOPOLOGY_SUPPORT) {
            if (curve == c) {
                return (supported_topologies & topologySupport(topology)) != 0;
            }
        }
        return false;
    }

    // explicit instantiations of topology variants
    template HardwareModel<Lattice2D> setupNMH<Lattice2D>(runconfig&);
    template HardwareModel<Torus6D> setupNMH<Torus6D>(runconfig&);
    template HardwareModel<ArbitraryGraph> setupNMH<ArbitraryGraph>(runconfig&);
    // |
    template void saveResult<Lattice2D>(runconfig&, std::vector<Lattice2D::Coord>);
    template void saveResult<Torus6D>(runconfig&, std::vector<Torus6D::Coord>);
    template void saveResult<ArbitraryGraph>(runconfig&, std::vector<ArbitraryGraph::Coord>);
}
