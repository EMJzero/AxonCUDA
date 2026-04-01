#include <string>
#include <limits>
#include <cfloat>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>

#include "hgraph.hpp"
#include "nmhardware.hpp"

#include "defines_plc.cuh"

#include "runconfig_plc.hpp"

using namespace hgraph;
using namespace hwmodel;
using namespace hwgeom;

namespace config_plc {

    void printHelp() {
        std::cout <<
            "Usage:\n"
            "  prog -r <input_file> [-s <output_file>]\n"
            "  prog -h\n\n"
            "Options:\n"
            "  -r <file>   Reload hypergraph from file\n"
            "  -s <file>   Save placement data to file\n"
            "  -c <name>   Constraints set to use (valid ones: truenorth, loihi64, loihi84, loihi1024 - default is loihi64)\n"
            "  -lpr <num>  Set the number of label propagation repeats during recursive bisection initial partitioning\n"
            "  -fdi <num>  Set the number of force-directed refinement iterations\n"
            "  -cnc <num>  Set the count of candidate swaps proposed per node during force-directed refinement\n"
            "  -ff         Replaces the 1D ordering heuristic with host-side sequential feedforward ordering\n"
            "  -dtc        When set, construct touching sets on the device, rather than on the host\n"
            "  -seed <num> Set the algorithm's seed to <num> (default: " << SEED << ") (ignored when '-ff' is passed)\n"
            "  -h          Show this help\n";
    }

    runconfig parseArgs(int argc, char** argv) {
        // defaults
        std::string load_path;
        std::string save_path;
        std::string constraints;
        uint32_t labelprop_repeats = LABELPROP_REPEATS;
        uint32_t fd_iterations = FD_ITERATIONS;
        uint32_t candidates_count = MAX_CANDIDATE_MOVES;
        bool feedforward_order = false; // NB: runs sequentially on the HOST!
        bool device_touching_construction = false;
        uint64_t seed = SEED;

        // CLI handling
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-h") { printHelp(); std::exit(0); }
            else if (arg == "-r") {
                if (i + 1 >= argc) { std::cerr << "Error: -r requires a file path\n"; std::exit(1); }
                load_path = argv[++i];
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
            } else if (arg == "-cnc") {
                if (i + 1 >= argc) { std::cerr << "Error: -cnc requires a positive integer value\n"; std::exit(1); }
                candidates_count = std::stoul(argv[++i]);
                if (candidates_count > MAX_CANDIDATE_MOVES) { std::cerr << "Error: -cnc must be less or equal to " << MAX_CANDIDATE_MOVES << "\n"; std::exit(1); }
            } else if (arg == "-ff") {
                feedforward_order = true;
            } else if (arg == "-dtc") {
                device_touching_construction = true;
            } else if (arg == "-seed") {
                if (i + 1 >= argc) { std::cerr << "Error: -seed requires a positive integer value\n"; std::exit(1); }
                seed = std::stoull(argv[++i]);
            } else { std::cerr << "Unknown option: " << arg << "\n"; std::exit(1); }
        }

        return {
            load_path,
            save_path,
            constraints,
            labelprop_repeats,
            fd_iterations,
            candidates_count,
            feedforward_order,
            device_touching_construction,
            seed
        };
    }

    HyperGraph loadHgraph(runconfig cfg) {
        HyperGraph hg(0, {}, {}); // placeholder -> overwritten if "-r" is given

        if (!cfg.load_path.empty()) {
            try {
                if (!std::filesystem::is_regular_file(cfg.load_path)) throw std::runtime_error("Failed to load hypergraph, the provided path is not a file.");
                std::filesystem::path file_path(cfg.load_path);
                if (file_path.extension() == ".hgr") {
                    std::cout << "Loading hypergraph from: " << cfg.load_path << " (hMETIS format) ...\n";
                    std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(cfg.load_path)) / (1 << 20) << " MB\n";
                    hg = HyperGraph::loadhMETIS(cfg.load_path);
                } else if (file_path.extension() == ".snn") {
                    std::cout << "Loading hypergraph from: " << cfg.load_path << " (SNN format) ...\n";
                    std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(cfg.load_path)) / (1 << 20) << " MB\n";
                    hg = HyperGraph::loadSNN(cfg.load_path);
                } else if (file_path.extension() == ".axh") {
                    std::cout << "Loading hypergraph from: " << cfg.load_path << " (AXH format) ...\n";
                    std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(cfg.load_path)) / (1 << 20) << " MB\n";
                    hg = HyperGraph::loadAXH(cfg.load_path);
                } else {
                    throw std::runtime_error("Failed to load hypergraph, unsupported file format (supported: '.hgr', '.snn', '.axh').");
                }
            } catch (const std::exception& e) {
                std::cerr << "Error loading file: " << e.what() << "\n";
                std::exit(1);
            }
        } else {
            std::cerr << "WARNING, no hypergraph provided (-r), performing a dry-run !!\n";
        }

        return hg;
    }

    HardwareModel setupNMH(runconfig cfg) {
        std::unordered_map<std::string, HardwareModel (*)()> configurations {
            { "loihi64", HardwareModel::createLoihiLarge },
            { "loihi84", HardwareModel::createLoihiJin84 },
            { "loihi1024", HardwareModel::createLoihiJin1024 },
            { "truenorth", HardwareModel::createTrueNorth }
        };
        auto hw_it = configurations.find(cfg.constraints);
        if (hw_it == configurations.end()) {
            std::cerr << "WARNING, no valid constraints provided (-c), using loihi64 !!\n";
            return HardwareModel::createLoihiLarge();
        }
        return hw_it->second();
    }

    void saveResult(runconfig cfg, std::vector<Coord2D> h_placement) {
        // save hypergraph
        if (!cfg.save_path.empty()) {
            if (cfg.load_path.empty()) {
                std::cerr << "Error: -s used without loading a hypergraph first.\n";
                std::exit(1);
            }
            try {
                // TODO: apply the partitioning before saving!
                coords_to_file(h_placement, cfg.save_path);
                std::cout << "Placement data saved to " << cfg.save_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error saving file: " << e.what() << "\n";
                std::exit(1);
            }
        }
    }
}