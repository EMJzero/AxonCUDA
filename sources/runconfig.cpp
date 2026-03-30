#include <string>
#include <limits>
#include <cfloat>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "hgraph.hpp"
#include "constr.hpp"

#include "defines.cuh"

#include "runconfig.hpp"

using namespace hgraph;
using namespace constraints;

void printHelp() {
    std::cout <<
        "Usage:\n"
        "  prog -r <input_file> [-c <constr>] [-s <output_file>] [-p <part_file>]\n"
        "  prog -r <input_file> [-k <k> <ε>] [-s <output_file>] [-p <part_file>]\n"
        "  prog -h\n\n"
        "Options:\n"
        "  -r <file>   Reload hypergraph from file\n"
        "  -s <file>   Save partitioned hypergraph to file\n"
        "  -p <file>   Save the partitioning to file (one line per node, containing its partition id)\n"
        "  -c <name>   Preconfigured constraints set to use (valid ones: truenorth, loihi64, loihi84, loihi1024 - default is loihi64)\n"
        "  -m <> <> <> Constraints set to use, in order: max part. size, max part. distinct inbound hedges, max num. of part.s (overrides '-c')\n"
        "  -k <k> <ε>  K-way balanced constraints set to use (overrides '-c' and '-m')\n"
        "  -om <mult>  Set the deduplication oversized segment size multiplier (increase to avoid the 'GM hash-set full!' assert)\n"
        //"  -som        Set the deduplication segment size to the sum of the merged set sizes (set to avoid the 'GM hash-set full!' assert)\n"
        "  -cnc <num>  Set the count of candidates proposed per node during coarsening\n"
        "  -rfr <num>  Set the number of refinement repetitions per level\n"
        "  -dtc        When set, construct touching sets on the device, rather than on the host\n"
        "  -h          Show this help message\n";
}

runconfig parseArgs(int argc, char** argv) {
    // defaults
    std::string load_path;
    std::string save_path;
    std::string part_path;
    Mode mode = Mode::INCC;
    ConstrType constr_type = ConstrType::NONE;
    ConstraintsConfig constr_config;
    uint32_t kway = UINT32_MAX;
    float epsi = FLT_MAX;
    float oversized_multiplier = OVERSIZED_SIZE_MULTIPLIER;
    uint32_t candidates_count = MAX_CANDIDATES;
    uint32_t refine_repeats = REFINE_REPEATS;
    bool device_touching_construction = false;

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
        } else if (arg == "-p") {
            if (i + 1 >= argc) { std::cerr << "Error: -p requires a file path\n"; std::exit(1); }
            part_path = argv[++i];
        } else if (arg == "-c") {
            if (i + 1 >= argc) { std::cerr << "Error: -c requires a config name\n"; std::exit(1); }
            constr_type = ConstrType::NAME;
            constr_config.name = argv[++i];
            mode = Mode::INCC;
        } else if (arg == "-m") {
            if (i + 3 >= argc) { std::cerr << "Error: -m requires integer values for the three constraints\n"; std::exit(1); }
            constr_type = ConstrType::MANL;
            constr_config.name = "manual";
            constr_config.nodes_per_part = std::stoul(argv[++i]);
            constr_config.inbound_per_part = std::stoul(argv[++i]);
            constr_config.max_parts = std::stoul(argv[++i]);
            mode = Mode::INCC;
        } else if (arg == "-k") {
            if (i + 2 >= argc) { std::cerr << "Error: -k requires values for 'k' and 'ε'\n"; std::exit(1); }
            constr_type = ConstrType::KWAY;
            kway = std::stoul(argv[++i]);
            epsi = std::stof(argv[++i]);
            mode = Mode::KWAY;
        } else if (arg == "-om") {
            if (i + 1 >= argc) { std::cerr << "Error: -om requires a float value\n"; std::exit(1); }
            oversized_multiplier = std::stof(argv[++i]);
        } else if (arg == "-cnc") {
            if (i + 1 >= argc) { std::cerr << "Error: -cnc requires a positive integer value\n"; std::exit(1); }
            candidates_count = std::stoul(argv[++i]);
            if (candidates_count > MAX_CANDIDATES) { std::cerr << "Error: -cnc must be less or equal to " << MAX_CANDIDATES << "\n"; std::exit(1); }
        } else if (arg == "-rfr") {
            if (i + 1 >= argc) { std::cerr << "Error: -rfr requires a positive integer value\n"; std::exit(1); }
            refine_repeats = std::stoul(argv[++i]);
        } else if (arg == "-dtc") {
            device_touching_construction = true;
        } else { std::cerr << "Unknown option: " << arg << "\n"; std::exit(1); }
    }
    assert((mode == Mode::KWAY && constr_type == ConstrType::KWAY) || (mode != Mode::KWAY && constr_type != ConstrType::KWAY));

    return {
        load_path,
        save_path,
        part_path,
        mode,
        constr_type,
        constr_config,
        kway,
        epsi,
        oversized_multiplier,
        candidates_count,
        refine_repeats,
        device_touching_construction
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

Constraints setupConstr(runconfig cfg, HyperGraph hg) {
    if (cfg.constr_type == ConstrType::KWAY) { // k-way mode ('-k')
        std::ostringstream epsistr;
        epsistr << std::fixed << std::setprecision(3) << cfg.epsi;
        cfg.constr_config.name = std::to_string(cfg.kway) + "-way " + epsistr.str() + " balanced";
        cfg.constr_config.nodes_per_part = (uint32_t)std::ceil((1 + cfg.epsi)*(float)hg.nodes()/cfg.kway);
        cfg.constr_config.inbound_per_part = INT32_MAX;
        cfg.constr_config.max_parts = cfg.kway;
        return Constraints(cfg.constr_config);
    } else if (cfg.constr_type == ConstrType::MANL) { // manual constraints ('-m')
        if (cfg.constr_config.nodes_per_part == 0) { std::cerr << "Error: the 1st constraint (max partition size) must be a positive integer \n"; std::exit(1); }
        if (cfg.constr_config.inbound_per_part == 0) { std::cerr << "Error: the 2nd constraint (max distinct inbound hedge per partition) must be a positive integer \n"; std::exit(1); }
        if (cfg.constr_config.max_parts == 0) { std::cerr << "Error: the 3rd constraint (max number of partitions) must be a positive integer \n"; std::exit(1); }
        return Constraints(cfg.constr_config);
    } else if (cfg.constr_type == ConstrType::NAME) { // preconfigured constraints ('-c')
        std::unordered_map<std::string, Constraints (*)()> configurations {
            { "loihi64", Constraints::createLoihiLarge },
            { "loihi84", Constraints::createLoihiJin84 },
            { "loihi1024", Constraints::createLoihiJin1024 },
            { "truenorth", Constraints::createTrueNorth }
        };
        auto constr_it = configurations.find(cfg.constr_config.name);
        return constr_it->second();
    } else { // no (valid) constraints provided
        std::cerr << "WARNING, no constraints provided (-c, -m, -k), using loihi64 !!\n";
        return Constraints::createLoihiLarge();
    }
}

void saveResult(runconfig cfg, HyperGraph partitioned_hg, std::vector<uint32_t> partitions) {
    // save hypergraph
    if (!cfg.save_path.empty()) {
        if (cfg.load_path.empty()) {
            std::cerr << "Error: -s used without loading a hypergraph first.\n";
            std::exit(1);
        }
        try {
            std::filesystem::path file_path(cfg.save_path);
            if (file_path.extension() == ".hgr") {
                std::cout << "Saving partitioned hypergraph to: " << cfg.save_path << " (hMETIS format) ...\n";
                partitioned_hg.savehMETIS(cfg.save_path);
            } else if (file_path.extension() == ".snn") {
                std::cout << "Saving partitioned hypergraph to: " << cfg.save_path << " (SNN format) ...\n";
                partitioned_hg.saveSNN(cfg.save_path);
            } else if (file_path.extension() == ".axh") {
                std::cout << "Saving partitioned hypergraph to: " << cfg.save_path << " (AXH format) ...\n";
                partitioned_hg.saveAXH(cfg.save_path);
            } else {
                throw std::runtime_error("Failed to save partitioned hypergraph, unsupported file format (supported: '.hgr', '.snn', '.axh').");
            }
            std::cout << "Partitioned hypergraph saved to " << cfg.save_path << "\n";
            std::cout << "Partitioned hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(cfg.save_path)) / (1 << 20) << " MB\n";
        } catch (const std::exception& e) {
            std::cerr << "Error saving file: " << e.what() << "\n";
            std::exit(1);
        }
    }

    // save partitioning
    if (!cfg.part_path.empty()) {
        try {
            std::cout << "Saving partitioning to: " << cfg.part_path << " (each node's partition id on its line by node idx) ...\n";
            std::ofstream f(cfg.part_path);
            if (!f) throw std::runtime_error("Cannot open output file");
            for (const auto& p : partitions)
                f << p << "\n";
            std::cout << "Partitioning saved to " << cfg.part_path << "\n";
            std::cout << "Partitioning file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(cfg.part_path)) / (1 << 20) << " MB\n";
        } catch (const std::exception& e) {
            std::cerr << "Error saving file: " << e.what() << "\n";
            std::exit(1);
        }
    }
}
