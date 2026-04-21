#include <tuple>
#include <string>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <iostream>
#include <optional>
#include <sstream>
#include <filesystem>
#include <unordered_map>

#include "hgraph.hpp"
#include "constr.hpp"
#include "nmhardware.hpp"

using namespace hgraph;
using namespace constraints;
using namespace hwmodel;
using namespace hwgeom;


enum class PartConstrType {
    KWAY, // k-way balanced constraints
    NAME, // named constraints configuration (incidence constraints)
    MANL, // manual constraints configuration (incidence constraints)
    NONE  // no constraints provided -> use default
};

void printHelp() {
    std::cout <<
        "Usage:\n"
        "  prog -r <hgraph_file> -prt <partitioning_file> [-plc <placement_file>] [-s <partitioned_hgraph_output_file>]\n"
        "  prog -r <partitioned_hgraph_file> -plc <placement_file>\n"
        "  prog -h\n\n"
        "Options:\n"
        "  -r <file>   Read hypergraph from file\n"
        "  -prt <file> Read partitioning data from file\n"
        "  -plc <file> Read placement data from file\n"
        "  -s <file>   Save partitioned hypergraph to file\n"
        "  -c-prt <name>   Partitioning constraints set to use (valid ones: truenorth, loihi64, loihi84, loihi1024)\n"
        "  -m-prt <> <> <> Partitioning constraints set to use, in order: max part. size, max part. distinct inbound hedges, max num. of part.s (overrides '-c-prt')\n"
        "  -k-prt <k> <ε>  K-way balanced constraints set to use (overrides '-c-prt' and '-m-prt')\n"
        "  -c-plc <name>   Placement constraints set to use (valid ones: truenorth, loihi, loihi64, loihi84, loihi1024)\n"
        "  -h          Show this help\n";
}

HyperGraph loadHgraph(std::string load_path) {
    HyperGraph hg(0, {}, {}); // placeholder -> overwritten if "-r" is given

    if (!load_path.empty()) {
        try {
            if (!std::filesystem::is_regular_file(load_path)) throw std::runtime_error("Failed to load hypergraph, the provided path is not a file.");
            std::filesystem::path file_path(load_path);
            if (file_path.extension() == ".hgr") {
                std::cout << "Loading hypergraph from: " << load_path << " (hMETIS format) ...\n";
                std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(load_path)) / (1 << 20) << " MB\n";
                hg = HyperGraph::loadhMETIS(load_path, true);
            } else if (file_path.extension() == ".snn") {
                std::cout << "Loading hypergraph from: " << load_path << " (SNN format) ...\n";
                std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(load_path)) / (1 << 20) << " MB\n";
                hg = HyperGraph::loadSNN(load_path, true);
            } else if (file_path.extension() == ".axh") {
                std::cout << "Loading hypergraph from: " << load_path << " (AXH format) ...\n";
                std::cout << "Hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(load_path)) / (1 << 20) << " MB\n";
                hg = HyperGraph::loadAXH(load_path, true);
            } else {
                throw std::runtime_error("Failed to load hypergraph, unsupported file format (supported: '.hgr', '.snn', '.axh').");
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading file: " << e.what() << "\n";
            std::exit(1);
        }
    } else {
        std::cerr << "WARNING, no hypergraph provided (-r), aborting !!\n";
        abort();
    }

    return hg;
}

Constraints setupPartConstr(PartConstrType constraints_type, ConstraintsConfig constr_config, const HyperGraph& hg, uint32_t kway, float epsi) {
    if (constraints_type == PartConstrType::KWAY) { // k-way mode ('-k')
        std::ostringstream epsistr;
        epsistr << std::fixed << std::setprecision(3) << epsi;
        constr_config.name = std::to_string(kway) + "-way " + epsistr.str() + " balanced";
        constr_config.nodes_per_part = (uint32_t)std::ceil((1 + epsi) * (float)hg.nodes() / kway);
        constr_config.inbound_per_part = INT32_MAX;
        constr_config.max_parts = kway;
        return Constraints(constr_config);
    } else if (constraints_type == PartConstrType::MANL) { // manual constraints ('-m')
        if (constr_config.nodes_per_part == 0) { std::cerr << "Error: the 1st constraint (max partition size) must be a positive integer \n"; std::exit(1); }
        if (constr_config.inbound_per_part == 0) { std::cerr << "Error: the 2nd constraint (max distinct inbound hedge per partition) must be a positive integer \n"; std::exit(1); }
        if (constr_config.max_parts == 0) { std::cerr << "Error: the 3rd constraint (max number of partitions) must be a positive integer \n"; std::exit(1); }
        return Constraints(constr_config);
    } else if (constraints_type == PartConstrType::NAME) { // preconfigured constraints ('-c')
        std::unordered_map<std::string, Constraints (*)()> configurations {
            { "loihi64", Constraints::createLoihiLarge },
            { "loihi84", Constraints::createLoihiJin84 },
            { "loihi1024", Constraints::createLoihiJin1024 },
            { "truenorth", Constraints::createTrueNorth }
        };
        auto constr_it = configurations.find(constr_config.name);
        if (constr_it == configurations.end()) {
            std::cerr << "WARNING, constraints name (-c " << constr_config.name << ") not recognized, aborting !!\n";
            abort();
        }
        return constr_it->second();
    } else { // no (valid) constraints provided
        std::cerr << "WARNING, no constraints provided (-c, -m, -k), aborting !!\n";
        abort();
    }
}

HardwareModel setupPlacConstr(std::string hw_name) {
    std::unordered_map<std::string, HardwareModel (*)()> configurations {
        { "loihi", HardwareModel::createLoihi },
        { "loihi64", HardwareModel::createLoihiLarge },
        { "loihi84", HardwareModel::createLoihiJin84 },
        { "loihi1024", HardwareModel::createLoihiJin1024 },
        { "truenorth", HardwareModel::createTrueNorth }
    };
    auto hw_it = configurations.find(hw_name);
    if (hw_it == configurations.end()) {
        std::cerr << "WARNING, no valid constraints provided (-c), aborting !!\n";
        abort();
    }
    return hw_it->second();
}

void savePartHgraph(std::string save_path, HyperGraph partitioned_hg) {
    // save hypergraph
    try {
        std::filesystem::path file_path(save_path);
        if (file_path.extension() == ".hgr") {
            std::cout << "Saving partitioned hypergraph to: " << save_path << " (hMETIS format) ...\n";
            partitioned_hg.savehMETIS(save_path);
        } else if (file_path.extension() == ".snn") {
            std::cout << "Saving partitioned hypergraph to: " << save_path << " (SNN format) ...\n";
            partitioned_hg.saveSNN(save_path);
        } else if (file_path.extension() == ".axh") {
            std::cout << "Saving partitioned hypergraph to: " << save_path << " (AXH format) ...\n";
            partitioned_hg.saveAXH(save_path);
        } else {
            throw std::runtime_error("Failed to save partitioned hypergraph, unsupported file format (supported: '.hgr', '.snn', '.axh').");
        }
        std::cout << "Partitioned hypergraph saved to " << save_path << "\n";
        std::cout << "Partitioned hypergraph file size: " << std::fixed << std::setprecision(1) << (float)(std::filesystem::file_size(save_path)) / (1 << 20) << " MB\n";
    } catch (const std::exception& e) {
        std::cerr << "Error saving file: " << e.what() << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc == 1) {
        printHelp();
        return 0;
    }

    // parse CLI args
    std::string load_path;
    std::string part_path;
    std::string plac_path;
    std::string save_path;
    // |
    PartConstrType part_constraints_type = PartConstrType::NONE;
    ConstraintsConfig part_constr_config; 
    uint32_t kway = 0;
    float epsi = 0.0f;
    // |
    std::string plac_constraints_name;

    // CLI handling
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h") { printHelp(); std::exit(0); }
        else if (arg == "-r") {
            if (i + 1 >= argc) { std::cerr << "Error: -r requires a file path\n"; std::exit(1); }
            load_path = argv[++i];
        } else if (arg == "-prt") {
            if (i + 1 >= argc) { std::cerr << "Error: -prt requires a file path\n"; std::exit(1); }
            part_path = argv[++i];
        } else if (arg == "-plc") {
            if (i + 1 >= argc) { std::cerr << "Error: -plc requires a file path\n"; std::exit(1); }
            plac_path = argv[++i];
        } else if (arg == "-s") {
            if (i + 1 >= argc) { std::cerr << "Error: -s requires a file path\n"; std::exit(1); }
            save_path = argv[++i];
        } else if (arg == "-c-prt") {
            if (i + 1 >= argc) { std::cerr << "Error: -c-prt requires a config name\n"; std::exit(1); }
            part_constraints_type = PartConstrType::NAME;
            part_constr_config.name = argv[++i];
        } else if (arg == "-m-prt") {
            if (i + 3 >= argc) { std::cerr << "Error: -m-prt requires integer values for the three constraints\n"; std::exit(1); }
            part_constraints_type = PartConstrType::MANL;
            part_constr_config.name = "manual";
            part_constr_config.nodes_per_part = std::stoul(argv[++i]);
            part_constr_config.inbound_per_part = std::stoul(argv[++i]);
            part_constr_config.max_parts = std::stoul(argv[++i]);
        } else if (arg == "-k-prt") {
            if (i + 2 >= argc) { std::cerr << "Error: -k-prt requires values for 'k' and 'ε'\n"; std::exit(1); }
            part_constraints_type = PartConstrType::KWAY;
            kway = std::stoul(argv[++i]);
            epsi = std::stof(argv[++i]);
        } else if (arg == "-c-plc") {
            if (i + 1 >= argc) { std::cerr << "Error: -c-plc requires a config name\n"; std::exit(1); }
            plac_constraints_name = argv[++i];
        } else { std::cerr << "Unknown option: " << arg << "\n"; std::exit(1); }
    }

    // task selection
    bool eval_part = part_constraints_type != PartConstrType::NONE; // true => evaluate partitioning
    bool eval_plac = !plac_constraints_name.empty(); // true => evaluate placement

    std::cout << "Evaluation task:\n";
    if (eval_part && eval_plac) std::cout << "  -> partitioning and placement\n";
    else if (eval_part) std::cout << "  -> partitioning only\n";
    else if (eval_plac) std::cout << "  -> placement only (assuming input hgraph to be an already partitioned one)\n";
    else {
        std::cout << "  -> no task provided (no option provided among -c-prt, -m-prt, -k-prt, -c-plc)\n";
        return 0;
    }

    if (load_path.empty()) {
        std::cerr << "Error: -r is required to load the hypergraph to evaluate.\n";
        return 1;
    }
    if (eval_part && part_path.empty()) {
        std::cerr << "Error: partitioning evaluation requires -prt <partitioning_file>.\n";
        return 1;
    } else if (!eval_part && !part_path.empty()) {
        std::cerr << "WARNING, a partitioning was provided (-prt), but no partitioning evaluation was requested (no partitioning constraints present).\n";
    }
    if (eval_plac && plac_path.empty()) {
        std::cerr << "Error: placement evaluation requires -plc <placement_file>.\n";
        return 1;
    } else if (!eval_plac && !plac_path.empty()) {
        std::cerr << "WARNING, a placement was provided (-plc), but no placement evaluation was requested (no placement constraints present).\n";
    }

    // load hypergraph
    HyperGraph hg = loadHgraph(load_path);

    // setup partitioning constraints
    std::optional<Constraints> part_constr;
    if (eval_part) part_constr.emplace(setupPartConstr(part_constraints_type, part_constr_config, hg, kway, epsi));

    // setup placement constraints
    std::optional<HardwareModel> plac_constr;
    if (eval_plac) plac_constr.emplace(setupPlacConstr(plac_constraints_name));

    std::optional<HyperGraph> partitioned_hg;

    // print statistics
    std::cout << "Loaded hypergraph:\n";
    std::cout << "  Nodes:      " << hg.nodes() << "\n";
    std::cout << "  Hyperedges: " << hg.hedges().size() << "\n";
    std::cout << "  Total pins: " << hg.hedgesFlat().size() << "\n";
    std::cout << "  Total connections weight: " << std::fixed << std::setprecision(3) << hg.connectivity() << "\n";

    if (eval_part) {
        std::cout << "Using partitioning constraints \"" << part_constr->name() << "\":\n";
        std::cout << "  Nodes per partition:         " << part_constr->nodesPerPart() << "\n";
        std::cout << "  Inbound hedge per partition: " << part_constr->inboundPerPart() << "\n";
        std::cout << "  Maximum partitions:          " << part_constr->maxParts() << "\n";
    }

    if (eval_plac) {
        std::cout << "Using placement constraints \"" << plac_constr->name() << "\":\n";
        std::cout << "  Neurons per core:  " << plac_constr->neuronsPerCore() << "\n";
        std::cout << "  Synapses per core: " << plac_constr->synapsesPerCore() << "\n";
        std::cout << "  Cores along x, y:  " << plac_constr->coresPerChipX() << ", " << plac_constr->coresPerChipY() << " (" << plac_constr->coresPerChipX() * plac_constr->coresPerChipY() << " tot.)" << "\n";
        std::cout << "  Chips along x, y:  " << plac_constr->chipsPerSystemX() << ", " << plac_constr->chipsPerSystemY() << " (" << plac_constr->chipsPerSystemX() * plac_constr->chipsPerSystemY() << " tot.)" << "\n";
        std::cout << "  Routing energy, latency: " << std::fixed << std::setprecision(3) << plac_constr->energyPerRouting() << " pJ, " << plac_constr->latencyPerRouting() << " ns\n";
        std::cout << "  Wire energy, latency:    " << std::fixed << std::setprecision(3) << plac_constr->energyPerWire() << " pJ, " << plac_constr->latencyPerWire() << " ns\n";
    }


    if (eval_part) {
        // load partitioning
        std::vector<uint32_t> partitions;
        try {
            if (!std::filesystem::is_regular_file(part_path)) throw std::runtime_error("Failed to load partitioning, the provided path is not a file.");
            partitions = hg.loadPartitioning(part_path);
        } catch (const std::exception& e) {
            std::cerr << "Error loading partitioning: " << e.what() << "\n";
            return 1;
        }

        // apply and grade partitioning
        if (part_constr->checkPartitionValidity(hg, partitions, true)) {
            // log metrics
            partitioned_hg.emplace(hg.getPartitionsHypergraph(partitions, 2, true)); // remove the destination if self-cycles happen
            auto hedge_overlap = part_constr->hedgeOverlap(hg, partitions);
            std::cout << "Partitioned hypergraph metrics:\n";
            std::cout << "  Nodes:         " << partitioned_hg->nodes() << "\n";
            std::cout << "  Hyperedges:    " << partitioned_hg->hedges().size() << "\n";
            std::cout << "  Total pins:    " << partitioned_hg->hedgesFlat().size() << "\n";
            std::cout << "  Cut-net:       " << partitioned_hg->cutnet() << "\n";
            std::cout << "  Connectivity:  " << partitioned_hg->connectivity() << "\n";
            std::cout << "  SOED:          " << hg.soedFromPart(partitions) << "\n";
            std::cout << "  Hedge overlap: " << std::fixed << std::setprecision(3) << hedge_overlap.ar_mean << " ar. mean, " << hedge_overlap.geo_mean << " geo. mean\n";
            
            // save partitioned hypergraph
            if (!save_path.empty()) savePartHgraph(save_path, *partitioned_hg);
        } else {
            std::cerr << "ERROR, invalid partitioning !!\n";
            return 1;
        }
    }

    if (eval_plac) {
        // load placement
        std::vector<Coord2D> placement;
        try {
            if (!std::filesystem::is_regular_file(plac_path)) throw std::runtime_error("Failed to load placement, the provided path is not a file.");
            placement = coords_from_file(plac_path);
        } catch (const std::exception& e) {
            std::cerr << "Error loading placement: " << e.what() << "\n";
            return 1;
        }

        // apply and grade placement
        const HyperGraph& placement_hg = partitioned_hg.has_value() ? *partitioned_hg : hg;
        if (plac_constr->checkPlacementValidity(placement_hg, placement, true)) {
            auto metrics = plac_constr->getAllMetrics(placement_hg, placement);
            std::cout << "Placement metrics:\n";
            std::cout << "  Energy:        " << std::fixed << std::setprecision(3) << metrics.energy.value() << "\n";
            std::cout << "  Avg. latency:  " << std::fixed << std::setprecision(3) << metrics.avg_latency.value() << "\n";
            std::cout << "  Max. latency:  " << std::fixed << std::setprecision(3) << metrics.max_latency.value() << "\n";
            std::cout << "  Avg. congestion:  " << std::fixed << std::setprecision(3) << metrics.avg_congestion.value() << "\n";
            std::cout << "  Max. congestion:  " << std::fixed << std::setprecision(3) << metrics.max_congestion.value() << "\n";
            std::cout << "  Connections locality:\n";
            std::cout << "    Flat:     " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean << " ar. mean, " << metrics.connections_locality.value().geo_mean << " geo. mean\n";
            std::cout << "    Weighted: " << std::fixed << std::setprecision(3) << metrics.connections_locality.value().ar_mean_weighted << " ar. mean, " << metrics.connections_locality.value().geo_mean_weighted << " geo. mean\n";
        } else {
            std::cerr << "WARNING, invalid placement !!\n";
        }
    }

    return 0;
}
