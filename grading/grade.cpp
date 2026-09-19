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
using namespace topology;


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
        "  -m-prt <>x4     Partitioning constraints set to use, in order: max part. size, max part. distinct inbound hedges, max part. pins, max num. of part.s (overrides '-c-prt')\n"
        "  -k-prt <k> <ε>  K-way balanced constraints set to use (overrides '-c-prt' and '-m-prt')\n"
        "  -c-plc <name>   Placement constraints set to use (valid ones: truenorth, loihi, loihi64, loihi84, loihi1024)\n"
        "  -t-plc <name>   Placement topology (valid ones: lat2d, tor6d)\n"
        "  -ff         Reorder the partitioned hypergraph's nodes with the greedy feedforward algorithm (use if '-ff' was used for placement)\n"
        "  -force-prt  Grade the placement even when the partitioning breaks a constraint (still reports it)\n"
        "  -ff-prt     Read the partitioning in feedforward-order node space, rather than the hypergraph's own\n"
        "              (AxonFlow reorders a hypergraph before partitioning it whenever it is not topologically\n"
        "               sorted, so its '.part' files for such graphs are indexed in that reordered space)\n"
        "  -rp <list>  Comma-separated routing policies to evaluate placement quality metrics under (default: unicast,xy):\n"
        "      - unicast: one independent minimum path per destination (pessimistic bound)\n"
        "      - xy: dimension-order multicast tree, X first and then Y (realistic reference)\n"
        "      - steiner: minimum Steiner tree multicast (optimistic bound, can take hours!)\n"
        "      - none: disables all placement quality metrics\n"
        "  -h          Show this help\n";
}

// a "-rp" list fully overrides the defaults, hence every flag is cleared before parsing
bool parseRoutingPolicies(const std::string& list, bool& unicast, bool& xy_multicast, bool& steiner_multicast) {
    unicast = false;
    xy_multicast = false;
    steiner_multicast = false;

    std::stringstream stream(list);
    std::string policy;
    bool any = false;
    while (std::getline(stream, policy, ',')) {
        if (policy.empty()) continue;
        any = true;
        if (policy == "unicast") unicast = true;
        else if (policy == "xy") xy_multicast = true;
        else if (policy == "steiner") steiner_multicast = true;
        else if (policy == "none") { /* leave every flag cleared */ }
        else return false;
    }
    return any;
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
        constr_config.pins_per_part = INT32_MAX;
        constr_config.max_parts = kway;
        return Constraints(constr_config);
    } else if (constraints_type == PartConstrType::MANL) { // manual constraints ('-m')
        if (constr_config.nodes_per_part == 0) { std::cerr << "Error: the 1st constraint (max partition size) must be a positive integer \n"; std::exit(1); }
        if (constr_config.inbound_per_part == 0) { std::cerr << "Error: the 2nd constraint (max distinct inbound hedge per partition) must be a positive integer \n"; std::exit(1); }
        if (constr_config.pins_per_part == 0) { std::cerr << "Error: the 3rd constraint (max pins per partition) must be a positive integer \n"; std::exit(1); }
        if (constr_config.max_parts == 0) { std::cerr << "Error: the 4th constraint (max number of partitions) must be a positive integer \n"; std::exit(1); }
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

template<Topology T>
HardwareModel<T> setupPlacConstr(std::string hw_name) {
    using Model = HardwareModel<T>;
    std::unordered_map<std::string, Model (*)()> configurations {
        { "loihi", Model::createLoihi },
        { "loihi64", Model::createLoihiLarge },
        { "loihi84", Model::createLoihiJin84 },
        { "loihi1024", Model::createLoihiJin1024 },
        { "truenorth", Model::createTrueNorth }
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
    std::string topology_name = "lat2d";
    bool feedforward_order = false;
    bool feedforward_partitioning = false;
    bool force_partitioning = false;
    bool unicast_metrics = true;
    bool xy_multicast_metrics = true;
    bool steiner_multicast_metrics = false; // opt-in: solving minimum Steiner trees can take hours

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
            if (i + 4 >= argc) { std::cerr << "Error: -m-prt requires integer values for the four constraints\n"; std::exit(1); }
            part_constraints_type = PartConstrType::MANL;
            part_constr_config.name = "manual";
            part_constr_config.nodes_per_part = std::stoul(argv[++i]);
            part_constr_config.inbound_per_part = std::stoul(argv[++i]);
            part_constr_config.pins_per_part = std::stoul(argv[++i]);
            part_constr_config.max_parts = std::stoul(argv[++i]);
        } else if (arg == "-k-prt") {
            if (i + 2 >= argc) { std::cerr << "Error: -k-prt requires values for 'k' and 'ε'\n"; std::exit(1); }
            part_constraints_type = PartConstrType::KWAY;
            kway = std::stoul(argv[++i]);
            epsi = std::stof(argv[++i]);
        } else if (arg == "-c-plc") {
            if (i + 1 >= argc) { std::cerr << "Error: -c-plc requires a config name\n"; std::exit(1); }
            plac_constraints_name = argv[++i];
        }  else if (arg == "-ff") {
            feedforward_order = true;
        } else if (arg == "-ff-prt") {
            feedforward_partitioning = true;
        } else if (arg == "-force-prt") {
            force_partitioning = true;
        } else if (arg == "-rp") {
            if (i + 1 >= argc) { std::cerr << "Error: -rp requires a comma-separated list of routing policies\n"; std::exit(1); }
            std::string policies = argv[++i];
            if (!parseRoutingPolicies(policies, unicast_metrics, xy_multicast_metrics, steiner_multicast_metrics)) {
                std::cerr << "Error: -rp requested an invalid routing policy name (valid ones: unicast, xy, steiner, none)\n"; std::exit(1);
            }
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

    std::optional<HyperGraph> partitioned_hg;

    // print statistics
    std::cout << "Loaded hypergraph:\n";
    std::cout << "  Nodes:      " << hg.nodes() << "\n";
    std::cout << "  Hyperedges: " << hg.hedges().size() << "\n";
    std::cout << "  Total pins: " << hg.hedgesFlat().size() << "\n";
    std::cout << "  Total connections weight: " << std::fixed << std::setprecision(3) << hg.connectivity() << "\n";

    if (eval_part) {
        // setup partitioning constraints
        Constraints part_constr = setupPartConstr(part_constraints_type, part_constr_config, hg, kway, epsi);

        std::cout << "Using partitioning constraints \"" << part_constr.name() << "\":\n";
        std::cout << "  Nodes per partition:         " << part_constr.nodesPerPart() << "\n";
        std::cout << "  Inbound hedge per partition: " << part_constr.inboundPerPart() << "\n";
        std::cout << "  Inbound pins per partition:  " << part_constr.pinsPerPart() << "\n";
        std::cout << "  Maximum partitions:          " << part_constr.maxParts() << "\n";

        // load partitioning
        std::vector<uint32_t> partitions;
        try {
            if (!std::filesystem::is_regular_file(part_path)) throw std::runtime_error("Failed to load partitioning, the provided path is not a file.");
            partitions = hg.loadPartitioning(part_path);
        } catch (const std::exception& e) {
            std::cerr << "Error loading partitioning: " << e.what() << "\n";
            return 1;
        }

        // the partitioning may be indexed in feedforward-order node space rather than the hypergraph's own
        // => new_id[n] is where node n lands in that order, so its partition sits at partitions[new_id[n]]
        if (feedforward_partitioning) {
            hg.buildIncidenceSets();
            const std::vector<uint32_t> new_id = hg.feedForwardOrder();
            std::vector<uint32_t> reindexed(partitions.size());
            for (uint32_t node = 0; node < hg.nodes(); ++node)
                reindexed[node] = partitions[new_id[node]];
            partitions = std::move(reindexed);
        }

        // apply and grade partitioning
        // NOTE: -force-prt keeps going on a constraint breach, so that a placement can still be graded
        // when only the partitioning constraints disagree with whoever produced the mapping
        const bool part_valid = part_constr.checkPartitionValidity(hg, partitions, true);
        if (part_valid || force_partitioning) {
            if (!part_valid)
                std::cerr << "WARNING, grading a partitioning that breaks a constraint (-force-prt) !!\n";
            // log metrics
            partitioned_hg.emplace(hg.getPartitionsHypergraph(partitions, 2, true)); // remove the destination if self-cycles happen
            auto hedge_overlap = part_constr.hedgeOverlap(hg, partitions);
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
        // topology-templated evaluation routine
        auto place_eval = [&]<Topology T>() -> int {
            // setup placement constraints
            HardwareModel<T> plac_constr = setupPlacConstr<T>(plac_constraints_name);

            using Coord = Coord_t<T>;

            std::cout << "Using placement constraints \"" << plac_constr.name() << "\":\n";
            std::cout << "  Topology:                 " << topology_name << "\n";
            std::cout << "  Neurons per core:         " << plac_constr.neuronsPerCore() << "\n";
            std::cout << "  Inbound axons per core:   " << plac_constr.inboundPerCore() << "\n";
            std::cout << "  Synapses (pins) per core: " << plac_constr.pinsPerCore() << "\n";
            std::cout << "  Cores per dim:            " << plac_constr.coresAlongDim(0);
            for (uint32_t dim = 1; dim < T::dimensions; dim++)
                std::cout << ", " << plac_constr.coresAlongDim(dim);
            std::cout << " (" << plac_constr.coresCount() << " tot.)" << "\n";
            std::cout << "  Routing energy, latency: " << std::fixed << std::setprecision(3) << plac_constr.energyPerRouting() << " pJ, " << plac_constr.latencyPerRouting() << " ns\n";
            std::cout << "  Wire energy, latency:    " << std::fixed << std::setprecision(3) << plac_constr.energyPerWire() << " pJ, " << plac_constr.latencyPerWire() << " ns\n";

            // load placement
            std::vector<Coord> placement;
            try {
                if (!std::filesystem::is_regular_file(plac_path)) throw std::runtime_error("Failed to load placement, the provided path is not a file.");
                placement = Coord::fromFile(plac_path);
            } catch (const std::exception& e) {
                std::cerr << "Error loading placement: " << e.what() << "\n";
                return 1;
            }

            // if partitioning was evaluated, use the partitioned hgraph
            HyperGraph& placement_hg = partitioned_hg.has_value() ? *partitioned_hg : hg;

            if (feedforward_order) {
                placement_hg.buildIncidenceSets();
                std::vector<uint32_t> nodes_order_idx = placement_hg.feedForwardOrder();
                std::vector<Coord> placement_ord(placement_hg.nodes());
                for (uint32_t i = 0; i < placement_ord.size(); i++) {
                    placement_ord[i] = placement[nodes_order_idx[i]];
                }
                placement = placement_ord;
            }
            
            // apply and grade placement
            if (plac_constr.checkPlacementValidity(placement_hg, placement, true)) {
                if (unicast_metrics) {
                    auto uc_metrics = plac_constr.getAllUnicastMetrics(placement_hg, placement);
                    std::cout << "Placement unicast metrics:\n";
                    std::cout << "  Energy:        " << std::fixed << std::setprecision(3) << uc_metrics.energy.value() << "\n";
                    std::cout << "  Avg. latency:  " << std::fixed << std::setprecision(3) << uc_metrics.avg_latency.value() << "\n";
                    std::cout << "  Max. latency:  " << std::fixed << std::setprecision(3) << uc_metrics.max_latency.value() << "\n";
                    std::cout << "  Avg. congestion:  " << std::fixed << std::setprecision(3) << uc_metrics.avg_congestion.value() << "\n";
                    std::cout << "  Max. congestion:  " << std::fixed << std::setprecision(3) << uc_metrics.max_congestion.value() << "\n";
                    std::cout << "  Connections locality:\n";
                    std::cout << "    Flat:     " << std::fixed << std::setprecision(3) << uc_metrics.connections_locality.value().ar_mean << " ar. mean, " << uc_metrics.connections_locality.value().geo_mean << " geo. mean\n";
                    std::cout << "    Weighted: " << std::fixed << std::setprecision(3) << uc_metrics.connections_locality.value().ar_mean_weighted << " ar. mean, " << uc_metrics.connections_locality.value().geo_mean_weighted << " geo. mean\n";
                }

                if (xy_multicast_metrics) {
                    auto xy_metrics = plac_constr.getAllXYMulticastMetrics(placement_hg, placement);
                    std::cout << "Placement XY-multicast metrics:\n";
                    if (xy_metrics.energy.has_value()) std::cout << "  Energy:          " << std::fixed << std::setprecision(3) << xy_metrics.energy.value() << "\n";
                    else std::cout << "  Energy:          N/A (not implemented for this topology)\n";
                    std::cout << "  Avg. latency:    " << std::fixed << std::setprecision(3) << xy_metrics.avg_latency.value() << "\n";
                    if (xy_metrics.avg_congestion.has_value()) std::cout << "  Avg. congestion: " << std::fixed << std::setprecision(3) << xy_metrics.avg_congestion.value() << "\n";
                    else std::cout << "  Avg. congestion: N/A (not implemented for this topology)\n";
                    if (xy_metrics.max_congestion.has_value()) std::cout << "  Max. congestion: " << std::fixed << std::setprecision(3) << xy_metrics.max_congestion.value() << "\n";
                    else std::cout << "  Max. congestion: N/A (not implemented for this topology)\n";
                }

                if (steiner_multicast_metrics) {
                    auto mc_metrics = plac_constr.getAllSteinerMulticastMetrics(placement_hg, placement);
                    std::cout << "Placement Steiner-multicast metrics:\n";
                    if (mc_metrics.energy.has_value()) std::cout << "  Energy:          " << std::fixed << std::setprecision(3) << mc_metrics.energy.value() << "\n";
                    else std::cout << "  Energy:          N/A (not implemented for this topology)\n";
                    std::cout << "  Avg. latency:    " << std::fixed << std::setprecision(3) << mc_metrics.avg_latency.value() << "\n";
                    if (mc_metrics.avg_congestion.has_value()) std::cout << "  Avg. congestion: " << std::fixed << std::setprecision(3) << mc_metrics.avg_congestion.value() << "\n";
                    else std::cout << "  Avg. congestion: N/A (not implemented for this topology)\n";
                    if (mc_metrics.max_congestion.has_value()) std::cout << "  Max. congestion: " << std::fixed << std::setprecision(3) << mc_metrics.max_congestion.value() << "\n";
                    else std::cout << "  Max. congestion: N/A (not implemented for this topology)\n";
                    std::cout << "  Evaluation fraction: " << std::fixed << std::setprecision(3) << mc_metrics.evaluation_fraction << "\n";
                }
            } else {
                std::cerr << "WARNING, invalid placement !!\n";
            }

            return 0;
        };  // place_eval end

        // dispatch based on topology
        if (topology_name == "lat2d")
            return place_eval.template operator()<Lattice<2>>();
        else if (topology_name == "tor6d")
            return place_eval.template operator()<Torus<6>>();
        else
            throw std::runtime_error("Topology not yet implemented *-* !");
    }

    return 0;
}
