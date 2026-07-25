#pragma once
#include <tuple>
#include <vector>
#include <string>
#include <cstdint>
#include <utility>
#include <optional>
#include <concepts>
#include <stdexcept>
#include <unordered_map>

#include "hgraph.hpp"
#include "topology.hpp"

// PLACEMENT COST MODEL (constraints & costs)

namespace hwmodel {

    using namespace hgraph;
    using namespace topology;

    struct SynapticReuseMetrics {
        double ar_mean{0.0};
        double geo_mean{0.0};
    };

    struct ConnectionsLocalityMetrics {
        double ar_mean{0.0};
        double geo_mean{0.0};
        double ar_mean_weighted{0.0};
        double geo_mean_weighted{0.0};
    };

    struct PlacementMetrics {
        bool valid{false};
        std::optional<double> energy;
        std::optional<double> avg_latency;
        std::optional<double> max_latency;
        std::optional<double> avg_congestion;
        std::optional<double> max_congestion;
        std::optional<ConnectionsLocalityMetrics> connections_locality;
    };

    struct MulticastPlacementMetrics {
        bool valid{false};
        std::optional<double> energy;
        std::optional<double> avg_latency;
        std::optional<double> max_congestion;
        // fraction of sourced-hyperedge weight for which every source's Steiner tree was solved
        float evaluation_fraction{0.0f};
    };
    
    struct HardwareModelConfig {
        std::string name;
        uint32_t nodes_per_core;
        uint32_t synapses_per_core;
        double energy_per_routing;
        double energy_per_wire;
        double latency_per_routing;
        double latency_per_wire;
    };

    // partitionSequential: sequential greedy partitioning with constraints on
    // max neurons (N), max inbound synapses (M), and max partitions (K).
    std::vector<uint32_t> partitionSequential(const HyperGraph& hg, uint32_t N, uint32_t M, uint32_t K);

    template<Topology T>
    class HardwareModel {
        public:
        using Coord = Coord_t<T>;

        private:
        std::string name_;
        // CONSTRAINTS
        T topology_; // the topology's extent determines the number of cores along each dimension
        uint32_t nodes_per_core_;
        uint32_t synapses_per_core_;

        // COSTS
        double energy_per_routing_; // in [pJ]
        double energy_per_wire_; // in [pJ]
        double latency_per_routing_; // in [ns]
        double latency_per_wire_; // in [ns]

        public:
        HardwareModel(const HardwareModelConfig& cfg, T topology) :
            name_(cfg.name),
            topology_(topology),
            nodes_per_core_(cfg.nodes_per_core),
            synapses_per_core_(cfg.synapses_per_core),
            energy_per_routing_(cfg.energy_per_routing),
            energy_per_wire_(cfg.energy_per_wire),
            latency_per_routing_(cfg.latency_per_routing),
            latency_per_wire_(cfg.latency_per_wire)
        {
            if (!(nodes_per_core_ > 0 && synapses_per_core_ > 0)) {
                throw std::invalid_argument("All hardware constraints must be > 0.");
            }
            if (!(energy_per_routing_ >= 0.0 && energy_per_wire_ >= 0.0 &&
                latency_per_routing_ >= 0.0 && latency_per_wire_ >= 0.0)) {
                throw std::invalid_argument("All hardware costs must be >= 0.");
            }
        }

        // accessors
        std::string name() const { return name_; }
        const T& topology() const noexcept { return topology_; }
        uint32_t neuronsPerCore() const { return nodes_per_core_; }
        uint32_t synapsesPerCore() const { return synapses_per_core_; }

        double energyPerRouting() const { return energy_per_routing_; }
        double energyPerWire() const { return energy_per_wire_; }
        double latencyPerRouting() const { return latency_per_routing_; }
        double latencyPerWire() const { return latency_per_wire_; }

        uint32_t coresCount() const { return topology_.extent().volume(); }
        uint32_t coresAlongDim(uint32_t dim) const { return topology_.extent()[dim]; }

        // empirical: can have false negatives (no false positives tho)
        bool checkHgraphFit(const HyperGraph& hgraph,
                        bool already_partitioned = false,
                        bool verbose = false) const;

        bool checkPartitionValidity(const HyperGraph& hgraph,
                                    const std::vector<uint32_t>& partitions,
                                    bool verbose = false) const;

        bool checkPlacementValidity(const HyperGraph& part_hgraph,
                                    const std::vector<Coord>& placement,
                                    bool verbose = false) const;

        // average inbound set overlap of nodes in the same partition
        SynapticReuseMetrics synapticReuse(const HyperGraph& hgraph,
                                        const std::vector<uint32_t>& partitions) const;

        // average hyperedge volume in the placement
        // => volume: # of coordinates trapped in the smallest rectangle that encloses all of an hyperedge's pins
        ConnectionsLocalityMetrics connectionsLocality(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        // probability that a uniformly selected minimum path traverses each coordinate
        std::unordered_map<Coord, double> expectedSpikeTransitProbability(
            const Coord& src, const Coord& dst) const;

        double placementEnergyConsumption(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        double placementAverageLatency(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        double placementMaximumLatency(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        // congestion is the weighted fraction of minimum src-dst paths over every hyperedge that cross a core
        double placementAverageCongestion(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        double placementMaximumCongestion(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        PlacementMetrics getAllUnicastMetrics(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        double multicastEnergyConsumption(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        double multicastAverageLatency(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        std::vector<double> multicastCongestion(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        double multicastMaximumCongestion(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        MulticastPlacementMetrics getAllMulticastMetrics(
            const HyperGraph& part_hgraph,
            const std::vector<Coord>& placement) const;

        // predefined harwdare models
        static HardwareModel<T> createLoihi() {
            std::array<int, T::dimensions> extent;
            extent.fill(8);
            extent[0] = 16; // first dimension is 16, all others are 8
            T topo = std::make_from_tuple<T>(extent);
            HardwareModelConfig cfg_loihi;
            cfg_loihi.name = "Loihi";
            cfg_loihi.nodes_per_core = 1024;
            cfg_loihi.synapses_per_core = 4096;
            cfg_loihi.energy_per_routing = 1.7;
            cfg_loihi.energy_per_wire = 3.5;
            cfg_loihi.latency_per_routing = 2.1;
            cfg_loihi.latency_per_wire = 5.3;
            return HardwareModel<T>(cfg_loihi, topo);
        };

        static HardwareModel<T> createLoihiLarge() {
            std::array<int, T::dimensions> extent;
            extent.fill(64);
            T topo = std::make_from_tuple<T>(extent);
            HardwareModelConfig cfg_loihi_large;
            cfg_loihi_large.name = "Loihi Large";
            cfg_loihi_large.nodes_per_core = 1024;
            cfg_loihi_large.synapses_per_core = 4096;
            cfg_loihi_large.energy_per_routing = 1.7;
            cfg_loihi_large.energy_per_wire = 3.5;
            cfg_loihi_large.latency_per_routing = 2.1;
            cfg_loihi_large.latency_per_wire = 5.3;
            return HardwareModel<T>(cfg_loihi_large, topo);
        };

        static HardwareModel<T> createLoihiJin84() {
            std::array<int, T::dimensions> extent;
            extent.fill(84);
            T topo = std::make_from_tuple<T>(extent);
            HardwareModelConfig cfg_loihi_jin_84;
            cfg_loihi_jin_84.name = "Loihi Jin 84";
            cfg_loihi_jin_84.nodes_per_core = 4096;
            cfg_loihi_jin_84.synapses_per_core = 1024*64;
            cfg_loihi_jin_84.energy_per_routing = 1.0;
            cfg_loihi_jin_84.energy_per_wire = 0.1;
            cfg_loihi_jin_84.latency_per_routing = 1.0;
            cfg_loihi_jin_84.latency_per_wire = 0.01;
            return HardwareModel<T>(cfg_loihi_jin_84, topo);
        };

        static HardwareModel<T> createLoihiJin1024() {
            std::array<int, T::dimensions> extent;
            extent.fill(1024);
            T topo = std::make_from_tuple<T>(extent);
            HardwareModelConfig cfg_loihi_jin_1024;
            cfg_loihi_jin_1024.name = "Loihi Jin 1024";
            cfg_loihi_jin_1024.nodes_per_core = 4096;
            cfg_loihi_jin_1024.synapses_per_core = 1024*64;
            cfg_loihi_jin_1024.energy_per_routing = 1.0;
            cfg_loihi_jin_1024.energy_per_wire = 0.1;
            cfg_loihi_jin_1024.latency_per_routing = 1.0;
            cfg_loihi_jin_1024.latency_per_wire = 0.01;
            return HardwareModel<T>(cfg_loihi_jin_1024, topo);
        };

        static HardwareModel<T> createTrueNorth() {
            std::array<int, T::dimensions> extent;
            extent.fill(64);
            T topo = std::make_from_tuple<T>(extent);
            HardwareModelConfig cfg_truenorth;
            cfg_truenorth.name = "TrueNorth";
            cfg_truenorth.nodes_per_core = 256;
            cfg_truenorth.synapses_per_core = 256;
            cfg_truenorth.energy_per_routing = 1.7; // unknown (this is from Loihi)
            cfg_truenorth.energy_per_wire = 3.5; // unknown (this is from Loihi)
            cfg_truenorth.latency_per_routing = 2.1; // unknown (this is from Loihi)
            cfg_truenorth.latency_per_wire = 5.3; // unknown (this is from Loihi)
            return HardwareModel<T>(cfg_truenorth, topo);
        };
    };

    // ready to use topologies
    extern template class HardwareModel<Lattice2D>;
    extern template class HardwareModel<Lattice3D>;
    extern template class HardwareModel<Torus2D>;
    extern template class HardwareModel<Torus3D>;
    extern template class HardwareModel<Torus6D>;

}
