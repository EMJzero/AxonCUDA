#pragma once
#include <string>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace hgraph {
    class HyperGraph;
}

// CONSTRAINTS

namespace constraints {

    struct HedgeOverlapMetrics {
        double ar_mean{0.0};
        double geo_mean{0.0};
    };
    
    struct ConstraintsConfig {
        std::string name;
        uint32_t nodes_per_part;
        uint32_t inbound_per_part; // distinct inbound axons per partition
        uint32_t pins_per_part; // synapses (inbound hgraph pins) per partition -> sum of the nodes' inbound set cardinalities
        uint32_t max_parts;
    };

    // sequential greedy partitioning with constraints:
    // max nodes per part (N), max inbound per part (M), max inbound pins per part (Q), and max partitions (K).
    std::vector<uint32_t> partitionSequential(const hgraph::HyperGraph& hg, uint32_t N, uint32_t M, uint32_t Q, uint32_t K);

    class Constraints {
        private:
        std::string name_;
        // CONSTRAINTS
        uint32_t nodes_per_part_;
        uint32_t inbound_per_part_; // distinct inbound axons per partition
        uint32_t pins_per_part_; // synapses (inbound hgraph pins) per partition
        uint32_t max_parts_;

        public:
        Constraints(const ConstraintsConfig& cfg) :
            name_(cfg.name),
            nodes_per_part_(cfg.nodes_per_part),
            inbound_per_part_(cfg.inbound_per_part),
            pins_per_part_(cfg.pins_per_part),
            max_parts_(cfg.max_parts)
        {
            if (!(nodes_per_part_ > 0 && inbound_per_part_ > 0 && pins_per_part_ > 0 && max_parts_ > 0))
                throw std::invalid_argument("All constraints must be > 0.");
        }

        // accessors
        std::string name() const { return name_; }
        uint32_t nodesPerPart() const { return nodes_per_part_; }
        uint32_t inboundPerPart() const { return inbound_per_part_; }
        uint32_t pinsPerPart() const { return pins_per_part_; }
        uint32_t maxParts() const { return max_parts_; }

        // empirical: can have false negatives (no false positives tho)
        bool checkFit(const hgraph::HyperGraph& snn, bool already_partitioned = false, bool verbose = false) const;

        bool checkPartitionValidity(const hgraph::HyperGraph& snn, const std::vector<uint32_t>& partitions, bool verbose = false) const;

        HedgeOverlapMetrics hedgeOverlap(const hgraph::HyperGraph& hg, const std::vector<uint32_t>& partitions) const;

        // predefined constraint sets
        static Constraints createLoihiLarge();
        static Constraints createLoihiJin84();
        static Constraints createLoihiJin1024();
        static Constraints createTrueNorth();
    };
}
