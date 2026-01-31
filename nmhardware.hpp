#pragma once
#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <iostream>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

#include "hgraph.hpp"

// CONSTRAINTS

namespace constraints {

    using namespace hgraph;

    struct HedgeOverlapMetrics {
        double ar_mean{0.0};
        double geo_mean{0.0};
    };
    
    struct ConstraintsConfig {
        std::string name;
        uint32_t nodes_per_part;
        uint32_t inbound_per_part;
        uint32_t max_parts;
    };

    // sequential greedy partitioning with constraints:
    // max nodes per part (N), max inbound per part (M), and max partitions (K).
    std::vector<uint32_t> partitionSequential(const HyperGraph& hg, uint32_t N, uint32_t M, uint32_t K);

    class Constraints {
        private:
        std::string name_;
        // CONSTRAINTS
        uint32_t nodes_per_part_;
        uint32_t inbound_per_part_;
        uint32_t max_parts_;

        public:
        Constraints(const ConstraintsConfig& cfg) :
            name_(cfg.name),
            nodes_per_part_(cfg.nodes_per_part),
            inbound_per_part_(cfg.inbound_per_part),
            max_parts_(cfg.max_parts)
        {
            if (!(nodes_per_part_ > 0 && inbound_per_part_ > 0 && max_parts_ > 0))
                throw std::invalid_argument("All constraints must be > 0.");
        }

        // accessors
        std::string name() const { return name_; }
        uint32_t nodesPerPart() const { return nodes_per_part_; }
        uint32_t inboundPerPart() const { return inbound_per_part_; }
        uint32_t maxParts() const { return max_parts_; }

        // empirical: can have false negatives (no false positives tho)
        bool checkFit(const HyperGraph& snn, bool already_partitioned = false, bool verbose = false) const;

        bool checkPartitionValidity(const HyperGraph& snn, const std::vector<uint32_t>& partitions, bool verbose = false) const;

        HedgeOverlapMetrics hedgeOverlap(const HyperGraph& hg, const std::vector<uint32_t>& partitions) const;

        // predefined constraint sets
        static Constraints createLoihiLarge() {
            ConstraintsConfig cfg_loihi_large;
            cfg_loihi_large.name = "Loihi Large";
            cfg_loihi_large.nodes_per_part = 1024;
            cfg_loihi_large.inbound_per_part = 4096;
            cfg_loihi_large.max_parts = 4096;
            return Constraints(cfg_loihi_large);
        };

        static Constraints createLoihiJin84() {
            ConstraintsConfig cfg_loihi_jin_84;
            cfg_loihi_jin_84.name = "Loihi Jin 84";
            cfg_loihi_jin_84.nodes_per_part = 4096;
            cfg_loihi_jin_84.inbound_per_part = 1024*64;
            cfg_loihi_jin_84.max_parts = 7056;
            return Constraints(cfg_loihi_jin_84);
        };

        static Constraints createLoihiJin1024() {
            ConstraintsConfig cfg_loihi_jin_1024;
            cfg_loihi_jin_1024.name = "Loihi Jin 1024";
            cfg_loihi_jin_1024.nodes_per_part = 4096;
            cfg_loihi_jin_1024.inbound_per_part = 1024*64;
            cfg_loihi_jin_1024.max_parts = 1048576;
            return Constraints(cfg_loihi_jin_1024);
        };

        static Constraints createTrueNorth() {
            ConstraintsConfig cfg_truenorth;
            cfg_truenorth.name = "TrueNorth";
            cfg_truenorth.nodes_per_part = 256;
            cfg_truenorth.inbound_per_part = 256;
            cfg_truenorth.max_parts = 4096;
            return Constraints(cfg_truenorth);
        };
    };

    std::vector<uint32_t> partitionSequential(const HyperGraph& hg, uint32_t N, uint32_t M, uint32_t K) {
        std::vector<uint32_t> partitioning;
        partitioning.reserve(hg.nodes());

        std::unordered_set<uint32_t> inbound_edges;
        uint32_t assigned_nodes = 0;
        uint32_t current_partition = 0;

        for (std::uint32_t node = 0; node < hg.nodes(); ++node) {
            ++assigned_nodes;

            const auto& current_inbound = hg.inboundIds(node);
            inbound_edges.insert(current_inbound.begin(), current_inbound.end());

            if (assigned_nodes > N || inbound_edges.size() > M) {
                // start a new partition
                assigned_nodes = 1;
                inbound_edges.clear();
                inbound_edges.insert(current_inbound.begin(), current_inbound.end());
                ++current_partition;
                if (current_partition >= K) {
                    throw std::runtime_error("Exceeded maximum number of partitions K.");
                }
            }

            partitioning.push_back(static_cast<uint32_t>(current_partition));
        }

        return partitioning;
    }

    bool Constraints::checkFit(const HyperGraph& hg, bool already_partitioned, bool verbose) const {
        if (already_partitioned) {
            if (hg.nodes() > max_parts_) {
                if (verbose)
                    std::cout << "HG CAN'T FIT CONSTRAINTS: more partitions than allowed\n";
                return false;
            }
            for (std::uint32_t n = 0; n < hg.nodes(); ++n) {
                if (hg.inboundIds(n).size() > inbound_per_part_) {
                    if (verbose)
                        std::cout << "HG CAN'T FIT CONSTRAINTS: more inbound hedges on a partition than allowed\n";
                    return false;
                }
            }
            return true;
        }

        if (hg.nodes() > max_parts_ * nodes_per_part_) {
            if (verbose)
                std::cout << "HG CAN'T FIT CONSTRAINTS: more nodes than partitions x part-size\n";
            return false;
        }

        for (std::uint32_t n = 0; n < hg.nodes(); ++n) {
            if (hg.inboundIds(n).size() > inbound_per_part_) {
                if (verbose)
                    std::cout << "HG CAN'T FIT CONSTRAINTS: more inbound hedges on a single node than one partition can handle\n";
                return false;
            }
        }

        try {
            (void)partitionSequential(hg, nodes_per_part_, inbound_per_part_, max_parts_);
        } catch (...) {
            if (verbose)
                std::cout << "HG WON'T LIKELY FIT CONSTRAINTS: no valid way to greedily split nodes among partitions\n";
            return false;
        }
        return true;
    }

    bool Constraints::checkPartitionValidity(const HyperGraph& snn, const std::vector<uint32_t>& partitions, bool verbose) const {
        if (partitions.size() != snn.nodes())
            throw std::runtime_error("Each node must be assigned to a partition.");

        // count nodes per partition and distinct partitions
        std::unordered_map<uint32_t, uint32_t> partitions_counter;
        partitions_counter.reserve(partitions.size());
        for (uint32_t p : partitions) {
            ++partitions_counter[p];
        }

        const uint32_t partitions_count = partitions_counter.size();
        if (partitions_count > max_parts_) {
            if (verbose)
                std::cout << "INVALID PARTITIONING: more partitions (" << partitions_count << ") than allowed (" << max_parts_ << ")\n";
            return false;
        }

        // nodes per partition constraint
        for (const auto& kv : partitions_counter) {
            if (kv.second > nodes_per_part_) {
                if (verbose)
                    std::cout << "INVALID PARTITIONING: more nodes per partition (" << kv.second << ") than allowed (" << nodes_per_part_ << ")\n";
                return false;
            }
        }

        // ensure partitions are incrementally indexed from 0 onward
        for (uint32_t i = 0; i < partitions_count; ++i) {
            if (!partitions_counter.count(i))
                throw std::runtime_error("Partitions must be incrementally indexed from 0 onward.");
        }

        std::vector<uint32_t> synapses_per_partition(partitions_count, 0);

        // for each hyperedge, count distinct inbound connections per partition
        for (const auto& he : snn.hedges()) {
            std::unordered_set<uint32_t> already_seen;
            already_seen.reserve(he.length());

            for (auto node : he.destinations()) {
                uint32_t partition = partitions[node];
                if (!already_seen.count(partition)) {
                    ++synapses_per_partition[partition];
                    already_seen.insert(partition);
                }
            }
        }

        for (uint32_t i = 0; i < partitions_count; ++i) {
            if (synapses_per_partition[i] > inbound_per_part_) {
                if (verbose)
                    std::cout << "INVALID PARTITIONING: more inbound hedges per partition (" << synapses_per_partition[i] << ") than allowed (" << inbound_per_part_ << ")\n";
                return false;
            }
        }
        return true;
    }

    HedgeOverlapMetrics Constraints::hedgeOverlap(const HyperGraph& hg, const std::vector<uint32_t>& partitions) const {
        if (partitions.empty())
            return {};

        uint32_t max_partition = *std::max_element(partitions.begin(), partitions.end());
        uint32_t partitions_count = max_partition + 1;

        std::vector<uint64_t> synapses_count(partitions_count, 0);
        std::vector<uint64_t> axons_count(partitions_count, 0);

        for (const auto& he : hg.hedges()) {
            std::unordered_set<uint32_t> already_seen;
            already_seen.reserve(he.length());

            for (auto node : he.destinations()) {
                uint32_t partition = partitions[node];
                if (!already_seen.count(partition)) {
                    ++axons_count[partition];
                    already_seen.insert(partition);
                }
                ++synapses_count[partition];
            }
        }

        std::vector<double> reuse(partitions_count, 0.0);
        for (uint32_t i = 0; i < partitions_count; ++i) {
            if (axons_count[i] > 0)
                reuse[i] = static_cast<double>(synapses_count[i]) / static_cast<double>(axons_count[i]);
        }

        double ar_mean = 0.0;
        for (double r : reuse)
            ar_mean += r;
        ar_mean /= static_cast<double>(partitions_count);

        double sum_log = 0.0;
        for (double r : reuse) {
            if (r > 0.0)
                sum_log += std::log(r);
        }

        double geo_mean = std::exp(sum_log / static_cast<double>(partitions_count));

        return HedgeOverlapMetrics{ar_mean, geo_mean};
    }
}
