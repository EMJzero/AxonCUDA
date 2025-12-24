#pragma once
#include <cmath>
#include <vector>
#include <cstdint>
#include <numeric>
#include <iostream>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

#include "hgraph.hpp"

// GEOMETRY HELPERS

namespace hwgeom {

    struct Coord2D {
        int x{};
        int y{};

        Coord2D() = default;
        Coord2D(int x_, int y_) : x(x_), y(y_) {}

        bool operator==(const Coord2D&) const = default;
    };

    struct Coord2DHash {
        uint64_t operator()(const Coord2D& c) const noexcept {
            return (static_cast<uint64_t>(c.x) << 32) ^ static_cast<uint64_t>(c.y);
        }
    };

    inline int manhattan(const Coord2D& a, const Coord2D& b) {
        return std::abs(a.x - b.x) + std::abs(a.y - b.y);
    }

    // cross product (b - a) * (c - a)
    inline long long cross(const Coord2D& a, const Coord2D& b, const Coord2D& c) {
        long long abx = static_cast<long long>(b.x - a.x);
        long long aby = static_cast<long long>(b.y - a.y);
        long long acx = static_cast<long long>(c.x - a.x);
        long long acy = static_cast<long long>(c.y - a.y);
        return abx * acy - aby * acx;
    }

    // monotone chain convex hull, returns vertices in CCW order, no duplicate last point
    inline std::vector<Coord2D> convexHull(std::vector<Coord2D> pts) {
        if (pts.size() <= 1) return pts;

        std::sort(pts.begin(), pts.end(), [](const Coord2D& a, const Coord2D& b) {
            return (a.x < b.x) || (a.x == b.x && a.y < b.y);
        });

        std::vector<Coord2D> lower, upper;
        lower.reserve(pts.size());
        upper.reserve(pts.size());

        for (const auto& p : pts) {
            while (lower.size() >= 2 &&
                cross(lower[lower.size() - 2], lower[lower.size() - 1], p) <= 0) {
                lower.pop_back();
            }
            lower.push_back(p);
        }

        for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
            const auto& p = *it;
            while (upper.size() >= 2 &&
                cross(upper[upper.size() - 2], upper[upper.size() - 1], p) <= 0) {
                upper.pop_back();
            }
            upper.push_back(p);
        }

        lower.pop_back();
        upper.pop_back();
        lower.insert(lower.end(), upper.begin(), upper.end());
        return lower;
    }

    // check if point lies on segment [a,b]
    inline bool pointOnSegment(const Coord2D& p, const Coord2D& a, const Coord2D& b) {
        if (cross(a, b, p) != 0) return false;
        int minx = std::min(a.x, b.x), maxx = std::max(a.x, b.x);
        int miny = std::min(a.y, b.y), maxy = std::max(a.y, b.y);
        return p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy;
    }

    // point in convex polygon (inclusive), polygon must be in CCW order and non-empty
    inline bool pointInConvexPolygon(const Coord2D& p, const std::vector<Coord2D>& poly) {
        const uint32_t n = poly.size();
        if (n == 0) return false;
        if (n == 1) {
            return p.x == poly[0].x && p.y == poly[0].y;
        }
        if (n == 2) {
            return pointOnSegment(p, poly[0], poly[1]);
        }

        // Check all directed edges have cross >= 0
        for (uint32_t i = 0; i < n; ++i) {
            const Coord2D& a = poly[i];
            const Coord2D& b = poly[(i + 1) % n];
            if (cross(a, b, p) < 0) {
                return false;
            }
        }
        return true;
    }

    // count how many lattice points (x, y) with 0 <= x < width and 0 <= y < height
    // lie inside (or on the boundary of) the convex hull of "hull_points"
    inline uint32_t intersectionWithConvexHull(const std::vector<Coord2D>& hull_points, int width, int height) {
        if (hull_points.empty()) return 0;

        auto hull = convexHull(hull_points);

        if (hull.size() == 1) {
            const auto& p = hull[0];
            return (0 <= p.x && p.x < width && 0 <= p.y && p.y < height) ? 1 : 0;
        }

        int min_x = width - 1;
        int max_x = 0;
        int min_y = height - 1;
        int max_y = 0;

        for (const auto& p : hull) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }

        min_x = std::max(min_x, 0);
        min_y = std::max(min_y, 0);
        max_x = std::min(max_x, width - 1);
        max_y = std::min(max_y, height - 1);

        if (min_x > max_x || min_y > max_y) return 0;

        uint32_t count = 0;
        for (int x = min_x; x <= max_x; ++x) {
            for (int y = min_y; y <= max_y; ++y) {
                Coord2D pt{x, y};
                if (pointInConvexPolygon(pt, hull)) {
                    ++count;
                }
            }
        }
        return count;
    }

    /*
    Iterate over all coordinates in a rectangular sub-lattice diagonally.
    Starts from (x_beg, y_beg) [included] and proceeds over minor diagonals
    towards (x_end, y_end) [included/excluded depending on 'end_included'].

    Note: this coincides with an enumeration by increasing manhattan distance from (x_beg, y_beg).
    */
    inline std::vector<Coord2D> iterMajorDiagonals(int x_beg, int y_beg, int x_end, int y_end, bool end_included = false) {
        int x_sign = (x_beg < x_end) ? 1 : -1;
        int y_sign = (y_beg < y_end) ? 1 : -1;
        if (end_included) {
            x_end += x_sign;
            y_end += y_sign;
        }
        int width = std::abs(x_end - x_beg);
        int height = std::abs(y_beg - y_end);

        std::vector<Coord2D> out;
        if (width <= 0 || height <= 0) {
            // Degenerate case: no interior points to traverse
            return out;
        }
        out.reserve(static_cast<uint32_t>(width) * static_cast<uint32_t>(height));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < std::min(i + 1, width); ++j) {
                out.emplace_back(x_beg + x_sign * j,
                                y_beg + y_sign * (i - j));
            }
        }
        for (int j = 1; j < width; ++j) {
            for (int i = 0; i < std::min(width - j, height); ++i) {
                out.emplace_back(x_beg + x_sign * (j + i),
                                y_end - y_sign * (1 + i));
            }
        }
        return out;
    }
}


// HARDWARE MODEL (constraints & costs)

namespace hwmodel {

    using namespace hgraph;

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
    
    struct HardwareModelConfig {
        std::string name;
        uint32_t neurons_per_core;
        uint32_t synapses_per_core;
        uint32_t cores_per_chip_x;
        uint32_t cores_per_chip_y;
        uint32_t chips_per_system_x;
        uint32_t chips_per_system_y;
        double energy_per_routing;
        double energy_per_wire;
        double latency_per_routing;
        double latency_per_wire;
    };

    // partitionSequential: sequential greedy partitioning with constraints on
    // max neurons (N), max inbound synapses (M), and max partitions (K).
    std::vector<uint32_t> partitionSequential(const HyperGraph& hg, uint32_t N, uint32_t M, uint32_t K);

    class HardwareModel {
        private:
        std::string name_;
        // CONSTRAINTS
        uint32_t neurons_per_core_;
        uint32_t synapses_per_core_;
        uint32_t cores_per_chip_x_;
        uint32_t cores_per_chip_y_;
        uint32_t chips_per_system_x_;
        uint32_t chips_per_system_y_;

        // COSTS
        double energy_per_routing_; // in [pJ]
        double energy_per_wire_; // in [pJ]
        double latency_per_routing_; // in [ns]
        double latency_per_wire_; // in [ns]

        public:
        HardwareModel(const HardwareModelConfig& cfg) :
            name_(cfg.name),
            neurons_per_core_(cfg.neurons_per_core),
            synapses_per_core_(cfg.synapses_per_core),
            cores_per_chip_x_(cfg.cores_per_chip_x),
            cores_per_chip_y_(cfg.cores_per_chip_y),
            chips_per_system_x_(cfg.chips_per_system_x),
            chips_per_system_y_(cfg.chips_per_system_y),
            energy_per_routing_(cfg.energy_per_routing),
            energy_per_wire_(cfg.energy_per_wire),
            latency_per_routing_(cfg.latency_per_routing),
            latency_per_wire_(cfg.latency_per_wire)
        {
            if (!(neurons_per_core_ > 0 && synapses_per_core_ > 0 &&
                cores_per_chip_x_ > 0 && cores_per_chip_y_ > 0 &&
                chips_per_system_x_ > 0 && chips_per_system_y_ > 0)) {
                throw std::invalid_argument("All hardware specifications must be > 0.");
            }
            if (!(energy_per_routing_ >= 0.0 && energy_per_wire_ >= 0.0 &&
                latency_per_routing_ >= 0.0 && latency_per_wire_ >= 0.0)) {
                throw std::invalid_argument("All hardware costs must be >= 0.");
            }
            if (!(chips_per_system_x_ == 1 && chips_per_system_y_ == 1)) {
                throw std::invalid_argument(
                    "Functionality not yet implemented: chips_per_system_x and chips_per_system_y must be 1."
                );
            }
        }

        // accessors
        std::string name() const { return name_; }
        uint32_t neuronsPerCore() const { return neurons_per_core_; }
        uint32_t synapsesPerCore() const { return synapses_per_core_; }
        uint32_t coresPerChipX() const { return cores_per_chip_x_; }
        uint32_t coresPerChipY() const { return cores_per_chip_y_; }
        uint32_t chipsPerSystemX() const { return chips_per_system_x_; }
        uint32_t chipsPerSystemY() const { return chips_per_system_y_; }

        double energyPerRouting() const { return energy_per_routing_; }
        double energyPerWire() const { return energy_per_wire_; }
        double latencyPerRouting() const { return latency_per_routing_; }
        double latencyPerWire() const { return latency_per_wire_; }

        uint32_t coresCount() const { return (cores_per_chip_x_ * cores_per_chip_y_) * chips_per_system_x_ * chips_per_system_y_; }
        uint32_t coresPerChipCount() const { return cores_per_chip_x_ * cores_per_chip_y_; }
        uint32_t chipsCount() const { return chips_per_system_x_ * chips_per_system_y_; }
        uint32_t coresAlongX() const { return cores_per_chip_x_ * chips_per_system_x_; }
        uint32_t coresAlongY() const { return cores_per_chip_y_ * chips_per_system_y_; }

        // empirical: can have false negatives (no false positives tho)
        bool checkSnnFit(const HyperGraph& snn,
                        bool already_partitioned = false,
                        bool verbose = false) const;

        bool checkPartitionValidity(const HyperGraph& snn,
                                    const std::vector<uint32_t>& partitions,
                                    bool verbose = false) const;

        bool checkPlacementValidity(const HyperGraph& part_snn,
                                    const std::vector<hwgeom::Coord2D>& placement,
                                    bool verbose = false) const;

        SynapticReuseMetrics synapticReuse(const HyperGraph& snn,
                                        const std::vector<uint32_t>& partitions) const;

        ConnectionsLocalityMetrics connectionsLocality(
            const HyperGraph& part_snn,
            const std::vector<hwgeom::Coord2D>& placement) const;

        double placementEnergyConsumption(
            const HyperGraph& part_snn,
            const std::vector<hwgeom::Coord2D>& placement) const;

        double placementAverageLatency(
            const HyperGraph& part_snn,
            const std::vector<hwgeom::Coord2D>& placement) const;

        double placementMaximumLatency(
            const HyperGraph& part_snn,
            const std::vector<hwgeom::Coord2D>& placement) const;

        double placementAverageCongestion(
            const HyperGraph& part_snn,
            const std::vector<hwgeom::Coord2D>& placement) const;

        double placementMaximumCongestion(
            const HyperGraph& part_snn,
            const std::vector<hwgeom::Coord2D>& placement) const;

        // computes the matrix of probabilities of a spike to pass through core (x, y) when going from cores (x_src, y_src) to core (x_dst, y_dst);
        // the element (x, y) in the matrix matches to (min(x_src, x_dst) + x, min(y_src, y_dst) + y) on the original lattice
        std::vector<std::vector<double>> expectedSpikeTransitProbability(
            int x_src, int y_src, int x_dst, int y_dst) const;

        PlacementMetrics getAllMetrics(
            const HyperGraph& part_snn,
            const std::vector<hwgeom::Coord2D>& placement) const;

        // predefined harwdare models
        static HardwareModel createLoihiLarge() {
            HardwareModelConfig cfg_loihi_large;
            cfg_loihi_large.name = "Loihi Large";
            cfg_loihi_large.neurons_per_core = 1024;
            cfg_loihi_large.synapses_per_core = 4096;
            cfg_loihi_large.cores_per_chip_x = 64;
            cfg_loihi_large.cores_per_chip_y = 64;
            cfg_loihi_large.chips_per_system_x = 1;
            cfg_loihi_large.chips_per_system_y = 1;
            cfg_loihi_large.energy_per_routing = 1.7;
            cfg_loihi_large.energy_per_wire = 3.5;
            cfg_loihi_large.latency_per_routing = 2.1;
            cfg_loihi_large.latency_per_wire = 5.3;
            return HardwareModel(cfg_loihi_large);
        };

        static HardwareModel createLoihiJin84() {
            HardwareModelConfig cfg_loihi_jin_84;
            cfg_loihi_jin_84.name = "Loihi Jin 84";
            cfg_loihi_jin_84.neurons_per_core = 4096;
            cfg_loihi_jin_84.synapses_per_core = 1024*64;
            cfg_loihi_jin_84.cores_per_chip_x = 84;
            cfg_loihi_jin_84.cores_per_chip_y = 84;
            cfg_loihi_jin_84.chips_per_system_x = 1;
            cfg_loihi_jin_84.chips_per_system_y = 1;
            cfg_loihi_jin_84.energy_per_routing = 1.0;
            cfg_loihi_jin_84.energy_per_wire = 0.1;
            cfg_loihi_jin_84.latency_per_routing = 1.0;
            cfg_loihi_jin_84.latency_per_wire = 0.01;
            return HardwareModel(cfg_loihi_jin_84);
        };

        static HardwareModel createTrueNorth() {
            HardwareModelConfig cfg_truenorth;
            cfg_truenorth.name = "TrueNorth";
            cfg_truenorth.neurons_per_core = 256;
            cfg_truenorth.synapses_per_core = 256;
            cfg_truenorth.cores_per_chip_x = 64;
            cfg_truenorth.cores_per_chip_y = 64;
            cfg_truenorth.chips_per_system_x = 1;
            cfg_truenorth.chips_per_system_y = 1;
            cfg_truenorth.energy_per_routing = 1.7; // unknown (this is from Loihi)
            cfg_truenorth.energy_per_wire = 3.5; // unknown (this is from Loihi)
            cfg_truenorth.latency_per_routing = 2.1; // unknown (this is from Loihi)
            cfg_truenorth.latency_per_wire = 5.3; // unknown (this is from Loihi)
            return HardwareModel(cfg_truenorth);
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

    bool HardwareModel::checkSnnFit(const HyperGraph& snn, bool already_partitioned, bool verbose) const {
        if (already_partitioned) {
            if (snn.nodes() > coresCount()) {
                if (verbose)
                    std::cout << "SNN CAN'T FIT ON THE HW: more neuron clusters than the HW cores\n";
                return false;
            }
            for (std::uint32_t n = 0; n < snn.nodes(); ++n) {
                if (snn.inboundIds(n).size() > synapses_per_core_) {
                    if (verbose)
                        std::cout << "SNN CAN'T FIT ON THE HW: more inbound synapses on a neuron cluster than the HW can handle\n";
                    return false;
                }
            }
            return true;
        }

        if (snn.nodes() > coresCount() * neurons_per_core_) {
            if (verbose)
                std::cout << "SNN CAN'T FIT ON THE HW: more neurons than the HW can house\n";
            return false;
        }

        for (std::uint32_t n = 0; n < snn.nodes(); ++n) {
            if (snn.inboundIds(n).size() > synapses_per_core_) {
                if (verbose)
                    std::cout << "SNN CAN'T FIT ON THE HW: more inbound synapses on a single neuron than the HW can handle\n";
                return false;
            }
        }

        try {
            (void)partitionSequential(snn, neurons_per_core_, synapses_per_core_, coresCount());
        } catch (...) {
            if (verbose)
                std::cout << "SNN WON'T LIKELY FIT ON THE HW: no valid way to split neurons (and their synapses) among cores\n";
            return false;
        }
        return true;
    }

    bool HardwareModel::checkPartitionValidity(const HyperGraph& snn, const std::vector<uint32_t>& partitions, bool verbose) const {
        if (partitions.size() != snn.nodes())
            throw std::runtime_error("Each neuron must be assigned to a partition.");

        // count neurons per partition and distinct partitions.
        std::unordered_map<uint32_t, uint32_t> partitions_counter;
        partitions_counter.reserve(partitions.size());
        for (uint32_t p : partitions) {
            ++partitions_counter[p];
        }

        const uint32_t partitions_count = partitions_counter.size();
        if (partitions_count > coresCount()) {
            if (verbose)
                std::cout << "INVALID PARTITIONING: more partitions (" << partitions_count << ") than cores (" << coresCount() << ")\n";
            return false;
        }

        // neurons per partition constraint
        for (const auto& kv : partitions_counter) {
            if (kv.second > neurons_per_core_) {
                if (verbose)
                    std::cout << "INVALID PARTITIONING: more neurons per partition (" << kv.second << ") than a core can store (" << neurons_per_core_ << ")\n";
                return false;
            }
        }

        // ensure partitions are incrementally indexed from 0 onward
        for (uint32_t i = 0; i < partitions_count; ++i) {
            if (!partitions_counter.count(i))
                throw std::runtime_error("Partitions must be incrementally indexed from 0 onward.");
        }

        std::vector<uint32_t> synapses_per_partition(partitions_count, 0);

        // for each hyperedge, count distinct inbound synapses per partition
        for (const auto& he : snn.hedges()) {
            std::unordered_set<uint32_t> already_seen;
            already_seen.reserve(he.length());

            for (auto neuron : he.destinations()) {
                uint32_t partition = partitions[neuron];
                if (!already_seen.count(partition)) {
                    ++synapses_per_partition[partition];
                    already_seen.insert(partition);
                }
            }
        }

        for (uint32_t i = 0; i < partitions_count; ++i) {
            if (synapses_per_partition[i] > synapses_per_core_) {
                if (verbose)
                    std::cout << "INVALID PARTITIONING: more inbound synapses per partition (" << synapses_per_partition[i] << ") than a core can handle (" << synapses_per_core_ << ")\n";
                return false;
            }
        }
        return true;
    }

    bool HardwareModel::checkPlacementValidity(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement, bool verbose) const {
        if (placement.size() != part_snn.nodes())
            throw std::runtime_error("Each partition must be assigned to a core.");

        std::unordered_set<hwgeom::Coord2D, hwgeom::Coord2DHash> seen_cores;
        seen_cores.reserve(placement.size());

        const uint32_t max_x = coresAlongX();
        const uint32_t max_y = coresAlongY();

        for (const auto& core : placement) {
            if (core.x < 0 || core.y < 0 ||
                static_cast<uint32_t>(core.x) >= max_x ||
                static_cast<uint32_t>(core.y) >= max_y) {
                if (verbose)
                    std::cout << "INVALID PLACEMENT: a core's coordinates are out of the hardware's range\n";
                return false;
            }
            auto [it, inserted] = seen_cores.insert(core);
            if (!inserted) {
                if (verbose)
                    std::cout << "INVALID PLACEMENT: a core is used more than once\n";
                return false;
            }
        }
        return true;
    }

    SynapticReuseMetrics HardwareModel::synapticReuse(const HyperGraph& snn, const std::vector<uint32_t>& partitions) const {
        if (partitions.empty())
            return {};

        uint32_t max_partition = *std::max_element(partitions.begin(), partitions.end());
        uint32_t partitions_count = max_partition + 1;

        std::vector<uint64_t> synapses_count(partitions_count, 0);
        std::vector<uint64_t> axons_count(partitions_count, 0);

        for (const auto& he : snn.hedges()) {
            std::unordered_set<uint32_t> already_seen;
            already_seen.reserve(he.length());

            for (auto neuron : he.destinations()) {
                uint32_t partition = partitions[neuron];
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

        return SynapticReuseMetrics{ar_mean, geo_mean};
    }

    ConnectionsLocalityMetrics HardwareModel::connectionsLocality(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        double ar_mean = 0.0;
        double geo_mean = 0.0;
        double ar_mean_weighted = 0.0;
        double geo_mean_weighted = 0.0;
        double weights_sum = 0.0;

        const auto& hedges = part_snn.hedges();
        if (hedges.empty())
            return {};

        for (const auto& he : hedges) {
            std::vector<hwgeom::Coord2D> pts;
            auto nodes = he.nodes();
            pts.reserve(nodes.size());
            for (auto n : nodes)
                pts.push_back(placement[n]);

            int traversed_cores = hwgeom::intersectionWithConvexHull(pts, static_cast<int>(coresAlongX()), static_cast<int>(coresAlongY()));

            // should not happen for a valid placement, but ...
            if (traversed_cores <= 0)
                continue;

            double t = static_cast<double>(traversed_cores);
            double w = static_cast<double>(he.weight());

            ar_mean += t;
            geo_mean += std::log(t);
            ar_mean_weighted += t * w;
            geo_mean_weighted += std::log(t) * w;
            weights_sum += w;
        }

        uint32_t hedge_count = hedges.size();
        ar_mean /= static_cast<double>(hedge_count);
        geo_mean = std::exp(geo_mean / static_cast<double>(hedge_count));

        if (weights_sum > 0.0) {
            ar_mean_weighted /= weights_sum;
            geo_mean_weighted = std::exp(geo_mean_weighted / weights_sum);
        }

        return ConnectionsLocalityMetrics{ar_mean, geo_mean, ar_mean_weighted, geo_mean_weighted};
    }

    double HardwareModel::placementEnergyConsumption(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        double result = 0.0;

        for (const auto& he : part_snn.hedges()) {
            auto src = he.source();
            const auto& src_core = placement[src];

            for (auto dst : he.destinations()) {
                const auto& dst_core = placement[dst];
                int manhattan_distance = hwgeom::manhattan(src_core, dst_core);
                double md = static_cast<double>(manhattan_distance);
                result += static_cast<double>(he.weight()) * ((md + 1.0) * energy_per_routing_ + md * energy_per_wire_);
            }
        }

        return result;
    }

    double HardwareModel::placementAverageLatency(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        double result = 0.0;
        double tot_spike_frequency = 0.0;

        for (const auto& he : part_snn.hedges()) {
            auto src = he.source();
            const auto& src_core = placement[src];

            double w = static_cast<double>(he.weight());
            // connections() == length() - 1
            tot_spike_frequency += w * static_cast<double>(he.length() - 1);

            for (auto dst : he.destinations()) {
                const auto& dst_core = placement[dst];
                int manhattan_distance = hwgeom::manhattan(src_core, dst_core);
                double md = static_cast<double>(manhattan_distance);
                result += w * ((md + 1.0) * latency_per_routing_ + md * latency_per_wire_);
            }
        }

        if (tot_spike_frequency > 0.0)
            return result / tot_spike_frequency;
        return 0.0;
    }

    double HardwareModel::placementMaximumLatency(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        double result = 0.0;

        for (const auto& he : part_snn.hedges()) {
            auto src = he.source();
            const auto& src_core = placement[src];

            for (auto dst : he.destinations()) {
                const auto& dst_core = placement[dst];
                int manhattan_distance = hwgeom::manhattan(src_core, dst_core);
                double md = static_cast<double>(manhattan_distance);
                double latency = (md + 1.0) * latency_per_routing_ + md * latency_per_wire_;
                if (latency > result)
                    result = latency;
            }
        }

        return result;
    }

    std::vector<std::vector<double>> HardwareModel::expectedSpikeTransitProbability(int x_src, int y_src, int x_dst, int y_dst) const {
        int width  = std::abs(x_dst - x_src) + 1;
        int height = std::abs(y_dst - y_src) + 1;

        std::vector<std::vector<double>> matrix(width, std::vector<double>(height, 0.0));

        matrix[0][0] = 1.0;

        int m = std::min(x_src, x_dst);
        x_src -= m;
        x_dst -= m;
        m = std::min(y_src, y_dst);
        y_src -= m;
        y_dst -= m;

        int x_sign = (x_src == 0) ? 1 : -1;
        int y_sign = (y_src == 0) ? 1 : -1;

        auto diag_points = hwgeom::iterMajorDiagonals(x_src, y_src, x_dst, y_dst, true);

        for (const auto& p : diag_points) {
            int x = p.x;
            int y = p.y;

            if (x == x_dst && y != y_dst) {
                matrix[x][y + y_sign] += matrix[x][y];
            } else if (x != x_dst && y == y_dst) {
                matrix[x + x_sign][y] += matrix[x][y];
            } else if (x != x_dst && y != y_dst) {
                matrix[x + x_sign][y] += matrix[x][y] * 0.5;
                matrix[x][y + y_sign] += matrix[x][y] * 0.5;
            }
        }

        return matrix;
    }

    double HardwareModel::placementAverageCongestion(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        double result = 0.0;

        for (const auto& he : part_snn.hedges()) {
            const auto& src_core = placement[he.source()];

            for (auto dst : he.destinations()) {
                const auto& dst_core = placement[dst];
                auto transit_prob_matrix = expectedSpikeTransitProbability(src_core.x, src_core.y, dst_core.x, dst_core.y);

                double sum_probs = 0.0;
                for (const auto& row : transit_prob_matrix) {
                    for (double v : row) {
                        sum_probs += v;
                    }
                }

                result += static_cast<double>(he.weight()) * sum_probs;
            }
        }

        if (coresCount() > 0)
            return result / static_cast<double>(coresCount());
        return 0.0;
    }

    double HardwareModel::placementMaximumCongestion(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        uint32_t max_x = coresAlongX();
        uint32_t max_y = coresAlongY();

        std::vector<std::vector<double>> congestion_matrix(max_x, std::vector<double>(max_y, 0.0));

        for (const auto& he : part_snn.hedges()) {
            const auto& src_core = placement[he.source()];

            for (auto dst : he.destinations()) {
                const auto& dst_core = placement[dst];
                auto transit_prob_matrix = expectedSpikeTransitProbability(src_core.x, src_core.y, dst_core.x, dst_core.y);

                int dx = std::abs(dst_core.x - src_core.x);
                int dy = std::abs(dst_core.y - src_core.y);

                int x_base = std::min(dst_core.x, src_core.x);
                int y_base = std::min(dst_core.y, src_core.y);

                for (int x = 0; x < dx; ++x) {
                    for (int y = 0; y < dy; ++y) {
                        congestion_matrix[static_cast<uint32_t>(x_base + x)][static_cast<uint32_t>(y_base + y)] += static_cast<double>(he.weight()) * transit_prob_matrix[x][y];
                    }
                }
            }
        }

        double max_congestion = 0.0;
        for (const auto& row : congestion_matrix) {
            for (double v : row) {
                if (v > max_congestion)
                    max_congestion = v;
            }
        }
        return max_congestion;
    }

    PlacementMetrics HardwareModel::getAllMetrics(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        PlacementMetrics metrics;

        bool valid = checkPlacementValidity(part_snn, placement);
        metrics.valid = valid;

        if (!valid)
            return metrics;

        metrics.energy = placementEnergyConsumption(part_snn, placement);
        metrics.avg_latency = placementAverageLatency(part_snn, placement);
        metrics.max_latency = placementMaximumLatency(part_snn, placement);
        metrics.avg_congestion = placementAverageCongestion(part_snn, placement);
        metrics.max_congestion  = placementMaximumCongestion(part_snn, placement);
        metrics.connections_locality = connectionsLocality(part_snn, placement);
        return metrics;
    }
}
