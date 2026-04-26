#include "nmhardware.hpp"

namespace hwmodel {
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
                    throw std::runtime_error("Exceeded maximum number of partitions K = " + std::to_string(K) + ".");
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
            throw std::runtime_error("Each neuron must be assigned to a partition (" + std::to_string(snn.nodes()) + " neuron != " + std::to_string(partitions.size()) + " part).");

        // count neurons per partition and distinct partitions
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
            throw std::runtime_error("Each partition must be assigned to a core (" + std::to_string(part_snn.nodes()) + " part != " + std::to_string(placement.size()) + " plac).");

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
            for (const auto src : he.sources()) {
                const auto& src_core = placement[src];

                for (auto dst : he.destinations()) {
                    const auto& dst_core = placement[dst];
                    int manhattan_distance = hwgeom::manhattan(src_core, dst_core);
                    double md = static_cast<double>(manhattan_distance);
                    result += static_cast<double>(he.weight()) * ((md + 1.0) * energy_per_routing_ + md * energy_per_wire_);
                }
            }
        }

        return result;
    }

    double HardwareModel::placementAverageLatency(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        double result = 0.0;
        double tot_spike_frequency = 0.0;

        for (const auto& he : part_snn.hedges()) {
            for (const auto src : he.sources()) {
                const auto& src_core = placement[src];

                double w = static_cast<double>(he.weight());
                tot_spike_frequency += w * static_cast<double>(he.connections());

                for (auto dst : he.destinations()) {
                    const auto& dst_core = placement[dst];
                    int manhattan_distance = hwgeom::manhattan(src_core, dst_core);
                    double md = static_cast<double>(manhattan_distance);
                    result += w * ((md + 1.0) * latency_per_routing_ + md * latency_per_wire_);
                }
            }
        }

        if (tot_spike_frequency > 0.0)
            return result / tot_spike_frequency;
        return 0.0;
    }

    double HardwareModel::placementMaximumLatency(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        double result = 0.0;

        for (const auto& he : part_snn.hedges()) {
            for (const auto src : he.sources()) {
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
            for (const auto src : he.sources()) {
                const auto& src_core = placement[src];

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
            for (const auto src : he.sources()) {
                const auto& src_core = placement[src];

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
