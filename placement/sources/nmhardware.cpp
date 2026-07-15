#include "nmhardware.hpp"

#include <omp.h>

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
                const uint32_t destinations_count = he.length() - he.src_count();
                tot_spike_frequency += w * static_cast<double>(destinations_count);

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

        const int x_base = std::min(x_src, x_dst);
        const int y_base = std::min(y_src, y_dst);
        x_src -= x_base;
        x_dst -= x_base;
        y_src -= y_base;
        y_dst -= y_base;

        matrix[x_src][y_src] = 1.0;

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

                    for (int x = 0; x <= dx; ++x) {
                        for (int y = 0; y <= dy; ++y) {
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

    PlacementMetrics HardwareModel::getAllUnicastMetrics(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
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

    namespace {
        constexpr uint32_t MAX_STEINER_STATES = 2000000;
        constexpr long double LOG_ZERO = -std::numeric_limits<long double>::infinity();

        struct SteinerStatistics {
            uint32_t hops{0};
            std::vector<std::pair<uint32_t, double>> transit_probability;
        };

        struct FrontierState {
            std::vector<uint16_t> labels;
            bool complete{false};

            bool operator==(const FrontierState&) const = default;
        };

        struct FrontierStateHash {
            size_t operator()(const FrontierState& state) const noexcept {
                size_t h = state.complete ? 1u : 0u;
                for (uint16_t label : state.labels)
                    h ^= static_cast<size_t>(label) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return h;
            }
        };

        struct DecisionArc {
            uint32_t predecessor;
            uint32_t position;
            bool included;
        };

        struct DecisionNode {
            uint32_t cost{UINT32_MAX};
            std::vector<DecisionArc> incoming;
        };

        struct MulticastEvaluation {
            double energy{0.0};
            double weighted_latency{0.0};
            double weight{0.0};
            std::vector<double> congestion;
        };

        struct MulticastTask {
            uint32_t hedge;
            uint32_t source;
        };

        // numerically stable addition of two values represented in the logarithmic domain
        long double logAdd(long double a, long double b) {
            if (a == LOG_ZERO) return b;
            if (b == LOG_ZERO) return a;
            if (a < b) std::swap(a, b);
            return a + std::log1p(std::exp(b - a));
        }

        // logarithm of the binomial coefficient, used to count minimum two-terminal paths without overflowing
        long double logBinomial(uint32_t n, uint32_t k) {
            if (k > n) return LOG_ZERO;
            return std::lgamma(static_cast<long double>(n) + 1.0L)
                 - std::lgamma(static_cast<long double>(k) + 1.0L)
                 - std::lgamma(static_cast<long double>(n - k) + 1.0L);
        }

        // rename frontier components by first occurrence so equivalent connectivity states compare equal
        void canonicalizeLabels(std::vector<uint16_t>& labels) {
            std::vector<uint16_t> renamed(labels.size() + 1u, 0u);
            uint16_t next = 1u;
            for (uint16_t& label : labels) {
                if (label == 0u) continue;
                if (renamed[label] == 0u)
                    renamed[label] = next++;
                label = renamed[label];
            }
        }

        // exact minimum-tree statistics for a set of terminals on a rectangular lattice
        SteinerStatistics exactSteinerStatistics(
            std::vector<hwgeom::Coord2D> terminals,
            uint32_t cores_y
        ) {
            std::sort(terminals.begin(), terminals.end(), [](const auto& a, const auto& b) {
                return a.x < b.x || (a.x == b.x && a.y < b.y);
            });
            terminals.erase(std::unique(terminals.begin(), terminals.end()), terminals.end());

            SteinerStatistics result;
            if (terminals.empty())
                return result;

            int min_x = terminals.front().x;
            int max_x = terminals.front().x;
            int min_y = terminals.front().y;
            int max_y = terminals.front().y;
            for (const auto& terminal : terminals) {
                min_x = std::min(min_x, terminal.x);
                max_x = std::max(max_x, terminal.x);
                min_y = std::min(min_y, terminal.y);
                max_y = std::max(max_y, terminal.y);
            }

            const uint32_t span_x = static_cast<uint32_t>(max_x - min_x + 1);
            const uint32_t span_y = static_cast<uint32_t>(max_y - min_y + 1);

            // a single terminal and collinear terminal sets have one minimum core set
            if (span_x == 1u || span_y == 1u) {
                result.hops = span_x + span_y - 2u;
                for (int x = min_x; x <= max_x; ++x) {
                    for (int y = min_y; y <= max_y; ++y) {
                        uint32_t core = static_cast<uint32_t>(x) * cores_y + static_cast<uint32_t>(y);
                        result.transit_probability.emplace_back(core, 1.0);
                    }
                }
                return result;
            }

            // every minimum two-terminal tree is a monotone path; binomial products give its core marginals
            if (terminals.size() == 2u) {
                const auto& source = terminals[0];
                const auto& destination = terminals[1];
                const uint32_t total_dx = static_cast<uint32_t>(std::abs(destination.x - source.x));
                const uint32_t total_dy = static_cast<uint32_t>(std::abs(destination.y - source.y));
                const uint32_t distance = total_dx + total_dy;
                const long double log_paths = logBinomial(distance, total_dx);

                result.hops = distance;
                result.transit_probability.reserve(static_cast<size_t>(span_x) * span_y);
                for (int x = min_x; x <= max_x; ++x) {
                    for (int y = min_y; y <= max_y; ++y) {
                        hwgeom::Coord2D core_coord{x, y};
                        const uint32_t first_dx = static_cast<uint32_t>(std::abs(x - source.x));
                        const uint32_t first_dy = static_cast<uint32_t>(std::abs(y - source.y));
                        const uint32_t second_dx = static_cast<uint32_t>(std::abs(destination.x - x));
                        const uint32_t second_dy = static_cast<uint32_t>(std::abs(destination.y - y));
                        long double log_containing = logBinomial(first_dx + first_dy, first_dx)
                                                   + logBinomial(second_dx + second_dy, second_dx);
                        double probability = static_cast<double>(std::exp(log_containing - log_paths));
                        uint32_t core = static_cast<uint32_t>(core_coord.x) * cores_y + static_cast<uint32_t>(core_coord.y);
                        result.transit_probability.emplace_back(core, probability);
                    }
                }
                return result;
            }

            // scan along the longest dimension to keep the connectivity frontier as narrow as possible
            const bool transpose = span_y < span_x;
            const uint32_t frontier_width = transpose ? span_y : span_x;
            const uint32_t rows = transpose ? span_x : span_y;
            const uint32_t area = frontier_width * rows;

            auto coordAt = [&](uint32_t position) {
                uint32_t row = position / frontier_width;
                uint32_t column = position % frontier_width;
                if (transpose)
                    return hwgeom::Coord2D{min_x + static_cast<int>(row), min_y + static_cast<int>(column)};
                return hwgeom::Coord2D{min_x + static_cast<int>(column), min_y + static_cast<int>(row)};
            };

            std::vector<uint8_t> required(area, 0u);
            for (uint32_t position = 0; position < area; ++position) {
                hwgeom::Coord2D coord = coordAt(position);
                required[position] = std::binary_search(terminals.begin(), terminals.end(), coord, [](const auto& a, const auto& b) {
                    return a.x < b.x || (a.x == b.x && a.y < b.y);
                });
            }

            std::vector<uint32_t> required_suffix(area + 1u, 0u);
            for (uint32_t position = area; position-- > 0u;)
                required_suffix[position] = required_suffix[position + 1u] + required[position];

            std::vector<DecisionNode> nodes;
            nodes.reserve(std::min<uint32_t>(MAX_STEINER_STATES, area * 64u));
            nodes.push_back(DecisionNode{0u, {}});

            std::unordered_map<FrontierState, uint32_t, FrontierStateHash> current;
            std::unordered_map<FrontierState, uint32_t, FrontierStateHash> next;
            current.emplace(FrontierState{std::vector<uint16_t>(frontier_width, 0u), false}, 0u);

            auto insertState = [&](FrontierState state, uint32_t predecessor, uint32_t position, bool included) {
                canonicalizeLabels(state.labels);
                uint32_t cost = nodes[predecessor].cost + static_cast<uint32_t>(included);
                auto [it, inserted] = next.emplace(std::move(state), static_cast<uint32_t>(nodes.size()));
                if (inserted) {
                    if (nodes.size() >= MAX_STEINER_STATES)
                        throw std::runtime_error("Exact Steiner solver exceeded its frontier-state limit.");
                    nodes.push_back(DecisionNode{cost, {{predecessor, position, included}}});
                } else {
                    DecisionNode& node = nodes[it->second];
                    if (cost < node.cost) {
                        node.cost = cost;
                        node.incoming.clear();
                        node.incoming.push_back({predecessor, position, included});
                    } else if (cost == node.cost) {
                        node.incoming.push_back({predecessor, position, included});
                    }
                }
            };

            for (uint32_t position = 0; position < area; ++position) {
                const uint32_t column = position % frontier_width;
                next.clear();

                for (const auto& [state, predecessor] : current) {
                    // exclude the current core; a closed component can never reconnect later
                    if (!required[position]) {
                        FrontierState excluded = state;
                        bool valid = true;
                        if (!excluded.complete) {
                            uint16_t removed = excluded.labels[column];
                            excluded.labels[column] = 0u;
                            if (removed != 0u && std::find(excluded.labels.begin(), excluded.labels.end(), removed) == excluded.labels.end()) {
                                bool another_component = std::any_of(excluded.labels.begin(), excluded.labels.end(), [](uint16_t x) { return x != 0u; });
                                if (another_component || required_suffix[position + 1u] != 0u)
                                    valid = false;
                                else
                                    excluded.complete = true;
                            }
                        }
                        if (valid)
                            insertState(std::move(excluded), predecessor, position, false);
                    }

                    // include the current core and join its already processed left/up neighbors
                    if (!state.complete) {
                        FrontierState included = state;
                        uint16_t up = included.labels[column];
                        uint16_t left = column == 0u ? 0u : included.labels[column - 1u];
                        uint16_t label = up != 0u ? up : left;

                        if (label == 0u) {
                            label = 1u;
                            for (uint16_t existing : included.labels)
                                label = std::max<uint16_t>(label, existing + 1u);
                        } else if (up != 0u && left != 0u && up != left) {
                            uint16_t keep = std::min(up, left);
                            uint16_t merge = std::max(up, left);
                            for (uint16_t& existing : included.labels) {
                                if (existing == merge)
                                    existing = keep;
                            }
                            label = keep;
                        }
                        included.labels[column] = label;
                        insertState(std::move(included), predecessor, position, true);
                    }
                }

                current.swap(next);
                if (current.empty())
                    throw std::runtime_error("Exact Steiner solver found no connected terminal set.");
            }

            uint32_t optimum = UINT32_MAX;
            std::vector<uint32_t> accepting;
            for (const auto& [state, node_id] : current) {
                uint16_t component = 0u;
                bool connected = state.complete;
                if (!connected) {
                    connected = true;
                    for (uint16_t label : state.labels) {
                        if (label == 0u) continue;
                        if (component == 0u) component = label;
                        else if (component != label) connected = false;
                    }
                    connected = connected && component != 0u;
                }
                if (!connected) continue;

                uint32_t cost = nodes[node_id].cost;
                if (cost < optimum) {
                    optimum = cost;
                    accepting.clear();
                    accepting.push_back(node_id);
                } else if (cost == optimum) {
                    accepting.push_back(node_id);
                }
            }
            if (accepting.empty())
                throw std::runtime_error("Exact Steiner solver found no minimum tree.");

            std::vector<long double> forward(nodes.size(), LOG_ZERO);
            forward[0] = 0.0L;
            for (uint32_t node_id = 1u; node_id < nodes.size(); ++node_id) {
                for (const auto& arc : nodes[node_id].incoming)
                    forward[node_id] = logAdd(forward[node_id], forward[arc.predecessor]);
            }

            long double total_log_count = LOG_ZERO;
            std::vector<long double> backward(nodes.size(), LOG_ZERO);
            for (uint32_t node_id : accepting) {
                total_log_count = logAdd(total_log_count, forward[node_id]);
                backward[node_id] = 0.0L;
            }

            std::vector<long double> containing_log_count(area, LOG_ZERO);
            for (uint32_t node_id = static_cast<uint32_t>(nodes.size()); node_id-- > 1u;) {
                if (backward[node_id] == LOG_ZERO) continue;
                for (const auto& arc : nodes[node_id].incoming) {
                    backward[arc.predecessor] = logAdd(backward[arc.predecessor], backward[node_id]);
                    if (arc.included) {
                        long double paths = forward[arc.predecessor] + backward[node_id];
                        containing_log_count[arc.position] = logAdd(containing_log_count[arc.position], paths);
                    }
                }
            }

            result.hops = optimum - 1u;
            result.transit_probability.reserve(area);
            for (uint32_t position = 0; position < area; ++position) {
                if (containing_log_count[position] == LOG_ZERO) continue;
                hwgeom::Coord2D coord = coordAt(position);
                uint32_t core = static_cast<uint32_t>(coord.x) * cores_y + static_cast<uint32_t>(coord.y);
                double probability = static_cast<double>(std::exp(containing_log_count[position] - total_log_count));
                result.transit_probability.emplace_back(core, probability);
            }
            return result;
        }

        // evaluate every source-multicast once, optionally accumulating the expensive Steiner-derived metrics
        MulticastEvaluation evaluateMulticast(
            const HyperGraph& part_snn,
            const std::vector<hwgeom::Coord2D>& placement,
            uint32_t cores_count,
            uint32_t cores_y,
            double energy_per_routing,
            double energy_per_wire,
            double latency_per_routing,
            double latency_per_wire,
            bool need_energy,
            bool need_latency,
            bool need_congestion
        ) {
            std::vector<MulticastTask> tasks;
            for (uint32_t hedge = 0; hedge < part_snn.hedges().size(); ++hedge) {
                const auto& he = part_snn.hedges()[hedge];
                if (!std::isfinite(he.weight()))
                    throw std::invalid_argument("Multicast metrics require finite hyperedge weights.");
                for (uint32_t source : he.sources())
                    tasks.push_back({hedge, source});
            }

            const int thread_count = std::max<int>(1, std::min<size_t>(static_cast<size_t>(omp_get_max_threads()), tasks.size()));
            std::vector<MulticastEvaluation> locals(static_cast<size_t>(thread_count));
            std::exception_ptr failure;

            #pragma omp parallel num_threads(thread_count)
            {
                const int tid = omp_get_thread_num();
                auto& local = locals[static_cast<size_t>(tid)];
                if (need_congestion)
                    local.congestion.assign(cores_count, 0.0);

                #pragma omp for schedule(dynamic)
                for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
                    try {
                        const auto& task = tasks[task_idx];
                        const auto& he = part_snn.hedges()[task.hedge];
                        const double weight = static_cast<double>(he.weight());
                        std::vector<uint32_t> destinations = he.destinations();

                        if (need_latency) {
                            int max_distance = 0;
                            for (uint32_t destination : destinations)
                                max_distance = std::max(max_distance, hwgeom::manhattan(placement[task.source], placement[destination]));
                            double latency = static_cast<double>(max_distance) * (latency_per_routing + latency_per_wire) + latency_per_routing;
                            local.weight += weight;
                            local.weighted_latency += weight * latency;
                        }

                        if (need_energy || need_congestion) {
                            std::vector<hwgeom::Coord2D> terminals;
                            terminals.reserve(destinations.size() + 1u);
                            terminals.push_back(placement[task.source]);
                            for (uint32_t destination : destinations)
                                terminals.push_back(placement[destination]);

                            SteinerStatistics stats = exactSteinerStatistics(std::move(terminals), cores_y);
                            if (need_energy) {
                                double energy = static_cast<double>(stats.hops) * (energy_per_routing + energy_per_wire) + energy_per_routing;
                                local.energy += weight * energy;
                            }
                            if (need_congestion) {
                                for (const auto& [core, probability] : stats.transit_probability)
                                    local.congestion[core] += weight * probability;
                            }
                        }
                    } catch (...) {
                        #pragma omp critical(multicast_metrics_failure)
                        {
                            if (!failure)
                                failure = std::current_exception();
                        }
                    }
                }
            }

            if (failure)
                std::rethrow_exception(failure);

            MulticastEvaluation result;
            if (need_congestion)
                result.congestion.assign(cores_count, 0.0);
            for (const auto& local : locals) {
                result.energy += local.energy;
                result.weighted_latency += local.weighted_latency;
                result.weight += local.weight;
                if (need_congestion && !local.congestion.empty()) {
                    for (uint32_t core = 0; core < cores_count; ++core)
                        result.congestion[core] += local.congestion[core];
                }
            }
            return result;
        }
    }

    // total spike-traffic energy under exact minimum multicast trees
    double HardwareModel::multicastEnergyConsumption(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        return evaluateMulticast(
            part_snn, placement, coresCount(), coresAlongY(),
            energy_per_routing_, energy_per_wire_, latency_per_routing_, latency_per_wire_,
            true, false, false
        ).energy;
    }

    // frequency-weighted delivery completion latency, with one multicast per source
    double HardwareModel::multicastAverageLatency(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        MulticastEvaluation result = evaluateMulticast(
            part_snn, placement, coresCount(), coresAlongY(),
            energy_per_routing_, energy_per_wire_, latency_per_routing_, latency_per_wire_,
            false, true, false
        );
        return result.weight > 0.0 ? result.weighted_latency / result.weight : 0.0;
    }

    // expected per-core traffic under the uniform distribution over exact minimum multicast core sets
    std::vector<double> HardwareModel::multicastCongestion(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        return evaluateMulticast(
            part_snn, placement, coresCount(), coresAlongY(),
            energy_per_routing_, energy_per_wire_, latency_per_routing_, latency_per_wire_,
            false, false, true
        ).congestion;
    }

    // peak expected per-core traffic under exact minimum multicast trees
    double HardwareModel::multicastMaximumCongestion(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        std::vector<double> congestion = multicastCongestion(part_snn, placement);
        return congestion.empty() ? 0.0 : *std::max_element(congestion.begin(), congestion.end());
    }

    // compute the three analytical multicast placement metrics in one pass over the hypergraph
    MulticastPlacementMetrics HardwareModel::getAllMulticastMetrics(const HyperGraph& part_snn, const std::vector<hwgeom::Coord2D>& placement) const {
        MulticastPlacementMetrics metrics;
        metrics.valid = checkPlacementValidity(part_snn, placement);
        if (!metrics.valid)
            return metrics;

        MulticastEvaluation result = evaluateMulticast(
            part_snn, placement, coresCount(), coresAlongY(),
            energy_per_routing_, energy_per_wire_, latency_per_routing_, latency_per_wire_,
            true, true, true
        );
        metrics.energy = result.energy;
        metrics.avg_latency = result.weight > 0.0 ? result.weighted_latency / result.weight : 0.0;
        metrics.max_congestion = result.congestion.empty() ? 0.0 : *std::max_element(result.congestion.begin(), result.congestion.end());
        return metrics;
    }
}
