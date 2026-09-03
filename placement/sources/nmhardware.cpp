#include <tuple>
#include <cmath>
#include <chrono>
#include <atomic>
#include <limits>
#include <iomanip>
#include <iostream>
#include <exception>
#include <algorithm>
#include <semaphore>
#include <unordered_map>
#include <unordered_set>

#include "nmhardware.hpp"

#include <omp.h>

// maximum number of nodes per hyperedge for Steiner tree evaluation
#define MULTICAST_STEINER_NODE_LIMIT 256u

// maximum wall-time allowed to complete a multicast metrics evaluation
#define MULTICAST_STEINER_TIME_LIMIT_HOURS 2

// maximum number of memory-intensive frontier solvers allowed to run concurrently
#define MULTICAST_STEINER_MAX_PARALLEL_SOLVERS 8

static_assert(MULTICAST_STEINER_MAX_PARALLEL_SOLVERS > 0);

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

    template<Topology T>
    bool HardwareModel<T>::checkHgraphFit(const HyperGraph& hgraph, bool already_partitioned, bool verbose) const {
        if (already_partitioned) {
            if (hgraph.nodes() > coresCount()) {
                if (verbose) std::cout << "HYPERGRAPH CAN'T FIT ON THE HW: more partitions than the HW cores\n";
                return false;
            }
            for (std::uint32_t n = 0; n < hgraph.nodes(); ++n) {
                if (hgraph.inboundIds(n).size() > synapses_per_core_) {
                    if (verbose) std::cout << "HYPERGRAPH CAN'T FIT ON THE HW: more inbound hyperedges on a partition than the HW can handle\n";
                    return false;
                }
            }
            return true;
        }

        if (hgraph.nodes() > coresCount() * nodes_per_core_) {
            if (verbose) std::cout << "HYPERGRAPH CAN'T FIT ON THE HW: more nodes than the HW can collectively handle\n";
            return false;
        }

        for (std::uint32_t n = 0; n < hgraph.nodes(); ++n) {
            if (hgraph.inboundIds(n).size() > synapses_per_core_) {
                if (verbose) std::cout << "HYPERGRAPH CAN'T FIT ON THE HW: more inbound hyperedges on a single node than the HW can handle\n";
                return false;
            }
        }

        try {
            (void)partitionSequential(hgraph, nodes_per_core_, synapses_per_core_, coresCount());
        } catch (...) {
            if (verbose) std::cout << "HYPERGRAPH WON'T LIKELY FIT ON THE HW: the greedy split of nodes (and their pins) among cores failed\n";
            return false;
        }
        return true;
    }

    template<Topology T>
    bool HardwareModel<T>::checkPartitionValidity(const HyperGraph& hgraph, const std::vector<uint32_t>& partitions, bool verbose) const {
        if (partitions.size() != hgraph.nodes())
            throw std::runtime_error("Each node must be assigned to a partition (" + std::to_string(hgraph.nodes()) + " node != " + std::to_string(partitions.size()) + " part).");

        // count neurons per partition and distinct partitions
        std::unordered_map<uint32_t, uint32_t> partitions_counter;
        partitions_counter.reserve(partitions.size());
        for (uint32_t p : partitions) {
            ++partitions_counter[p];
        }

        const uint32_t partitions_count = partitions_counter.size();
        if (partitions_count > coresCount()) {
            if (verbose) std::cout << "INVALID PARTITIONING: more partitions (" << partitions_count << ") than cores (" << coresCount() << ")\n";
            return false;
        }

        // neurons per partition constraint
        for (const auto& kv : partitions_counter) {
            if (kv.second > nodes_per_core_) {
                if (verbose) std::cout << "INVALID PARTITIONING: more node per partition (" << kv.second << ") than a core can store (" << nodes_per_core_ << ")\n";
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
        for (const auto& he : hgraph.hedges()) {
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
                if (verbose) std::cout << "INVALID PARTITIONING: more inbound hyperedges per partition (" << synapses_per_partition[i] << ") than a core can handle (" << synapses_per_core_ << ")\n";
                return false;
            }
        }
        return true;
    }

    template<Topology T>
    bool HardwareModel<T>::checkPlacementValidity(const HyperGraph& part_hgraph, const std::vector<Coord>& placement, bool verbose) const {
        if (placement.size() != part_hgraph.nodes())
            throw std::runtime_error("Each partition must be assigned to a core (" + std::to_string(part_hgraph.nodes()) + " part != " + std::to_string(placement.size()) + " plac).");

        std::unordered_set<Coord> seen_cores;
        seen_cores.reserve(placement.size());

        for (const auto& core : placement) {
            if (!topology_.contains(core)) {
                if (verbose) std::cout << "INVALID PLACEMENT: a core's coordinates are outside the hardware topology\n";
                return false;
            }
            auto [it, inserted] = seen_cores.insert(core);
            if (!inserted) {
                if (verbose) std::cout << "INVALID PLACEMENT: a core is used more than once\n";
                return false;
            }
        }
        return true;
    }

    template<Topology T>
    SynapticReuseMetrics HardwareModel<T>::synapticReuse(const HyperGraph& hgraph, const std::vector<uint32_t>& partitions) const {
        if (partitions.empty())
            return {};

        uint32_t max_partition = *std::max_element(partitions.begin(), partitions.end());
        uint32_t partitions_count = max_partition + 1;

        std::vector<uint64_t> synapses_count(partitions_count, 0);
        std::vector<uint64_t> axons_count(partitions_count, 0);

        for (const auto& he : hgraph.hedges()) {
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

    template<Topology T>
    ConnectionsLocalityMetrics HardwareModel<T>::connectionsLocality(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        double ar_mean = 0.0;
        double geo_mean = 0.0;
        double ar_mean_weighted = 0.0;
        double geo_mean_weighted = 0.0;
        double weights_sum = 0.0;

        const auto& hedges = part_hgraph.hedges();
        const auto& hedges_flat = part_hgraph.hedgesFlat();
        if (hedges.empty())
            return {};

        for (const auto& hyperedge : hedges) {
            Coord min, max;
            min.setAll(std::numeric_limits<int>::max());
            max.setAll(0);
            for (uint32_t pin = hyperedge.offset(); pin < hyperedge.offset() + hyperedge.length(); ++pin) {
                Coord plc = placement[hedges_flat[pin]];
                min = min.componentWiseMin(plc);
                max = max.componentWiseMax(plc);
            }

            double volume = 1.0;
            for (uint32_t dim = 0; dim < Coord::dimensions(); ++dim)
                volume *= static_cast<double>(max[dim] - min[dim] + 1);
            const double weight = static_cast<double>(hyperedge.weight());
            ar_mean += volume;
            geo_mean += std::log(volume);
            ar_mean_weighted += volume * weight;
            geo_mean_weighted += std::log(volume) * weight;
            weights_sum += weight;
        }

        const double hedge_count = static_cast<double>(hedges.size());
        ar_mean /= hedge_count;
        geo_mean = std::exp(geo_mean / hedge_count);
        if (weights_sum > 0.0) {
            ar_mean_weighted /= weights_sum;
            geo_mean_weighted = std::exp(geo_mean_weighted / weights_sum);
        }

        return {ar_mean, geo_mean, ar_mean_weighted, geo_mean_weighted};
    }

    template<Topology T>
    std::unordered_map<typename HardwareModel<T>::Coord, double> HardwareModel<T>::expectedSpikeTransitProbability(const Coord& src, const Coord& dst) const {
        // NOTE: updated formulation, now differing from Jin et al.
        // => every minimum-distance path between source and destination is considered equally likely
        // =>=> the probability of a core being traversed equates the number of minimum paths including it vs the total number of minimum paths

        const auto paths = topology_.iterMinimumPaths(src, dst);
        if (paths.empty())
            throw std::runtime_error("Topology returned no minimum path between two placement coordinates.");

        std::unordered_map<Coord, double> probabilities;
        for (const auto& path : paths) {
            for (const Coord& coordinate : path)
                ++probabilities[coordinate];
        }

        const double path_count = static_cast<double>(paths.size());
        for (auto& entry : probabilities)
            entry.second /= path_count;

        return probabilities;
    }

    template<Topology T>
    double HardwareModel<T>::placementEnergyConsumption(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        double result = 0.0;

        for (const auto& he : part_hgraph.hedges()) {
            for (const auto src : he.sources()) {
                const auto& src_core = placement[src];
                for (auto dst : he.destinations()) {
                    const auto& dst_core = placement[dst];
                    double md = static_cast<double>(topology_.distance(src_core, dst_core));
                    result += static_cast<double>(he.weight()) * ((md + 1.0) * energy_per_routing_ + md * energy_per_wire_);
                }
            }
        }

        return result;
    }

    template<Topology T>
    double HardwareModel<T>::placementAverageLatency(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        double result = 0.0;
        double tot_spike_frequency = 0.0;

        for (const auto& he : part_hgraph.hedges()) {
            for (const auto src : he.sources()) {
                const auto& src_core = placement[src];

                double w = static_cast<double>(he.weight());
                const uint32_t destinations_count = he.length() - he.src_count();
                tot_spike_frequency += w * static_cast<double>(destinations_count);

                for (auto dst : he.destinations()) {
                    const auto& dst_core = placement[dst];
                    double md = static_cast<double>(topology_.distance(src_core, dst_core));
                    result += w * ((md + 1.0) * latency_per_routing_ + md * latency_per_wire_);
                }
            }
        }

        if (tot_spike_frequency > 0.0)
            return result / tot_spike_frequency;
        return 0.0;
    }

    template<Topology T>
    double HardwareModel<T>::placementMaximumLatency(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        double result = 0.0;

        for (const auto& he : part_hgraph.hedges()) {
            for (const auto src : he.sources()) {
                const auto& src_core = placement[src];

                for (auto dst : he.destinations()) {
                    const auto& dst_core = placement[dst];
                    double md = static_cast<double>(topology_.distance(src_core, dst_core));
                    double latency = (md + 1.0) * latency_per_routing_ + md * latency_per_wire_;
                    if (latency > result)
                        result = latency;
                }
            }
        }

        return result;
    }

    template<Topology T>
    double HardwareModel<T>::placementAverageCongestion(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        double result = 0.0;

        for (const auto& he : part_hgraph.hedges()) {
            for (const auto src : he.sources()) {
                const auto& src_core = placement[src];

                for (auto dst : he.destinations()) {
                    const auto& dst_core = placement[dst];
                    const double path_length = static_cast<double>(topology_.distance(src_core, dst_core)) + 1.0;
                    result += static_cast<double>(he.weight()) * path_length;
                }
            }
        }

        if (coresCount() > 0)
            return result / static_cast<double>(coresCount());
        return 0.0;
    }

    template<Topology T>
    double HardwareModel<T>::placementMaximumCongestion(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        std::unordered_map<Coord, double> congestion;

        for (const auto& he : part_hgraph.hedges()) {
            for (const auto src : he.sources()) {
                const auto& src_core = placement[src];

                for (auto dst : he.destinations()) {
                    const auto& dst_core = placement[dst];
                    const auto paths = topology_.iterMinimumPaths(src_core, dst_core);
                    if (paths.empty())
                        throw std::runtime_error("Topology returned no minimum path between two placement coordinates.");

                    const double path_weight = static_cast<double>(he.weight()) / static_cast<double>(paths.size());
                    for (const auto& path : paths) {
                        for (const Coord& coordinate : path) {
                            congestion[coordinate] += path_weight;
                        }
                    }
                }
            }
        }

        double max_congestion = 0.0;
        for (const auto& [coordinate, value] : congestion) {
            (void)coordinate;
            max_congestion = std::max(max_congestion, value);
        }
        return max_congestion;
    }

    template<Topology T>
    PlacementMetrics HardwareModel<T>::getAllUnicastMetrics(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        PlacementMetrics metrics;

        bool valid = checkPlacementValidity(part_hgraph, placement);
        metrics.valid = valid;

        if (!valid)
            return metrics;

        metrics.energy = placementEnergyConsumption(part_hgraph, placement);
        metrics.avg_latency = placementAverageLatency(part_hgraph, placement);
        metrics.max_latency = placementMaximumLatency(part_hgraph, placement);
        metrics.avg_congestion = placementAverageCongestion(part_hgraph, placement);
        metrics.max_congestion = placementMaximumCongestion(part_hgraph, placement);
        metrics.connections_locality = connectionsLocality(part_hgraph, placement);
        return metrics;
    }

    namespace {
        constexpr uint32_t MAX_STEINER_STATES = 2000000;
        constexpr long double LOG_ZERO = -std::numeric_limits<long double>::infinity();

        struct SteinerStateLimit {};
        struct SteinerTimeout {};

        std::counting_semaphore<MULTICAST_STEINER_MAX_PARALLEL_SOLVERS> steiner_solver_slots{
            MULTICAST_STEINER_MAX_PARALLEL_SOLVERS
        };

        class SteinerSolverPermit {
            public:
            explicit SteinerSolverPermit(const std::chrono::steady_clock::time_point* deadline) {
                if (deadline != nullptr) {
                    if (!steiner_solver_slots.try_acquire_until(*deadline))
                        throw SteinerTimeout{};
                } else {
                    steiner_solver_slots.acquire();
                }
            }

            ~SteinerSolverPermit() { steiner_solver_slots.release(); }

            SteinerSolverPermit(const SteinerSolverPermit&) = delete;
            SteinerSolverPermit& operator=(const SteinerSolverPermit&) = delete;
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

        struct alignas(64) MulticastEvaluation {
            double energy{0.0};
            double weighted_latency{0.0};
            double weight{0.0};
            std::vector<double> congestion;
            float evaluation_fraction{1.0f};
        };

        struct MulticastTask {
            uint32_t hedge;
            uint32_t source;
        };

        enum class MulticastTaskStatus : uint8_t {
            pending,
            completed,
            state_limited
        };

        // numerically stable addition of two values represented in the logarithmic domain
        long double logAdd(long double a, long double b) {
            if (a == LOG_ZERO) return b;
            if (b == LOG_ZERO) return a;
            if (a < b) std::swap(a, b);
            return a + std::log1p(std::exp(b - a));
        }

        // logarithm of the binomial coefficient, backed by a thread-local cache to avoid repeated gamma evaluations
        long double logBinomial(uint32_t n, uint32_t k) {
            if (k > n) return LOG_ZERO;

            thread_local std::vector<long double> log_factorials{0.0L};
            if (log_factorials.size() <= n) {
                size_t old_size = log_factorials.size();
                log_factorials.resize(static_cast<size_t>(n) + 1u);
                for (size_t i = old_size; i <= n; ++i)
                    log_factorials[i] = log_factorials[i - 1u] + std::log(static_cast<long double>(i));
            }
            return log_factorials[n] - log_factorials[k] - log_factorials[n - k];
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
        uint32_t exactSteinerHops(
            std::vector<Coord<2>>& terminals,
            uint32_t cores_y,
            std::vector<double>* congestion,
            double weight,
            const std::chrono::steady_clock::time_point* deadline
        ) {
            std::sort(terminals.begin(), terminals.end(), [](const auto& a, const auto& b) {
                return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
            });
            terminals.erase(std::unique(terminals.begin(), terminals.end()), terminals.end());

            if (terminals.empty())
                return 0u;

            int min_x = terminals.front()[0];
            int max_x = terminals.front()[0];
            int min_y = terminals.front()[1];
            int max_y = terminals.front()[1];
            for (const auto& terminal : terminals) {
                min_x = std::min(min_x, terminal[0]);
                max_x = std::max(max_x, terminal[0]);
                min_y = std::min(min_y, terminal[1]);
                max_y = std::max(max_y, terminal[1]);
            }

            const uint32_t span_x = static_cast<uint32_t>(max_x - min_x + 1);
            const uint32_t span_y = static_cast<uint32_t>(max_y - min_y + 1);

            // a single terminal and collinear terminal sets have one minimum core set
            if (span_x == 1u || span_y == 1u) {
                if (congestion != nullptr) {
                    for (int x = min_x; x <= max_x; ++x) {
                        for (int y = min_y; y <= max_y; ++y) {
                            uint32_t core = static_cast<uint32_t>(x) * cores_y + static_cast<uint32_t>(y);
                            (*congestion)[core] += weight;
                        }
                    }
                }
                return span_x + span_y - 2u;
            }

            // every minimum two-terminal tree is a monotone path; binomial products give its core marginals
            if (terminals.size() == 2u) {
                const auto& source = terminals[0];
                const auto& destination = terminals[1];
                const uint32_t total_dx = static_cast<uint32_t>(std::abs(destination[0] - source[0]));
                const uint32_t total_dy = static_cast<uint32_t>(std::abs(destination[1] - source[1]));
                const uint32_t distance = total_dx + total_dy;
                const long double log_paths = logBinomial(distance, total_dx);

                if (congestion == nullptr)
                    return distance;

                for (int x = min_x; x <= max_x; ++x) {
                    for (int y = min_y; y <= max_y; ++y) {
                        const uint32_t first_dx = static_cast<uint32_t>(std::abs(x - source[0]));
                        const uint32_t first_dy = static_cast<uint32_t>(std::abs(y - source[1]));
                        const uint32_t second_dx = static_cast<uint32_t>(std::abs(destination[0] - x));
                        const uint32_t second_dy = static_cast<uint32_t>(std::abs(destination[1] - y));
                        long double log_containing = logBinomial(first_dx + first_dy, first_dx)
                                                   + logBinomial(second_dx + second_dy, second_dx);
                        double probability = static_cast<double>(std::exp(log_containing - log_paths));
                        uint32_t core = static_cast<uint32_t>(x) * cores_y + static_cast<uint32_t>(y);
                        (*congestion)[core] += weight * probability;
                    }
                }
                return distance;
            }

            // bound concurrent frontier solvers because each can consume hundreds of MB near its state limit
            SteinerSolverPermit solver_permit(deadline);

            // scan along the longest dimension to keep the connectivity frontier as narrow as possible
            const bool transpose = span_y < span_x;
            const uint32_t frontier_width = transpose ? span_y : span_x;
            const uint32_t rows = transpose ? span_x : span_y;
            const uint32_t area = frontier_width * rows;

            auto coordAt = [&](uint32_t position) {
                uint32_t row = position / frontier_width;
                uint32_t column = position % frontier_width;
                if (transpose)
                    return Coord<2>{min_x + static_cast<int>(row), min_y + static_cast<int>(column)};
                return Coord<2>{min_x + static_cast<int>(column), min_y + static_cast<int>(row)};
            };

            std::vector<uint8_t> required(area, 0u);
            for (uint32_t position = 0; position < area; ++position) {
                Coord<2> coord = coordAt(position);
                required[position] = std::binary_search(terminals.begin(), terminals.end(), coord, [](const auto& a, const auto& b) {
                    return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
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
                        throw SteinerStateLimit{};
                    nodes.push_back(DecisionNode{cost, {}});
                    if (congestion != nullptr)
                        nodes.back().incoming.push_back({predecessor, position, included});
                } else {
                    DecisionNode& node = nodes[it->second];
                    if (cost < node.cost) {
                        node.cost = cost;
                        node.incoming.clear();
                        if (congestion != nullptr)
                            node.incoming.push_back({predecessor, position, included});
                    } else if (cost == node.cost && congestion != nullptr) {
                        node.incoming.push_back({predecessor, position, included});
                    }
                }
            };

            for (uint32_t position = 0; position < area; ++position) {
                if (deadline != nullptr && std::chrono::steady_clock::now() >= *deadline)
                    throw SteinerTimeout{};

                const uint32_t column = position % frontier_width;
                next.clear();

                uint32_t checked_states = 0u;
                for (const auto& [state, predecessor] : current) {
                    if (deadline != nullptr && (++checked_states & 1023u) == 0u && std::chrono::steady_clock::now() >= *deadline)
                        throw SteinerTimeout{};

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

            uint32_t hops = optimum - 1u;
            if (congestion == nullptr)
                return hops;

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

            for (uint32_t position = 0; position < area; ++position) {
                if (containing_log_count[position] == LOG_ZERO) continue;
                Coord<2> coord = coordAt(position);
                uint32_t core = static_cast<uint32_t>(coord[0]) * cores_y + static_cast<uint32_t>(coord[1]);
                double probability = static_cast<double>(std::exp(containing_log_count[position] - total_log_count));
                (*congestion)[core] += weight * probability;
            }
            return hops;
        }

        // evaluate every source-multicast once, optionally accumulating the expensive Steiner-derived metrics
        MulticastEvaluation evaluateMulticast(
            const HyperGraph& part_hgraph,
            const std::vector<Coord<2>>& placement,
            const Lattice2D& lattice,
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
            const auto& hedges = part_hgraph.hedges();
            const auto& hedges_flat = part_hgraph.hedgesFlat();
            const size_t hedges_count = hedges.size();

            size_t tasks_count = 0u;
            for (const auto& he : hedges) {
                if (!std::isfinite(he.weight()))
                    throw std::invalid_argument("Multicast metrics require finite hyperedge weights.");
                tasks_count += he.src_count();
            }

            std::vector<MulticastTask> tasks;
            tasks.reserve(tasks_count);
            for (uint32_t hedge = 0; hedge < hedges_count; ++hedge) {
                const auto& he = hedges[hedge];
                for (uint32_t source_idx = 0; source_idx < he.src_count(); ++source_idx)
                    tasks.push_back({hedge, hedges_flat[he.offset() + source_idx]});
            }

            const bool need_steiner = need_energy || need_congestion;
            const bool limit_steiner_time = need_steiner && part_hgraph.nodes() >= MULTICAST_STEINER_NODE_LIMIT;
            if (limit_steiner_time) {
                std::stable_sort(tasks.begin(), tasks.end(), [&](const auto& a, const auto& b) {
                    return hedges[a.hedge].weight() > hedges[b.hedge].weight();
                });
            }

            const auto deadline = std::chrono::steady_clock::now() + std::chrono::hours(MULTICAST_STEINER_TIME_LIMIT_HOURS);
            const int thread_count = std::max<int>(1, std::min<size_t>(static_cast<size_t>(omp_get_max_threads()), tasks.size()));
            std::vector<MulticastEvaluation> locals(static_cast<size_t>(thread_count));
            std::vector<MulticastTaskStatus> task_status(
                need_steiner ? tasks.size() : 0u,
                MulticastTaskStatus::pending
            );
            std::atomic<bool> timed_out{false};
            std::atomic<bool> stop_requested{false};
            std::exception_ptr failure;

            #pragma omp parallel num_threads(thread_count)
            {
                const int tid = omp_get_thread_num();
                auto& local = locals[static_cast<size_t>(tid)];
                if (need_congestion)
                    local.congestion.assign(cores_count, 0.0);
                std::vector<Coord<2>> terminals;

                #pragma omp for schedule(dynamic, 1)
                for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
                    if (stop_requested.load(std::memory_order_relaxed))
                        continue;
                    if (limit_steiner_time && std::chrono::steady_clock::now() >= deadline) {
                        timed_out.store(true, std::memory_order_relaxed);
                        stop_requested.store(true, std::memory_order_relaxed);
                        continue;
                    }

                    try {
                        const auto& task = tasks[task_idx];
                        const auto& he = hedges[task.hedge];
                        const double weight = static_cast<double>(he.weight());
                        const uint32_t destinations_begin = he.offset() + he.src_count();
                        const uint32_t destinations_end = he.offset() + he.length();

                        double weighted_latency = 0.0;

                        if (need_latency) {
                            int max_distance = 0;
                            for (uint32_t pin = destinations_begin; pin < destinations_end; ++pin) {
                                uint32_t destination = hedges_flat[pin];
                                max_distance = std::max<int>(max_distance, lattice.distance(placement[task.source], placement[destination]));
                            }
                            double latency = static_cast<double>(max_distance) * (latency_per_routing + latency_per_wire) + latency_per_routing;
                            weighted_latency = weight * latency;
                        }

                        double energy = 0.0;
                        if (need_steiner) {
                            terminals.clear();
                            terminals.reserve(static_cast<size_t>(destinations_end - destinations_begin) + 1u);
                            terminals.push_back(placement[task.source]);
                            for (uint32_t pin = destinations_begin; pin < destinations_end; ++pin)
                                terminals.push_back(placement[hedges_flat[pin]]);

                            uint32_t hops = exactSteinerHops(
                                terminals, cores_y,
                                need_congestion ? &local.congestion : nullptr,
                                weight,
                                limit_steiner_time ? &deadline : nullptr
                            );
                            if (need_energy)
                                energy = weight * (static_cast<double>(hops) * (energy_per_routing + energy_per_wire) + energy_per_routing);
                        }

                        local.energy += energy;
                        local.weighted_latency += weighted_latency;
                        if (need_latency)
                            local.weight += weight;
                        if (need_steiner)
                            task_status[task_idx] = MulticastTaskStatus::completed;
                    } catch (const SteinerStateLimit&) {
                        task_status[task_idx] = MulticastTaskStatus::state_limited;
                    } catch (const SteinerTimeout&) {
                        timed_out.store(true, std::memory_order_relaxed);
                        stop_requested.store(true, std::memory_order_relaxed);
                    } catch (...) {
                        #pragma omp critical(multicast_metrics_failure)
                        {
                            if (!failure)
                                failure = std::current_exception();
                        }
                        stop_requested.store(true, std::memory_order_relaxed);
                    }
                }
            }

            if (failure)
                std::rethrow_exception(failure);

            auto summarize_incomplete = [&](const auto& is_incomplete) {
                std::vector<uint8_t> incomplete_hedges(hedges_count, 0u);
                for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
                    if (is_incomplete(task_idx))
                        incomplete_hedges[tasks[task_idx].hedge] = 1u;
                }

                size_t incomplete_count = 0u;
                double incomplete_weight = 0.0;
                double total_weight = 0.0;
                for (size_t hedge = 0; hedge < hedges_count; ++hedge) {
                    if (hedges[hedge].src_count() == 0u)
                        continue;
                    total_weight += static_cast<double>(hedges[hedge].weight());
                    if (incomplete_hedges[hedge]) {
                        ++incomplete_count;
                        incomplete_weight += static_cast<double>(hedges[hedge].weight());
                    }
                }
                return std::tuple{incomplete_count, incomplete_weight, total_weight};
            };

            if (std::find(task_status.begin(), task_status.end(), MulticastTaskStatus::state_limited) != task_status.end()) {
                auto [incomplete_count, incomplete_weight, total_weight] = summarize_incomplete([&](size_t task_idx) {
                    return task_status[task_idx] == MulticastTaskStatus::state_limited;
                });
                double incomplete_percentage = total_weight != 0.0 ? 100.0 * incomplete_weight / total_weight : 0.0;
                std::cerr << "WARNING: " << incomplete_count << " hyperedges exceeded the "
                          << MAX_STEINER_STATES << "-state Steiner limit with cumulative weight "
                          << std::fixed << std::setprecision(3) << incomplete_weight << " / " << total_weight << " ("
                          << incomplete_percentage << "% of the total). Excluding them from the reported metrics.\n";
            }

            if (timed_out.load(std::memory_order_relaxed)) {
                auto [incomplete_count, incomplete_weight, total_weight] = summarize_incomplete([&](size_t task_idx) {
                    return task_status[task_idx] != MulticastTaskStatus::completed;
                });
                double incomplete_percentage = total_weight != 0.0 ? 100.0 * incomplete_weight / total_weight : 0.0;
                std::cerr << "WARNING: multicast Steiner evaluation reached its "
                          << MULTICAST_STEINER_TIME_LIMIT_HOURS << "h time limit; "
                          << incomplete_count << " hyperedges left with cumulative weight "
                          << std::fixed << std::setprecision(3) << incomplete_weight << " / " << total_weight << " ("
                          << incomplete_percentage << "% of the total). Reporting metrics from completed hyperedges.\n";
            }

            float evaluation_fraction = 1.0f;
            if (need_steiner) {
                auto [incomplete_count, incomplete_weight, total_weight] = summarize_incomplete([&](size_t task_idx) {
                    return task_status[task_idx] != MulticastTaskStatus::completed;
                });
                (void)incomplete_count;
                if (total_weight != 0.0)
                    evaluation_fraction = static_cast<float>((total_weight - incomplete_weight) / total_weight);
            }

            MulticastEvaluation result;
            result.evaluation_fraction = evaluation_fraction;
            if (need_congestion)
                result.congestion.assign(cores_count, 0.0);
            for (const auto& local : locals) {
                result.energy += local.energy;
                result.weighted_latency += local.weighted_latency;
                result.weight += local.weight;
            }
            if (need_congestion) {
                #pragma omp parallel for if(cores_count >= 65536u)
                for (uint32_t core = 0; core < cores_count; ++core) {
                    double congestion = 0.0;
                    for (const auto& local : locals) {
                        if (!local.congestion.empty())
                            congestion += local.congestion[core];
                    }
                    result.congestion[core] = congestion;
                }
            }
            return result;
        }
    }

    // total spike-traffic energy under exact minimum multicast trees
    template<Topology T>
    double HardwareModel<T>::multicastEnergyConsumption(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        if constexpr (!std::same_as<T, Lattice2D>) {
            // TODO: the exact Steiner solver is specialized for rectangular 2D lattices.
            throw std::logic_error("Multicast energy is only implemented for Lattice2D.");
        } else {
            return evaluateMulticast(
                part_hgraph, placement, topology_, coresCount(), coresAlongDim(1) /*y*/,
                energy_per_routing_, energy_per_wire_, latency_per_routing_, latency_per_wire_,
                true, false, false
            ).energy;
        }
    }

    // frequency-weighted delivery completion latency, with one multicast per source
    template<Topology T>
    double HardwareModel<T>::multicastAverageLatency(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        double weighted_latency = 0.0;
        double total_weight = 0.0;

        for (const auto& hyperedge : part_hgraph.hedges()) {
            const double weight = static_cast<double>(hyperedge.weight());
            for (uint32_t source : hyperedge.sources()) {
                uint32_t max_distance = 0;
                for (uint32_t destination : hyperedge.destinations()) {
                    max_distance = std::max(
                        max_distance,
                        topology_.distance(placement[source], placement[destination])
                    );
                }

                const double latency = static_cast<double>(max_distance) *
                    (latency_per_routing_ + latency_per_wire_) + latency_per_routing_;
                weighted_latency += weight * latency;
                total_weight += weight;
            }
        }

        return total_weight > 0.0 ? weighted_latency / total_weight : 0.0;
    }

    // expected per-core traffic under the uniform distribution over exact minimum multicast core sets
    template<Topology T>
    std::vector<double> HardwareModel<T>::multicastCongestion(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        if constexpr (!std::same_as<T, Lattice2D>) {
            // TODO: congestion requires topology-wide coordinate indexing and a generic multicast solver.
            throw std::logic_error("Multicast congestion is only implemented for Lattice2D.");
        } else {
            return evaluateMulticast(
                part_hgraph, placement, topology_, coresCount(), coresAlongDim(1) /*y*/,
                energy_per_routing_, energy_per_wire_, latency_per_routing_, latency_per_wire_,
                false, false, true
            ).congestion;
        }
    }

    // peak expected per-core traffic under exact minimum multicast trees
    template<Topology T>
    double HardwareModel<T>::multicastMaximumCongestion(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        std::vector<double> congestion = multicastCongestion(part_hgraph, placement);
        return congestion.empty() ? 0.0 : *std::max_element(congestion.begin(), congestion.end());
    }

    // compute the three analytical multicast placement metrics in one pass over the hypergraph
    template<Topology T>
    MulticastPlacementMetrics HardwareModel<T>::getAllMulticastMetrics(const HyperGraph& part_hgraph, const std::vector<Coord>& placement) const {
        MulticastPlacementMetrics metrics;
        metrics.valid = checkPlacementValidity(part_hgraph, placement);
        if (!metrics.valid)
            return metrics;

        if constexpr (std::same_as<T, Lattice2D>) {
            MulticastEvaluation result = evaluateMulticast(
                part_hgraph, placement, topology_, coresCount(), coresAlongDim(1) /*y*/,
                energy_per_routing_, energy_per_wire_, latency_per_routing_, latency_per_wire_,
                true, true, true
            );
            metrics.energy = result.energy;
            metrics.avg_latency = result.weight > 0.0 ? result.weighted_latency / result.weight : 0.0;
            metrics.max_congestion = result.congestion.empty() ? 0.0 : *std::max_element(result.congestion.begin(), result.congestion.end());
            metrics.evaluation_fraction = result.evaluation_fraction;
        } else {
            // TODO: multicast Steiner energy and congestion are still specialized for Lattice2D.
            metrics.energy = std::nullopt;
            metrics.avg_latency = multicastAverageLatency(part_hgraph, placement);
            metrics.max_congestion = std::nullopt;
            metrics.evaluation_fraction = 0.0f;
        }
        return metrics;
    }

    template class HardwareModel<Lattice2D>;
    template class HardwareModel<Lattice3D>;
    template class HardwareModel<Torus2D>;
    template class HardwareModel<Torus3D>;
    template class HardwareModel<Torus6D>;
    template class HardwareModel<ArbitraryGraph>;
}
