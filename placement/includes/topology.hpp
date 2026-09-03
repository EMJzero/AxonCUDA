#pragma once
#include <cmath>
#include <vector>
#include <string>
#include <limits>
#include <fstream>
#include <cstdint>
#include <cstddef>
#include <concepts>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#include "hgraph.hpp"

// TOPOLOGIES
// => coordinates are usable on both host and device
// => minimum-path traversal and file I/O remain host-side

#if defined(__CUDACC__)
#define TOPOLOGY_HOST_DEVICE __host__ __device__
#else
#define TOPOLOGY_HOST_DEVICE
#endif

namespace topology {

    // COORDINATES

    template<uint32_t N>
    class Coord {
        static_assert(N > 0, "A coordinate must have at least one dimension.");

        private:
        int components_[N]{};

        public:
        TOPOLOGY_HOST_DEVICE constexpr Coord() noexcept : components_{} {}

        // basic constructor, use like:
        // - Coord<2> coord(3, 4);
        // - Coord<2>{3, 4};
        template<typename... Components> requires (sizeof...(Components) == N && (std::is_convertible_v<Components, int> && ...))
        TOPOLOGY_HOST_DEVICE constexpr Coord(Components... components) noexcept : components_{static_cast<int>(components)...} {}

        // number of dimensions
        TOPOLOGY_HOST_DEVICE static constexpr uint32_t dimensions() noexcept {
            return N;
        }

        TOPOLOGY_HOST_DEVICE constexpr int& operator[](uint32_t dim) noexcept {
            return components_[dim];
        }

        TOPOLOGY_HOST_DEVICE constexpr const int& operator[](uint32_t dim) const noexcept {
            return components_[dim];
        }

        TOPOLOGY_HOST_DEVICE constexpr int* data() noexcept {
            return components_;
        }

        TOPOLOGY_HOST_DEVICE constexpr const int* data() const noexcept {
            return components_;
        }

        TOPOLOGY_HOST_DEVICE constexpr bool operator==(const Coord& other) const noexcept {
            for (uint32_t dim = 0; dim < N; ++dim) {
                if (components_[dim] != other.components_[dim]) return false;
            }
            return true;
        }

        TOPOLOGY_HOST_DEVICE constexpr Coord operator-(const Coord& other) const noexcept {
            Coord diff;
            for (uint32_t dim = 0; dim < N; ++dim)
                diff.components_[dim] = components_[dim] - other.components_[dim];
            return diff;
        }

        // set all components to the provided value
        TOPOLOGY_HOST_DEVICE void setAll(int val) noexcept {
            for (uint32_t dim = 0; dim < N; ++dim)
                components_[dim] = val;
        }

        // return the component-wise minimum of this and the other coordinate
        TOPOLOGY_HOST_DEVICE constexpr Coord componentWiseMin(const Coord& other) const noexcept {
            Coord min;
            for (uint32_t dim = 0; dim < N; ++dim)
                min.components_[dim] = components_[dim] < other.components_[dim] ? components_[dim] : other.components_[dim];
            return min;
        }

        // return the component-wise maximum of this and the other coordinate
        TOPOLOGY_HOST_DEVICE constexpr Coord componentWiseMax(const Coord& other) const noexcept {
            Coord max;
            for (uint32_t dim = 0; dim < N; ++dim)
                max.components_[dim] = components_[dim] > other.components_[dim] ? components_[dim] : other.components_[dim];
            return max;
        }

        // number of points in the volume between the origin and this coordinate (excluded)
        TOPOLOGY_HOST_DEVICE uint64_t volume() const noexcept {
            uint64_t product = 1u;
            for (uint32_t dim = 0; dim < N; ++dim)
                product *= std::abs(components_[dim]);
            return product;
        }

        TOPOLOGY_HOST_DEVICE uint64_t hash() const noexcept {
            uint64_t hash = N;
            for (uint32_t dim = 0; dim < N; ++dim)
                hash ^= static_cast<uint64_t>(components_[dim]) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
            return hash;
        }

        std::string toString() const {
            //std::string result = "Coord<" + std::to_string(N) + ">(";
            std::string result = "(";
            for (uint32_t dim = 0; dim < N; ++dim) {
                if (dim != 0) result += ", ";
                result += std::to_string(components_[dim]);
            }
            result += ')';
            return result;
        }

        /*
        Binary format:
        [std::size_t count][count * N * int]
        */
        static void toFile(const std::vector<Coord>& coordinates, const std::string& path) {
            static_assert(sizeof(Coord) == N * sizeof(int));

            std::ofstream out(path, std::ios::binary);
            if (!out) throw std::runtime_error("Failed to save coordinates, cannot open file.");

            const std::size_t size = coordinates.size();
            out.write(reinterpret_cast<const char*>(&size), sizeof(size));
            if (!coordinates.empty()) {
                out.write(
                    reinterpret_cast<const char*>(coordinates.data()),
                    static_cast<std::streamsize>(coordinates.size() * sizeof(Coord))
                );
            }
            if (!out) throw std::runtime_error("Failed to save coordinates, error writing file.");
        }

        static std::vector<Coord> fromFile(const std::string& path) {
            static_assert(sizeof(Coord) == N * sizeof(int));

            std::ifstream in(path, std::ios::binary);
            if (!in) throw std::runtime_error("Failed to load coordinates, cannot open file.");

            std::size_t size{};
            in.read(reinterpret_cast<char*>(&size), sizeof(size));
            if (!in) throw std::runtime_error("Failed to load coordinates, invalid file header.");
            if (size > std::numeric_limits<std::size_t>::max() / sizeof(Coord))
                throw std::runtime_error("Failed to load coordinates, coordinate count is too large.");

            std::vector<Coord> coordinates(size);
            if (!coordinates.empty()) {
                in.read(
                    reinterpret_cast<char*>(coordinates.data()),
                    static_cast<std::streamsize>(coordinates.size() * sizeof(Coord))
                );
            }
            if (!in) throw std::runtime_error("Failed to load coordinates, incomplete coordinate data.");
            return coordinates;
        }
    };

    // convenient constructor deduction, use like:
    // - Coord coord(3, 4); -> deduces Coord<2>
    // - Coord{3, 4};       -> deduces Coord<2>
    template<typename... Components>
        requires (sizeof...(Components) > 0 && (std::is_convertible_v<Components, int> && ...))
    Coord(Components...) -> Coord<sizeof...(Components)>;

}

// specialize std::hash (for std containers) for any Coord<N>
namespace std {
    template<std::uint32_t N>
    struct hash<topology::Coord<N>> {
        std::size_t operator()(const topology::Coord<N>& value) const noexcept {
            return value.hash();
        }
    };
}


namespace topology {

    // TOPOLOGIES

    template<typename T>
    concept Topology = requires(
        const T& topology,
        const typename T::Coord& src,
        const typename T::Coord& dst,
        const uint32_t neighbor_idx
    ) {
        { T::dimensions } -> std::convertible_to<uint32_t>;
        requires std::same_as<typename T::Coord, Coord<T::dimensions>>;
        typename T::Path;
        typename T::Paths;
        // maximum number of neighbors around a point
        { T::neighborsCount() } -> std::same_as<uint32_t>;
        // returns the size of the topology along each dimension
        { topology.extent() } -> std::same_as<const typename T::Coord&>;
        // check if a certain point is part of the topology or outside of it
        { topology.contains(src) } -> std::same_as<bool>;
        // returns the idx-th neighbor around a point (idx must be in [0, neighborsCount-1])
        // NOTE: returned neighbors can be outside the topology, check them with "contains"
        { topology.neighbor(src, neighbor_idx) } -> std::same_as<typename T::Coord>;
        // inverse of 'neighbor', returns the index of a neighbor around a point, or UINT32_MAX if the coordinates aren't neighbors
        { topology.neighborIdx(src, dst) } -> std::same_as<uint32_t>;
        // distance over the topology (# of hops) between two points
        { topology.distance(src, dst) } -> std::same_as<uint32_t>;
        // index obtained from flattening a coordinate over the extent of each dimension:
        // dimensions are flattened in the order they appear in 'components', earlier dimensions move the index by an entire volume slice of the subsequent dimensions
        // NOTE: the coordinate must have all positive components
        { topology.flattenedIdx(src) } -> std::same_as<uint32_t>;
        // returns all minimum paths between two points
        { topology.iterMinimumPaths(src, dst) } -> std::same_as<typename T::Paths>;
    };

    // easy-access typename, use as: 'Coord_t<T>' to replace 'typename T::Coord'
    template<Topology T>
    using Coord_t = typename T::Coord;

    template<uint32_t N>
    class Lattice final {
        public:
        static constexpr uint32_t dimensions = N;

        using Coord = typename topology::Coord<N>;
        using Path = typename std::vector<Coord>;
        using Paths = typename std::vector<Path>;

        private:
        Coord extent_;

        public:
        // default unit lattice, used as initial storage on device
        TOPOLOGY_HOST_DEVICE constexpr Lattice() noexcept : extent_{} {
            for (uint32_t dim = 0; dim < N; ++dim)
                extent_[dim] = 1;
        }

        // basic constructor, use like:
        // - Lattice<4> lattice(Coord<4>{5, 6, 7, 6});
        // - Lattice<4>{Coord<4>{5, 6, 7, 6}}
        explicit Lattice(const Coord& extent) : extent_(extent) {
            for (uint32_t dim = 0; dim < N; ++dim) {
                if (extent_[dim] <= 0)
                    throw std::invalid_argument("Every lattice extent must be greater than zero.");
            }
        }

        // convenient constructor, use like:
        // - Lattice<4> lattice(5, 6, 7, 6);
        // - Lattice<4>{5, 6, 7, 6}
        template<typename... Extents> requires (sizeof...(Extents) == N && (std::is_convertible_v<Extents, int> && ...))
        explicit Lattice(Extents... extents) : Lattice(Coord{extents...}) {}

        const Coord& extent() const noexcept {
            return extent_;
        }

        TOPOLOGY_HOST_DEVICE bool contains(const Coord& coordinate) const noexcept {
            for (uint32_t dim = 0; dim < N; ++dim) {
                if (coordinate[dim] < 0 || coordinate[dim] >= extent_[dim])
                    return false;
            }
            return true;
        }

        TOPOLOGY_HOST_DEVICE static constexpr uint32_t neighborsCount() noexcept {
            return 2 * N;
        }

        TOPOLOGY_HOST_DEVICE Coord neighbor(const Coord& src, uint32_t neighbor_idx) const noexcept {
            Coord result = src;
            const uint32_t dim = neighbor_idx >> 1;
            result[dim] += (neighbor_idx & 1u) ? 1 : -1;
            return result;
        }

        TOPOLOGY_HOST_DEVICE uint32_t neighborIdx(const Coord& src, const Coord& neigh) const noexcept {
            uint32_t result = UINT32_MAX;
            for (uint32_t dim = 0; dim < N; ++dim) {
                const int64_t delta = static_cast<int64_t>(neigh[dim]) - src[dim];
                if (delta == 0) continue;
                if (result != UINT32_MAX || (delta != -1 && delta != 1))
                    return UINT32_MAX;
                result = 2 * dim + static_cast<uint32_t>(delta > 0);
            }
            return result;
        }

        TOPOLOGY_HOST_DEVICE uint32_t distance(const Coord& src, const Coord& dst) const noexcept {
            uint32_t result = 0;
            for (uint32_t dim = 0; dim < N; ++dim) {
                const int64_t delta = static_cast<int64_t>(src[dim]) - dst[dim];
                result += static_cast<uint32_t>(delta < 0 ? -delta : delta);
            }
            return result;
        }

        TOPOLOGY_HOST_DEVICE uint32_t flattenedIdx(const Coord& plc) const noexcept {
            uint32_t flat_idx = 0u;
            for (uint32_t dim = 0; dim < N; ++dim)
                flat_idx = flat_idx * static_cast<uint32_t>(extent_[dim]) + static_cast<uint32_t>(plc[dim]);
            return flat_idx;
        }

        Paths iterMinimumPaths(const Coord& src, const Coord& dst) const {
            if (!contains(src) || !contains(dst))
                throw std::invalid_argument("Lattice coordinates are outside its extent.");

            Paths paths;
            Path path{src};
            Coord current = src;

            auto visit = [&](auto&& self) -> void {
                if (current == dst) {
                    paths.push_back(path);
                    return;
                }

                for (uint32_t dim = 0; dim < N; ++dim) {
                    if (current[dim] == dst[dim]) continue;

                    const int step = current[dim] < dst[dim] ? 1 : -1;
                    current[dim] += step;
                    path.push_back(current);
                    self(self);
                    path.pop_back();
                    current[dim] -= step;
                }
            };

            visit(visit);
            return paths;
        }
    };

    template<uint32_t N>
    class Torus final {
        public:
        static constexpr uint32_t dimensions = N;

        using Coord = typename topology::Coord<N>;
        using Path = typename std::vector<Coord>;
        using Paths = typename std::vector<Path>;

        private:
        Coord extent_;

        public:
        // default unit torus, used as initial storage on device
        TOPOLOGY_HOST_DEVICE constexpr Torus() noexcept : extent_{} {
            for (uint32_t dim = 0; dim < N; ++dim)
                extent_[dim] = 1;
        }

        explicit Torus(const Coord& extent) : extent_(extent) {
            for (uint32_t dim = 0; dim < N; ++dim) {
                if (extent_[dim] <= 0)
                    throw std::invalid_argument("Every torus extent must be greater than zero.");
            }
        }

        template<typename... Extents> requires (sizeof...(Extents) == N && (std::is_convertible_v<Extents, int> && ...))
        explicit Torus(Extents... extents) : Torus(Coord{extents...}) {}

        const Coord& extent() const noexcept {
            return extent_;
        }

        TOPOLOGY_HOST_DEVICE bool contains(const Coord& coordinate) const noexcept {
            for (uint32_t dim = 0; dim < N; ++dim) {
                if (coordinate[dim] < 0 || coordinate[dim] >= extent_[dim])
                    return false;
            }
            return true;
        }

        // maximum number of neighbors around a point
        TOPOLOGY_HOST_DEVICE static constexpr uint32_t neighborsCount() noexcept {
            return 2 * N;
        }

        TOPOLOGY_HOST_DEVICE Coord neighbor(const Coord& src, uint32_t neighbor_idx) const noexcept {
            Coord result = src;
            const uint32_t dim = neighbor_idx >> 1;
            if (neighbor_idx & 1u)
                result[dim] = src[dim] == extent_[dim] - 1 ? 0 : src[dim] + 1;
            else
                result[dim] = src[dim] == 0 ? extent_[dim] - 1 : src[dim] - 1;
            return result;
        }

        // NOTE: extents of one or two can map multiple indices to the same neighbor
        TOPOLOGY_HOST_DEVICE uint32_t neighborIdx(const Coord& src, const Coord& neigh) const noexcept {
            uint32_t differing_dim = N;
            for (uint32_t dim = 0; dim < N; ++dim) {
                if (src[dim] == neigh[dim]) continue;
                if (differing_dim != N) return UINT32_MAX;
                differing_dim = dim;
            }

            if (differing_dim != N) {
                const uint32_t dim = differing_dim;
                const int backward = src[dim] == 0 ? extent_[dim] - 1 : src[dim] - 1;
                if (neigh[dim] == backward) return 2 * dim;
                const int forward = src[dim] == extent_[dim] - 1 ? 0 : src[dim] + 1;
                if (neigh[dim] == forward) return 2 * dim + 1;
                return UINT32_MAX;
            }

            for (uint32_t dim = 0; dim < N; ++dim) {
                if (extent_[dim] == 1) return 2 * dim;
            }
            return UINT32_MAX;
        }

        TOPOLOGY_HOST_DEVICE uint32_t distance(const Coord& src, const Coord& dst) const noexcept {
            uint32_t result = 0;
            for (uint32_t dim = 0; dim < N; ++dim) {
                const int64_t difference = static_cast<int64_t>(src[dim]) - dst[dim];
                const uint64_t magnitude = static_cast<uint64_t>(difference < 0 ? -difference : difference);
                const uint32_t extent = static_cast<uint32_t>(extent_[dim]);
                const uint32_t direct = static_cast<uint32_t>(magnitude % extent);
                result += (direct < extent - direct) ? direct : extent - direct; // min
            }
            return result;
        }

        TOPOLOGY_HOST_DEVICE uint32_t flattenedIdx(const Coord& plc) const noexcept {
            uint32_t flat_idx = 0u;
            for (uint32_t dim = 0; dim < N; ++dim)
                flat_idx = flat_idx * static_cast<uint32_t>(extent_[dim]) + static_cast<uint32_t>(plc[dim]);
            return flat_idx;
        }

        Paths iterMinimumPaths(const Coord& src, const Coord& dst) const {
            if (!contains(src) || !contains(dst))
                throw std::invalid_argument("Torus coordinates are outside its extent.");

            Paths paths;
            Path path{src};
            Coord current = src;

            auto visit = [&](auto&& self) -> void {
                if (current == dst) {
                    paths.push_back(path);
                    return;
                }

                const uint32_t remaining_distance = distance(current, dst);
                for (uint32_t dim = 0; dim < N; ++dim) {
                    if (current[dim] == dst[dim]) continue;

                    const int original = current[dim];
                    const int extent = extent_[dim];
                    const int forward = (original + 1) % extent;
                    const int backward = (original + extent - 1) % extent;

                    current[dim] = forward;
                    if (distance(current, dst) + 1 == remaining_distance) {
                        path.push_back(current);
                        self(self);
                        path.pop_back();
                    }

                    if (backward != forward) {
                        current[dim] = backward;
                        if (distance(current, dst) + 1 == remaining_distance) {
                            path.push_back(current);
                            self(self);
                            path.pop_back();
                        }
                    }

                    current[dim] = original;
                }
            };

            visit(visit);
            return paths;
        }
    };

    // ARBITRARY: an undirected weighted graph, used as-is as the target topology
    // => weights are the "length" of each edge, distances are precomputed shortest weighted paths
    // => since neighborsCount() is used as a fixed-size register/stack array bound in device kernels, the
    //    maximum degree of any node must be known at compile time (MaxDegree), same as Torus/Lattice's N
    // => the whole class is copied byte-for-byte into CUDA __constant__ memory, whose hardware budget is a
    //    fixed ~64KB shared with every other topology instantiated in the program; this caps how large a
    //    graph can be supported (MaxNodes) - see the size static_assert below. A future "ArbitrarySparse"
    //    variant (hash-table based) is the intended escape hatch for graphs too large to fit here.
    template<uint32_t MaxNodes, uint32_t MaxDegree>
    class Arbitrary final {
        static_assert(MaxNodes > 0, "An arbitrary topology must support at least one node.");
        static_assert(MaxDegree > 0, "An arbitrary topology must support at least one neighbor per node.");
        static_assert(
            sizeof(uint32_t) * (2u * MaxNodes * MaxDegree + MaxNodes * MaxNodes) + sizeof(int) <= 65536,
            "Arbitrary<MaxNodes, MaxDegree> would exceed CUDA's ~64KB __constant__ memory budget; "
            "reduce MaxNodes/MaxDegree, or wait for the future ArbitrarySparse variant to support larger graphs."
        );

        public:
        static constexpr uint32_t dimensions = 1;

        using Coord = typename topology::Coord<1>;
        using Path = typename std::vector<Coord>;
        using Paths = typename std::vector<Path>;

        // sentinel marking an unused neighbor slot
        static constexpr uint32_t INVALID_NODE = UINT32_MAX;
        // sentinel marking two nodes with no path between them (a disconnected graph is rejected at load time)
        static constexpr uint32_t UNREACHABLE = UINT32_MAX;
        // fixed-point scale applied to edge weights ("lengths") so accumulated distances stay integral
        static constexpr float LENGTH_SCALE = 1024.0f;

        private:
        Coord extent_; // extent_[0] = number of nodes actually in use (<= MaxNodes)
        uint32_t neighbors_[MaxNodes * MaxDegree]; // neighbors_[node*MaxDegree + k] -> k-th adjacent node, or INVALID_NODE
        uint32_t edge_lengths_[MaxNodes * MaxDegree]; // parallel to neighbors_: fixed-point length of that edge
        uint32_t distances_[MaxNodes * MaxNodes]; // distances_[src*MaxNodes + dst] -> precomputed fixed-point shortest-path length

        public:
        // default: single isolated node, used as initial storage on device before the real graph is uploaded
        TOPOLOGY_HOST_DEVICE constexpr Arbitrary() noexcept : extent_{1}, neighbors_{}, edge_lengths_{}, distances_{} {
            for (uint32_t i = 0; i < MaxNodes * MaxDegree; ++i) neighbors_[i] = INVALID_NODE;
            for (uint32_t i = 0; i < MaxNodes * MaxDegree; ++i) edge_lengths_[i] = 0;
            for (uint32_t i = 0; i < MaxNodes * MaxNodes; ++i) distances_[i] = 0;
        }

        // build the topology from an arbitrary undirected weighted graph, reusing hgraph's loading infrastructure:
        // every hyperedge is expanded into one undirected edge per (source, destination) pair, weighted by the
        // hyperedge's weight (its "length") -> a plain 2-pin hyperedge is just a regular graph edge
        static Arbitrary fromGraph(const hgraph::HyperGraph& graph) {
            const uint32_t node_count = graph.nodes();
            if (node_count == 0)
                throw std::invalid_argument("Arbitrary topology graph has no nodes.");
            if (node_count > MaxNodes)
                throw std::invalid_argument(
                    "Arbitrary topology graph has " + std::to_string(node_count) + " nodes, exceeding the "
                    "maximum supported (" + std::to_string(MaxNodes) + "); increase the MaxNodes template parameter."
                );

            // collect undirected edges, keeping the minimum length for any duplicate pair
            std::vector<std::unordered_map<uint32_t, uint32_t>> adjacency(node_count);
            auto addEdge = [&](uint32_t u, uint32_t v, float weight) {
                if (u == v) return; // ignore self-loops
                if (!(weight > 0.0f))
                    throw std::invalid_argument("Arbitrary topology edge weights (lengths) must be positive.");
                const uint32_t length = std::max<uint32_t>(1u, static_cast<uint32_t>(std::llround(static_cast<double>(weight) * LENGTH_SCALE)));
                auto insertMin = [&](uint32_t a, uint32_t b) {
                    auto it = adjacency[a].find(b);
                    if (it == adjacency[a].end() || length < it->second)
                        adjacency[a][b] = length;
                };
                insertMin(u, v);
                insertMin(v, u);
            };

            for (const auto& he : graph.hedges()) {
                const std::vector<uint32_t> srcs = he.sources();
                const std::vector<uint32_t> dsts = he.destinations();
                for (uint32_t u : srcs) {
                    if (u >= node_count) throw std::invalid_argument("Arbitrary topology graph has an out-of-range node id.");
                    for (uint32_t v : dsts) {
                        if (v >= node_count) throw std::invalid_argument("Arbitrary topology graph has an out-of-range node id.");
                        addEdge(u, v, he.weight());
                    }
                }
            }

            Arbitrary topo;
            topo.extent_[0] = static_cast<int>(node_count);
            for (uint32_t i = 0; i < MaxNodes * MaxDegree; ++i) topo.neighbors_[i] = INVALID_NODE;
            for (uint32_t i = 0; i < MaxNodes * MaxDegree; ++i) topo.edge_lengths_[i] = 0;
            for (uint32_t i = 0; i < MaxNodes * MaxNodes; ++i) topo.distances_[i] = UNREACHABLE;

            for (uint32_t u = 0; u < node_count; ++u) {
                if (adjacency[u].size() > MaxDegree)
                    throw std::invalid_argument(
                        "Node " + std::to_string(u) + " has degree " + std::to_string(adjacency[u].size()) +
                        ", exceeding the maximum supported degree (" + std::to_string(MaxDegree) + "); increase the MaxDegree template parameter."
                    );

                std::vector<uint32_t> neighs;
                neighs.reserve(adjacency[u].size());
                for (const auto& [v, len] : adjacency[u]) neighs.push_back(v);
                std::sort(neighs.begin(), neighs.end()); // deterministic ordering

                for (uint32_t k = 0; k < neighs.size(); ++k) {
                    topo.neighbors_[u * MaxDegree + k] = neighs[k];
                    topo.edge_lengths_[u * MaxDegree + k] = adjacency[u].at(neighs[k]);
                }
            }

            // Floyd-Warshall all-pairs shortest paths (fixed-point integer arithmetic)
            for (uint32_t u = 0; u < node_count; ++u) {
                topo.distances_[u * MaxNodes + u] = 0;
                for (const auto& [v, len] : adjacency[u]) {
                    const uint32_t idx = u * MaxNodes + v;
                    if (len < topo.distances_[idx]) topo.distances_[idx] = len;
                }
            }
            for (uint32_t mid = 0; mid < node_count; ++mid) {
                for (uint32_t u = 0; u < node_count; ++u) {
                    const uint32_t d_u_mid = topo.distances_[u * MaxNodes + mid];
                    if (d_u_mid == UNREACHABLE) continue;
                    for (uint32_t v = 0; v < node_count; ++v) {
                        const uint32_t d_mid_v = topo.distances_[mid * MaxNodes + v];
                        if (d_mid_v == UNREACHABLE) continue;
                        const uint64_t via = static_cast<uint64_t>(d_u_mid) + d_mid_v;
                        if (via < topo.distances_[u * MaxNodes + v])
                            topo.distances_[u * MaxNodes + v] = static_cast<uint32_t>(via);
                    }
                }
            }

            for (uint32_t u = 0; u < node_count; ++u) {
                for (uint32_t v = 0; v < node_count; ++v) {
                    if (topo.distances_[u * MaxNodes + v] == UNREACHABLE)
                        throw std::invalid_argument(
                            "Arbitrary topology graph is disconnected (no path from node " + std::to_string(u) +
                            " to node " + std::to_string(v) + ")."
                        );
                }
            }

            return topo;
        }

        const Coord& extent() const noexcept {
            return extent_;
        }

        TOPOLOGY_HOST_DEVICE bool contains(const Coord& coordinate) const noexcept {
            return coordinate[0] >= 0 && coordinate[0] < extent_[0];
        }

        TOPOLOGY_HOST_DEVICE static constexpr uint32_t neighborsCount() noexcept {
            return MaxDegree;
        }

        TOPOLOGY_HOST_DEVICE Coord neighbor(const Coord& src, uint32_t neighbor_idx) const noexcept {
            const uint32_t node = static_cast<uint32_t>(src[0]);
            const uint32_t neigh = neighbors_[node * MaxDegree + neighbor_idx];
            return neigh == INVALID_NODE ? Coord{-1} : Coord{static_cast<int>(neigh)};
        }

        TOPOLOGY_HOST_DEVICE uint32_t neighborIdx(const Coord& src, const Coord& neigh) const noexcept {
            const uint32_t node = static_cast<uint32_t>(src[0]);
            const uint32_t target = static_cast<uint32_t>(neigh[0]);
            for (uint32_t k = 0; k < MaxDegree; ++k) {
                if (neighbors_[node * MaxDegree + k] == target)
                    return k;
            }
            return UINT32_MAX;
        }

        TOPOLOGY_HOST_DEVICE uint32_t distance(const Coord& src, const Coord& dst) const noexcept {
            return distances_[static_cast<uint32_t>(src[0]) * MaxNodes + static_cast<uint32_t>(dst[0])];
        }

        TOPOLOGY_HOST_DEVICE uint32_t flattenedIdx(const Coord& plc) const noexcept {
            return static_cast<uint32_t>(plc[0]);
        }

        Paths iterMinimumPaths(const Coord& src, const Coord& dst) const {
            if (!contains(src) || !contains(dst))
                throw std::invalid_argument("Arbitrary coordinates are outside its extent.");

            Paths paths;
            Path path{src};
            Coord current = src;

            auto visit = [&](auto&& self) -> void {
                if (current == dst) {
                    paths.push_back(path);
                    return;
                }

                const uint32_t remaining = distance(current, dst);
                const uint32_t node = static_cast<uint32_t>(current[0]);
                for (uint32_t k = 0; k < MaxDegree; ++k) {
                    const uint32_t neigh = neighbors_[node * MaxDegree + k];
                    if (neigh == INVALID_NODE) continue;

                    const uint32_t edge_len = edge_lengths_[node * MaxDegree + k];
                    const Coord next{static_cast<int>(neigh)};
                    if (distance(next, dst) + edge_len != remaining) continue;

                    const Coord saved = current;
                    current = next;
                    path.push_back(next);
                    self(self);
                    path.pop_back();
                    current = saved;
                }
            };

            visit(visit);
            return paths;
        }
    };

    // predefined topologies ready for use
    using Lattice2D = Lattice<2>;
    using Lattice3D = Lattice<3>;
    using Torus2D = Torus<2>;
    using Torus3D = Torus<3>;
    using Torus6D = Torus<6>;
    using ArbitraryGraph = Arbitrary<64, 32>; // capped by the __constant__ memory budget - see static_assert above

    // verify that defined topologies satisfy the concept
    static_assert(Topology<Lattice2D>);
    static_assert(Topology<Lattice3D>);
    static_assert(Topology<Torus2D>);
    static_assert(Topology<Torus3D>);
    static_assert(Topology<Torus6D>);
    static_assert(Topology<ArbitraryGraph>);
    static_assert(std::is_trivially_copyable_v<ArbitraryGraph>);

    // verify that coordinates are allocated as fixed size buffers
    static_assert(sizeof(Coord<2>) == 2 * sizeof(int));
    static_assert(sizeof(Coord<3>) == 3 * sizeof(int));
    static_assert(sizeof(Coord<6>) == 6 * sizeof(int));
    static_assert(std::is_trivially_copyable_v<Coord<2>>);
    static_assert(std::is_trivially_copyable_v<Coord<3>>);
    static_assert(std::is_trivially_copyable_v<Coord<6>>);
}

#undef TOPOLOGY_HOST_DEVICE
