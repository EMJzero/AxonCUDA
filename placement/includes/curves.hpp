#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <concepts>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "topology.hpp"
#include "runconfig_plc.hpp"

using namespace topology;
using namespace config_plc;

// TOPOLOGY FAMILIES SUPPORTED BY CURVES

template<typename T>
concept MeshLikeTopology = Topology<T> && (
    std::same_as<std::remove_cv_t<T>, Lattice<T::dimensions>> ||
    std::same_as<std::remove_cv_t<T>, Torus<T::dimensions>>
);


// CURVE GENERATION

inline uint32_t nextPow2(uint32_t value) {
    if (value <= 1)
        return 1;
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

inline uint64_t divCeil(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline uint64_t cappedProduct(uint64_t lhs, uint64_t rhs) {
    if (rhs != 0 && lhs > UINT64_MAX / rhs)
        return UINT64_MAX;
    return lhs * rhs;
}

// visit every relevant fitting box, optimizing one dimension analytically
// => the last dimension only needs to stop at a required size, another side's size, or a power of two
template<MeshLikeTopology T, typename Visitor>
inline void iterFittingRegions(
    uint32_t nodes,
    const T& topology,
    uint32_t side_multiple,
    bool include_powers_of_two,
    Visitor&& visit
) {
    using Coord = Coord_t<T>;
    constexpr uint32_t dimensions = T::dimensions;

    if (side_multiple == 0)
        side_multiple = 1;

    std::array<uint32_t, dimensions> max_sides{};
    uint32_t optimized_dim = 0;
    uint32_t most_choices = 0;

    uint64_t side_bound = divCeil(nodes, side_multiple) * side_multiple;
    if (include_powers_of_two) {
        uint64_t power_bound = 1;
        while (power_bound < nodes)
            power_bound <<= 1;
        side_bound = std::max(side_bound, power_bound);
    }
    for (uint32_t dim = 0; dim < dimensions; ++dim) {
        const uint64_t extent = static_cast<uint64_t>(topology.extent()[dim]);
        const uint64_t bounded_extent = std::min(extent, side_bound);
        max_sides[dim] = static_cast<uint32_t>((bounded_extent / side_multiple) * side_multiple);
        if (max_sides[dim] == 0)
            return;

        const uint32_t choices = max_sides[dim] / side_multiple;
        if (choices > most_choices) {
            most_choices = choices;
            optimized_dim = dim;
        }
    }

    std::array<uint32_t, dimensions> iterated_dims{};
    uint32_t iterated_count = 0;
    for (uint32_t dim = 0; dim < dimensions; ++dim) {
        if (dim != optimized_dim)
            iterated_dims[iterated_count++] = dim;
    }

    Coord region;
    auto iterate = [&](auto&& self, uint32_t iter_idx, uint64_t partial_volume) -> void {
        if (iter_idx == iterated_count) {
            const uint64_t required = divCeil(divCeil(nodes, partial_volume), side_multiple) * side_multiple;
            if (required == 0 || required > max_sides[optimized_dim])
                return;

            std::array<uint32_t, dimensions + 33> candidates{};
            uint32_t candidates_count = 0;
            candidates[candidates_count++] = static_cast<uint32_t>(required);

            // range objectives only change when crossing one of the other sides
            for (uint32_t idx = 0; idx < iterated_count; ++idx) {
                const uint32_t side = static_cast<uint32_t>(region[iterated_dims[idx]]);
                if (side >= required)
                    candidates[candidates_count++] = side;
            }

            if (include_powers_of_two) {
                for (uint64_t side = 1; side <= max_sides[optimized_dim]; side <<= 1) {
                    if (side >= required)
                        candidates[candidates_count++] = static_cast<uint32_t>(side);
                }
            }

            std::sort(candidates.begin(), candidates.begin() + candidates_count);
            const auto last = std::unique(candidates.begin(), candidates.begin() + candidates_count);
            candidates_count = static_cast<uint32_t>(last - candidates.begin());

            for (uint32_t idx = 0; idx < candidates_count; ++idx) {
                const uint32_t side = candidates[idx];
                if (side > max_sides[optimized_dim] || side % side_multiple != 0)
                    continue;
                region[optimized_dim] = static_cast<int>(side);
                visit(region, cappedProduct(partial_volume, side));
            }
            return;
        }

        const uint32_t dim = iterated_dims[iter_idx];
        for (uint32_t side = side_multiple; side <= max_sides[dim]; side += side_multiple) {
            region[dim] = static_cast<int>(side);
            self(self, iter_idx + 1, cappedProduct(partial_volume, side));
        }
    };

    iterate(iterate, 0, 1);
}

template<size_t N>
inline void axesToHilbertTranspose(
    std::array<uint32_t, N>& axes,
    uint32_t dimensions,
    uint32_t bits
) {
    if constexpr (N > 1) {
        if (dimensions <= 1 || bits == 0)
            return;

        // inverse undo, followed by the Gray encoding from Skilling's transform
        const uint32_t most_significant_bit = 1u << (bits - 1);
        for (uint32_t q = most_significant_bit; q > 1; q >>= 1) {
            const uint32_t p = q - 1;
            for (uint32_t dim = 0; dim < dimensions; ++dim) {
                if (axes[dim] & q) {
                    axes[0] ^= p;
                } else {
                    const uint32_t exchange = (axes[0] ^ axes[dim]) & p;
                    axes[0] ^= exchange;
                    axes[dim] ^= exchange;
                }
            }
        }

        for (uint32_t dim = 1; dim < dimensions; ++dim)
            axes[dim] ^= axes[dim - 1];

        uint32_t correction = 0;
        for (uint32_t q = most_significant_bit; q > 1; q >>= 1)
            if (axes[dimensions - 1] & q)
                correction ^= q - 1;
        for (uint32_t dim = 0; dim < dimensions; ++dim)
            axes[dim] ^= correction;
    }
}

// find the most square box (w x h x ...) such that:
//   1) w <= width, h <= height, ..., w * h * ... >= nodes
//   2) w, h, ... are all multiples of 'side_multiple'
// prefer: smallest 'max(w, h, ...) - min(w, h, ...)', then smallest volume
template<MeshLikeTopology T>
inline Coord_t<T> squareishRegion(uint32_t nodes, const T& topology, uint32_t side_multiple = 1) {
    using Coord = Coord_t<T>;

    if (nodes > topology.extent().volume())
        throw std::runtime_error("Topology extent too small to hold all nodes.");

    if (nodes == 0)
        return Coord{};

    Coord best_region;
    uint64_t best_diff = UINT64_MAX;
    uint64_t best_volume = UINT64_MAX;

    iterFittingRegions(
        nodes,
        topology,
        side_multiple,
        false,
        [&](const Coord& region, uint64_t volume) {
            uint32_t min_side = UINT32_MAX;
            uint32_t max_side = 0;
            for (uint32_t dim = 0; dim < T::dimensions; ++dim) {
                const uint32_t side = static_cast<uint32_t>(region[dim]);
                min_side = std::min(min_side, side);
                max_side = std::max(max_side, side);
            }

            const uint64_t diff = static_cast<uint64_t>(max_side) - min_side;
            if (diff < best_diff || (diff == best_diff && volume < best_volume)) {
                best_diff = diff;
                best_volume = volume;
                best_region = region;
            }
        }
    );

    if (best_volume == UINT64_MAX)
        throw std::runtime_error("Could not find a valid sub-region.");
    return best_region;
}

// picks a compact box with at least one power-of-two side
// => this keeps one recursive axis clean for prefix-truncated curves, while
//    'squareishRegion()' remains the fallback for awkward hardware bounds
// => the preference for dimensions to be power-of-two follows their
//    components order inside the topology/coordinate system
template<MeshLikeTopology T>
inline Coord_t<T> dyadicPrefixRegion(uint32_t nodes, const T& topology, bool* used_fallback = nullptr) {
    using Coord = Coord_t<T>;

    if (nodes > topology.extent().volume())
        throw std::runtime_error("Topology extent too small to hold all nodes.");

    if (used_fallback != nullptr)
        *used_fallback = false;

    if (nodes == 0)
        return Coord{};

    Coord best_region;
    uint64_t best_side_overrun = UINT64_MAX;
    uint64_t best_diff = UINT64_MAX;
    uint64_t best_volume = UINT64_MAX;
    uint32_t best_power_dim = UINT32_MAX;

    iterFittingRegions(
        nodes,
        topology,
        1,
        true,
        [&](const Coord& region, uint64_t volume) {
            uint32_t min_side = UINT32_MAX;
            uint32_t max_side = 0;
            for (uint32_t dim = 0; dim < T::dimensions; ++dim) {
                const uint32_t side = static_cast<uint32_t>(region[dim]);
                min_side = std::min(min_side, side);
                max_side = std::max(max_side, side);
            }
            const uint64_t diff = static_cast<uint64_t>(max_side) - min_side;

            for (uint32_t dim = 0; dim < T::dimensions; ++dim) {
                const uint32_t side = static_cast<uint32_t>(region[dim]);
                if ((side & (side - 1)) != 0)
                    continue;

                const uint64_t side_overrun = max_side > side ? max_side - side : 0;
                if (side_overrun < best_side_overrun ||
                    (side_overrun == best_side_overrun && diff < best_diff) ||
                    (side_overrun == best_side_overrun && diff == best_diff && volume < best_volume) ||
                    (side_overrun == best_side_overrun && diff == best_diff && volume == best_volume && dim < best_power_dim)) {
                    best_side_overrun = side_overrun;
                    best_diff = diff;
                    best_volume = volume;
                    best_power_dim = dim;
                    best_region = region;
                }
            }
        }
    );

    if (best_volume != UINT64_MAX)
        return best_region;

    if (used_fallback != nullptr)
        *used_fallback = true;
    return squareishRegion(nodes, topology);
}

// returns a vector of 'nodes' Hilbert space-filling curve coordinates on the mesh topology
// => lattices and tori behave the same here, topology wrap-around is deliberately ignored
template<MeshLikeTopology T>
inline std::vector<Coord_t<T>> hilbertPlacement(uint32_t nodes, const T& topology, bool verbose = false) {
    using Coord = Coord_t<T>;
    constexpr uint32_t dimensions = T::dimensions;
    std::vector<Coord> result;

    if (nodes > topology.extent().volume())
        throw std::runtime_error("Topology extent too small to hold all nodes.");
    if (nodes == 0)
        return result;

    bool used_fallback = false;
    const Coord region = dyadicPrefixRegion(nodes, topology, &used_fallback);

    if (verbose) {
        std::cout << "Generating Hilbert curve of size: ";
        for (uint32_t dim = 0; dim < dimensions; ++dim) {
            if (dim != 0) std::cout << " x ";
            std::cout << region[dim];
        }
        std::cout << "\n";
        if (used_fallback)
            std::cout << "WARNING: no fitting power-of-two-sided Hilbert region found, using square-ish fallback\n";
    }

    std::array<uint32_t, dimensions> active_dims{};
    uint32_t active_count = 0;
    uint32_t max_side = 1;
    for (uint32_t dim = 0; dim < dimensions; ++dim) {
        if (region[dim] > 1)
            active_dims[active_count++] = dim;
        max_side = std::max(max_side, static_cast<uint32_t>(region[dim]));
    }

    uint32_t bits = 0;
    for (uint32_t side = max_side - 1; side != 0; side >>= 1)
        ++bits;

    struct HilbertPoint {
        Coord coordinate;
        std::array<uint32_t, dimensions> transpose;
    };

    std::vector<HilbertPoint> points;
    points.reserve(region.volume());

    Coord coordinate;
    while (true) {
        HilbertPoint point{coordinate, {}};
        for (uint32_t idx = 0; idx < active_count; ++idx)
            point.transpose[idx] = static_cast<uint32_t>(coordinate[active_dims[idx]]);
        axesToHilbertTranspose(point.transpose, active_count, bits);
        points.push_back(point);

        uint32_t dim = 0;
        for (; dim < dimensions; ++dim) {
            ++coordinate[dim];
            if (coordinate[dim] < region[dim])
                break;
            coordinate[dim] = 0;
        }
        if (dim == dimensions)
            break;
    }

    std::sort(points.begin(), points.end(), [&](const HilbertPoint& lhs, const HilbertPoint& rhs) {
        for (uint32_t bit = bits; bit-- > 0;) {
            for (uint32_t dim = 0; dim < active_count; ++dim) {
                const uint32_t lhs_bit = (lhs.transpose[dim] >> bit) & 1u;
                const uint32_t rhs_bit = (rhs.transpose[dim] >> bit) & 1u;
                if (lhs_bit != rhs_bit)
                    return lhs_bit < rhs_bit;
            }
        }
        return false;
    });

    result.reserve(nodes);
    for (uint32_t idx = 0; idx < nodes; ++idx)
        result.push_back(points[idx].coordinate);
    return result;
}

// returns a vector of coordinates that form a back-and-forth "S" shape in a 'nodes'-sized almost square region of the mesh topology
// => the 'S' proceeds over dimensions in their components order inside the topology/coordinate system, earlier dimensions are traversed first
template<MeshLikeTopology T>
inline std::vector<Coord_t<T>> snakePlacement(uint32_t nodes, const T& topology, bool verbose = false) {
    using Coord = Coord_t<T>;
    std::vector<Coord> result;

    if (nodes > topology.extent().volume())
        throw std::runtime_error("Topology extent too small to hold all nodes.");
    if (nodes == 0)
        return result;

    bool used_fallback = false;
    const Coord region = dyadicPrefixRegion(nodes, topology, &used_fallback);

    if (verbose) {
        std::cout << "Generating serpentine curve of size: ";
        for (uint32_t dim = 0; dim < T::dimensions; ++dim) {
            if (dim != 0) std::cout << " x ";
            std::cout << region[dim];
        }
        std::cout << "\n";
        if (used_fallback)
            std::cout << "WARNING: no fitting power-of-two-sided serpentine region found, using square-ish fallback\n";
    }

    result.reserve(nodes);
    Coord coordinate;

    auto generate = [&](auto&& self, uint32_t dim, bool reverse) -> void {
        if (result.size() >= nodes)
            return;

        const int side = region[dim];
        if (dim == 0) {
            for (int offset = 0; offset < side && result.size() < nodes; ++offset) {
                coordinate[0] = reverse ? side - 1 - offset : offset;
                result.push_back(coordinate);
            }
            return;
        }

        for (int offset = 0; offset < side && result.size() < nodes; ++offset) {
            coordinate[dim] = reverse ? side - 1 - offset : offset;
            self(self, dim - 1, reverse != ((coordinate[dim] & 1) != 0));
        }
    };

    generate(generate, T::dimensions - 1, false);
    return result;
}

// returns a vector of coordinates laid out in Z-order (Morton order)
// over a 'nodes'-sized almost square region of the mesh topology
// => dimensions are deinterleaved in their components order
template<MeshLikeTopology T>
inline std::vector<Coord_t<T>> zorderPlacement(uint32_t nodes, const T& topology, bool verbose = false) {
    using Coord = Coord_t<T>;
    constexpr uint32_t dimensions = T::dimensions;
    std::vector<Coord> result;

    if (nodes > topology.extent().volume())
        throw std::runtime_error("Topology extent too small to hold all nodes.");
    if (nodes == 0)
        return result;

    bool used_fallback = false;
    const Coord region = dyadicPrefixRegion(nodes, topology, &used_fallback);

    if (verbose) {
        std::cout << "Generating Z-order curve of size: ";
        for (uint32_t dim = 0; dim < dimensions; ++dim) {
            if (dim != 0) std::cout << " x ";
            std::cout << region[dim];
        }
        std::cout << "\n";
        if (used_fallback)
            std::cout << "WARNING: no fitting power-of-two-sided Z-order region found, using square-ish fallback\n";
    }

    std::array<uint32_t, dimensions> active_dims{};
    uint32_t active_count = 0;
    uint32_t max_side = 1;
    for (uint32_t dim = 0; dim < dimensions; ++dim) {
        if (region[dim] > 1)
            active_dims[active_count++] = dim;
        max_side = std::max(max_side, static_cast<uint32_t>(region[dim]));
    }

    result.reserve(nodes);
    const uint32_t enclosing_side = nextPow2(max_side);
    auto generate = [&](auto&& self, const Coord& origin, uint32_t side) -> void {
        if (result.size() >= nodes)
            return;

        for (uint32_t dim = 0; dim < dimensions; ++dim) {
            if (origin[dim] >= region[dim])
                return;
        }

        if (side == 1) {
            result.push_back(origin);
            return;
        }

        const uint32_t half = side >> 1;
        Coord child = origin;
        auto visitChildren = [&](auto&& visit, uint32_t active_idx) -> void {
            if (result.size() >= nodes)
                return;
            if (active_idx == 0) {
                self(self, child, half);
                return;
            }

            const uint32_t dim = active_dims[active_idx - 1];
            child[dim] = origin[dim];
            visit(visit, active_idx - 1);
            child[dim] = origin[dim] + static_cast<int>(half);
            visit(visit, active_idx - 1);
        };
        visitChildren(visitChildren, active_count);
    };

    generate(generate, Coord{}, enclosing_side);
    return result;
}

// returns a vector of coordinates laid out according to a cyclic 2^N-tree pattern
// over a 'nodes'-sized almost square region of the mesh topology
// => in 2D the local 2x2 motif remains:
//   0 -> (0, 0)
//   1 -> (0, 1)
//   2 -> (1, 1)
//   3 -> (1, 0)
// => so each group of 4 is arranged circularly as:
//   0 3
//   1 2
// => in higher dimensions the same binary-reflected Gray order visits all 2^N children cyclically
template<MeshLikeTopology T>
inline std::vector<Coord_t<T>> quadPlacement(uint32_t nodes, const T& topology, bool verbose = false) {
    using Coord = Coord_t<T>;
    constexpr uint32_t dimensions = T::dimensions;
    std::vector<Coord> result;

    if (nodes > topology.extent().volume())
        throw std::runtime_error("Topology extent too small to hold all nodes.");
    if (nodes == 0)
        return result;

    bool used_fallback = false;
    const Coord region = dyadicPrefixRegion(nodes, topology, &used_fallback);

    if (verbose) {
        std::cout << "Generating quadtree curve of size: ";
        for (uint32_t dim = 0; dim < dimensions; ++dim) {
            if (dim != 0) std::cout << " x ";
            std::cout << region[dim];
        }
        std::cout << "\n";
        if (used_fallback)
            std::cout << "WARNING: no fitting power-of-two-sided quadtree region found, using square-ish fallback\n";
    }

    std::array<uint32_t, dimensions> active_dims{};
    uint32_t active_count = 0;
    uint32_t max_side = 1;
    for (uint32_t dim = 0; dim < dimensions; ++dim) {
        if (region[dim] > 1)
            active_dims[active_count++] = dim;
        max_side = std::max(max_side, static_cast<uint32_t>(region[dim]));
    }

    result.reserve(nodes);
    const uint32_t enclosing_side = nextPow2(max_side);
    auto generate = [&](auto&& self, const Coord& origin, uint32_t side) -> void {
        if (result.size() >= nodes)
            return;

        for (uint32_t dim = 0; dim < dimensions; ++dim) {
            if (origin[dim] >= region[dim])
                return;
        }

        if (side == 1) {
            result.push_back(origin);
            return;
        }

        const uint32_t half = side >> 1;
        Coord child = origin;
        auto visitChildren = [&](auto&& visit, uint32_t active_idx, bool reverse) -> void {
            if (result.size() >= nodes)
                return;
            if (active_idx == active_count) {
                self(self, child, half);
                return;
            }

            const uint32_t dim = active_dims[active_idx];
            if (!reverse) {
                child[dim] = origin[dim];
                visit(visit, active_idx + 1, false);
                child[dim] = origin[dim] + static_cast<int>(half);
                visit(visit, active_idx + 1, true);
            } else {
                child[dim] = origin[dim] + static_cast<int>(half);
                visit(visit, active_idx + 1, false);
                child[dim] = origin[dim];
                visit(visit, active_idx + 1, true);
            }
        };
        visitChildren(visitChildren, 0, false);
    };

    generate(generate, Coord{}, enclosing_side);
    return result;
}

// EXTERNAL INTERFACE

template<Topology T>
std::vector<Coord_t<T>> generatePlacementCurve(
    SpaceFillingCurve curve,
    uint32_t nodes,
    const T& topology,
    bool verbose
) {
    switch (curve) {
        case SpaceFillingCurve::HILB:
            if constexpr (requires { hilbertPlacement<T>(nodes, topology, verbose); }) {
                return hilbertPlacement<T>(nodes, topology, verbose);
            }
            break;

        case SpaceFillingCurve::SNAK:
            if constexpr (requires { snakePlacement<T>(nodes, topology, verbose); }) {
                return snakePlacement<T>(nodes, topology, verbose);
            }
            break;

        case SpaceFillingCurve::ZORD:
            if constexpr (requires { zorderPlacement<T>(nodes, topology, verbose); }) {
                return zorderPlacement<T>(nodes, topology, verbose);
            }
            break;

        case SpaceFillingCurve::QUAD:
            if constexpr (requires { quadPlacement<T>(nodes, topology, verbose); }) {
                return quadPlacement<T>(nodes, topology, verbose);
            }
            break;
    }

    // if this triggers there is a BUG!!
    // => specifically, a disagreement between 'run_config' validation and the compile-time curve constraints here specified
    throw std::logic_error("Validated curve/topology pair is not compile-time compatible.");
}