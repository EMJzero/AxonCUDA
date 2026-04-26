#pragma once
#include <cstdlib>
#include <cfloat>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "data_types_plc.cuh"

// find the most square rectangle (rw x rh) such that:
//   rw <= width, rh <= height, rw * rh >= nodes
//   rw and rh are multiples of side_multiple
// prefer: smallest |rw - rh|, then smallest area
inline dimensions squareishRegion(uint32_t nodes, uint32_t width, uint32_t height, uint32_t side_multiple = 1) {
    if (static_cast<uint64_t>(nodes) > static_cast<uint64_t>(width) * static_cast<uint64_t>(height))
        throw std::runtime_error("Grid too small to hold all nodes.");

    if (nodes == 0)
        return (dimensions){ 0, 0 };

    if (side_multiple == 0)
        side_multiple = 1;

    auto roundUp = [](uint64_t value, uint64_t multiple) -> uint64_t {
        return ((value + multiple - 1) / multiple) * multiple;
    };

    uint32_t best_w = 0;
    uint32_t best_h = 0;
    uint64_t best_diff = UINT64_MAX;
    uint64_t best_area = UINT64_MAX;

    uint64_t step = side_multiple;
    uint64_t max_w = width;

    if (nodes > step && nodes < max_w)
        max_w = nodes;
    else if (nodes <= step && step < max_w)
        max_w = step;

    max_w = (max_w / step) * step;

    for (uint64_t rw = step; rw <= max_w; rw += step) {
        uint64_t rh = roundUp((static_cast<uint64_t>(nodes) + rw - 1) / rw, step);
        if (rh > height)
            continue;

        uint64_t diff = (rw > rh) ? (rw - rh) : (rh - rw);
        uint64_t area = rw * rh;

        if (diff < best_diff || (diff == best_diff && area < best_area)) {
            best_diff = diff;
            best_area = area;
            best_w = static_cast<uint32_t>(rw);
            best_h = static_cast<uint32_t>(rh);
        }
    }

    if (best_w == 0 || best_h == 0)
        throw std::runtime_error("Could not find a valid sub-region.");

    return (dimensions){ best_w, best_h };
}

inline uint32_t nextPow2(uint32_t v) {
    if (v <= 1)
        return 1;
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

inline uint64_t divCeil(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

// Pick a compact rectangle with at least one power-of-two side. This keeps one
// recursive axis clean for prefix-truncated curves, while squareishRegion()
// remains the fallback for awkward hardware bounds.
inline dimensions dyadicPrefixRegion(
    uint32_t nodes,
    uint32_t width,
    uint32_t height,
    bool prefer_vertical = true,
    bool* used_fallback = nullptr
) {
    if (static_cast<uint64_t>(nodes) > static_cast<uint64_t>(width) * height)
        throw std::runtime_error("Grid too small to hold all nodes.");

    if (used_fallback != nullptr)
        *used_fallback = false;

    if (nodes == 0)
        return (dimensions){ 0, 0 };

    uint32_t best_w = 0;
    uint32_t best_h = 0;
    uint64_t best_side_overrun = UINT64_MAX;
    uint64_t best_diff = UINT64_MAX;
    uint64_t best_area = UINT64_MAX;
    uint64_t best_orientation_penalty = UINT64_MAX;

    auto consider = [&](uint32_t rw, uint32_t rh, bool power_side_is_height) {
        if (rw == 0 || rh == 0 || rw > width || rh > height)
            return;

        uint64_t area = static_cast<uint64_t>(rw) * rh;
        if (area < nodes)
            return;

        uint64_t diff = (rw > rh) ? (rw - rh) : (rh - rw);
        uint64_t side_overrun = power_side_is_height
            ? ((rw > rh) ? (rw - rh) : 0)
            : ((rh > rw) ? (rh - rw) : 0);
        uint64_t orientation_penalty = (power_side_is_height == prefer_vertical) ? 0 : 1;

        if (side_overrun < best_side_overrun ||
            (side_overrun == best_side_overrun && diff < best_diff) ||
            (side_overrun == best_side_overrun && diff == best_diff && area < best_area) ||
            (side_overrun == best_side_overrun && diff == best_diff && area == best_area && orientation_penalty < best_orientation_penalty)) {
            best_side_overrun = side_overrun;
            best_diff = diff;
            best_area = area;
            best_orientation_penalty = orientation_penalty;
            best_w = rw;
            best_h = rh;
        }
    };

    for (uint32_t pow2_h = 1; pow2_h != 0 && pow2_h <= height; pow2_h <<= 1) {
        uint64_t rw = divCeil(nodes, pow2_h);
        if (rw <= UINT32_MAX)
            consider(static_cast<uint32_t>(rw), pow2_h, true);
    }

    for (uint32_t pow2_w = 1; pow2_w != 0 && pow2_w <= width; pow2_w <<= 1) {
        uint64_t rh = divCeil(nodes, pow2_w);
        if (rh <= UINT32_MAX)
            consider(pow2_w, static_cast<uint32_t>(rh), false);
    }

    if (best_w != 0 && best_h != 0)
        return (dimensions){ best_w, best_h };

    if (used_fallback != nullptr)
        *used_fallback = true;
    return squareishRegion(nodes, width, height);
}

// returns a vector of 'nodes' Hilbert space-filling curve coordinates on the 'width x height' lattice
inline std::vector<coords> hilbertPlacement(uint32_t nodes, uint32_t width, uint32_t height, bool verbose = false) {
    auto sgn = [](int x) -> int { return (x > 0) - (x < 0); };
    std::vector<coords> result;

    if (nodes == 0)
        return result;

    bool used_fallback = false;
    dimensions region = dyadicPrefixRegion(nodes, width, height, height >= width, &used_fallback);
    width = region.w;
    height = region.h;

    if (verbose) {
        std::cout << "Generating Hilbert curve of size: " << width << " x " << height << "\n";
        if (used_fallback)
            std::cout << "WARNING: no fitting power-of-two-sided Hilbert region found, using square-ish fallback\n";
    }

    result.reserve(nodes);
    // recursive generator
    auto generate = [&](auto&& self, int x, int y, int ax, int ay, int bx, int by) -> void {
        if (result.size() >= nodes) return;
        int w = std::abs(ax + ay);
        int h = std::abs(bx + by);

        int dax = sgn(ax);
        int day = sgn(ay);
        int dbx = sgn(bx);
        int dby = sgn(by);

        // trivial row fill
        if (h == 1) {
            for (int i = 0; i < w && result.size() < nodes; ++i) {
                result.push_back((coords){ x, y });
                x += dax;
                y += day;
            }
            return;
        }

        // trivial column fill
        if (w == 1) {
            for (int i = 0; i < h && result.size() < nodes; ++i) {
                result.push_back((coords){ x, y });
                x += dbx;
                y += dby;
            }
            return;
        }

        int ax2 = ax / 2;
        int ay2 = ay / 2;
        int bx2 = bx / 2;
        int by2 = by / 2;

        int w2 = std::abs(ax2 + ay2);
        int h2 = std::abs(bx2 + by2);

        if (2 * w > 3 * h) {
            if ((w2 & 1) && w > 2) {
                ax2 += dax;
                ay2 += day;
            }

            // long case
            self(self, x, y, ax2, ay2, bx, by);
            self(self, x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by);
        } else {
            if ((h2 & 1) && h > 2) {
                bx2 += dbx;
                by2 += dby;
            }

            // standard case
            self(self, x, y, bx2, by2, ax2, ay2);
            self(self, x + bx2, y + by2, ax, ay, bx - bx2, by - by2);
            self(self, x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby), -bx2, -by2, -(ax - ax2), -(ay - ay2));
        }
    };

    if (width >= height)
        generate(generate, 0, 0, (int)width, 0, 0, (int)height);
    else
        generate(generate, 0, 0, 0, (int)height, (int)width, 0);

    return result;
}

// returns a vector of coordinates that form an back-and-forth "S" shape in a 'nodes'-sized almost square region of the 'width x height' grid
// when 'vertical_first' is true, the 'S' proceeds over ys first, and steps between xs column after column
inline std::vector<coords> snakePlacement(uint32_t nodes, uint32_t width, uint32_t height, bool vertical_first = true, bool verbose = false) {
    std::vector<coords> result;

    if (nodes == 0)
        return result;

    bool used_fallback = false;
    dimensions region = dyadicPrefixRegion(nodes, width, height, vertical_first, &used_fallback);
    uint32_t best_w = region.w;
    uint32_t best_h = region.h;

    if (verbose) {
        std::cout << "Generating serpentine curve of size: " << best_w << " x " << best_h << "\n";
        if (used_fallback)
            std::cout << "WARNING: no fitting power-of-two-sided serpentine region found, using square-ish fallback\n";
    }

    result.reserve(nodes);

    if (vertical_first) {
        for (int x = 0; x < (int)best_w && result.size() < nodes; ++x) {
            if ((x & 1u) == 0) {
                for (int y = 0; y < (int)best_h && result.size() < nodes; ++y)
                    result.push_back((coords){ x, y });
            } else {
                for (int y = (int)best_h; y-- > 0 && result.size() < nodes;)
                    result.push_back((coords){ x, y });
            }
        }
    } else {
        for (int y = 0; y < (int)best_h && result.size() < nodes; ++y) {
            if ((y & 1u) == 0) {
                for (int x = 0; x < (int)best_w && result.size() < nodes; ++x)
                    result.push_back((coords){ x, y });
            } else {
                for (int x = (int)best_w; x-- > 0 && result.size() < nodes;)
                    result.push_back((coords){ x, y });
            }
        }
    }

    return result;
}

// returns a vector of coordinates laid out in Z-order (Morton order)
// over a 'nodes'-sized almost square region of the 'width x height' grid
inline std::vector<coords> zorderPlacement(uint32_t nodes, uint32_t width, uint32_t height, bool verbose = false) {
    std::vector<coords> result;

    if (nodes == 0)
        return result;

    bool used_fallback = false;
    dimensions region = dyadicPrefixRegion(nodes, width, height, true, &used_fallback);
    uint32_t best_w = region.w;
    uint32_t best_h = region.h;

    if (verbose) {
        std::cout << "Generating Z-order curve of size: " << best_w << " x " << best_h << "\n";
        if (used_fallback)
            std::cout << "WARNING: no fitting power-of-two-sided Z-order region found, using square-ish fallback\n";
    }

    result.reserve(nodes);

    // decode Morton code -> (x, y) by deinterleaving even/odd bits
    auto mortonDecode = [](uint64_t code, uint32_t& x, uint32_t& y) -> void {
        x = 0;
        y = 0;
        for (uint32_t i = 0; i < 32; ++i) {
            x |= static_cast<uint32_t>((code >> (2 * i)) & 1ull) << i;
            y |= static_cast<uint32_t>((code >> (2 * i + 1)) & 1ull) << i;
        }
    };

    uint32_t side = nextPow2((best_w > best_h) ? best_w : best_h);
    uint64_t limit = static_cast<uint64_t>(side) * side;

    for (uint64_t code = 0; code < limit && result.size() < nodes; ++code) {
        uint32_t x, y;
        mortonDecode(code, x, y);

        if (x < best_w && y < best_h)
            result.push_back((coords){ static_cast<int>(x), static_cast<int>(y) });
    }

    return result;
}

// returns a vector of coordinates laid out according to a cyclic quadtree pattern
// over a 'nodes'-sized almost square region of the 'width x height' grid
// => the local 2x2 motif is:
//   0 -> (0, 0)
//   1 -> (0, 1)
//   2 -> (1, 1)
//   3 -> (1, 0)
// => so each group of 4 is arranged circularly as:
//   0 3
//   1 2
// => this matches a hierarchy where every two binary levels are naturally grouped
// into one 4-way split with circular locality
inline std::vector<coords> quadPlacement(uint32_t nodes, uint32_t width, uint32_t height, bool verbose = false) {
    if (static_cast<uint64_t>(nodes) > static_cast<uint64_t>(width) * static_cast<uint64_t>(height))
        throw std::runtime_error("Grid too small to hold all nodes.");

    std::vector<coords> result;

    if (nodes == 0)
        return result;

    bool used_fallback = false;
    dimensions region = dyadicPrefixRegion(nodes, width, height, true, &used_fallback);
    uint32_t best_w = region.w;
    uint32_t best_h = region.h;

    if (verbose) {
        std::cout << "Generating quadtree curve of size: " << best_w << " x " << best_h << "\n";
        if (used_fallback)
            std::cout << "WARNING: no fitting power-of-two-sided quadtree region found, using square-ish fallback\n";
    }

    result.reserve(nodes);

    auto decodeCircularQuad = [](uint64_t index, uint32_t levels, uint32_t& x, uint32_t& y) -> void {
        x = 0;
        y = 0;

        for (uint32_t lvl = 0; lvl < levels; ++lvl) {
            uint32_t digit = static_cast<uint32_t>((index >> (2 * (levels - 1 - lvl))) & 0x3ull);

            uint32_t xb = 0;
            uint32_t yb = 0;

            switch (digit) {
                case 0: xb = 0; yb = 0; break;
                case 1: xb = 0; yb = 1; break;
                case 2: xb = 1; yb = 1; break;
                case 3: xb = 1; yb = 0; break;
            }

            x = (x << 1) | xb;
            y = (y << 1) | yb;
        }
    };

    uint32_t side = nextPow2((best_w > best_h) ? best_w : best_h);

    uint32_t levels = 0;
    for (uint32_t s = side; s > 1; s >>= 1)
        ++levels;

    uint64_t limit = static_cast<uint64_t>(side) * side;

    for (uint64_t i = 0; i < limit && result.size() < nodes; ++i) {
        uint32_t x, y;
        decodeCircularQuad(i, levels, x, y);

        if (x < best_w && y < best_h)
            result.push_back((coords){ static_cast<int>(x), static_cast<int>(y) });
    }

    return result;
}
