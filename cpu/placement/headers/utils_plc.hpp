#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>

#include <omp.h>

#include "utils.hpp"
#include "data_types.hpp"
#include "data_types_plc.hpp"
#include "defines_plc.hpp"

// USED BY: everyone

// visits every pin across a node's 'touching_count' touching hyperedges, flattening pin iteration across hedge boundaries
// -> calls 'fn(lane, pin, hedge_weight)' once per visited pin, 'lane' being the one accumulating it
// NOTE: in CUDA a warp stages up to WARP_SIZE touching hedges at a time and strides over their pins as one flat sequence,
//       lane "l" visiting the group's flat positions l, l + WARP_SIZE, ...; the same positions are replayed here, so that
//       per-lane float sums, reduced afterwards, associate as in CUDA
template<typename Fn>
inline void forEachTouchingPin(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t* __restrict__ my_touching,
    const uint32_t touching_count,
    const uint32_t nodes_base, // offset of the multi-start owning these pins, 0 outside of batched kernels
    Fn&& fn
) {
    for (uint32_t group_start = 0; group_start < touching_count; group_start += WARP_SIZE) {
        const uint32_t group_size = std::min(WARP_SIZE, touching_count - group_start);
        uint32_t flat_pos = 0u; // position of the current pin in the group's flat pin sequence
        for (uint32_t h = 0; h < group_size; h++) {
            const uint32_t hedge_idx = my_touching[group_start + h];
            const float hedge_weight = hedge_weights[hedge_idx];
            for (dim_t i = hedges_offsets[hedge_idx]; i < hedges_offsets[hedge_idx + 1]; i++, flat_pos++)
                fn(LANE_OF(flat_pos), nodes_base + hedges[i], hedge_weight); // pins are node idxs local to a multi-start, rebase them
        }
    }
}


// USED BY: recursive bisection

// XORWOW
// NOTE: in CUDA random keys come from curand's host API, one XORWOW generator per multi-start ("CURAND_RNG_PSEUDO_DEFAULT",
//       default ordering), the generator is copied here to draw the very same numbers:
// => the g-th number drawn by a generator (counted across calls) is the (g / 4096)-th of its (g % 4096)-th subsequence
// => each subsequence starts 2^67 steps after the previous one, the jump matrix (the step raised to 2^67) gets there

#define XORWOW_SUBSEQUENCES 4096u
#define XORWOW_SUBSEQUENCE_LOG 67u

struct xorwow_state {
    uint32_t v[5]; // xorshift state
    uint32_t d; // weyl sequence
};

// one xorwow step, as 'curand(curandStateXORWOW_t*)'
inline uint32_t xorwow_next(xorwow_state &state) {
    const uint32_t t = state.v[0] ^ (state.v[0] >> 2);
    state.v[0] = state.v[1];
    state.v[1] = state.v[2];
    state.v[2] = state.v[3];
    state.v[3] = state.v[4];
    state.v[4] = (state.v[4] ^ (state.v[4] << 4)) ^ (t ^ (t << 1));
    state.d += 362437;
    return state.v[4] + state.d;
}

// the xorshift part of a step is linear over GF(2): a 160x160 bit matrix, stored by columns
// => column c -> image of the state having only bit c set, 5 words (v[0..4])
struct xorwow_matrix {
    uint32_t col[160][5];

    // image of 'v' through the matrix, the xor of the columns of v's set bits
    void apply(const uint32_t (&v)[5], uint32_t (&out)[5]) const {
        uint32_t acc[5] = {};
        for (uint32_t c = 0; c < 160; c++)
            if ((v[c >> 5] >> (c & 31)) & 1u)
                for (uint32_t w = 0; w < 5; w++) acc[w] ^= col[c][w];
        for (uint32_t w = 0; w < 5; w++) out[w] = acc[w];
    }
};

// the jump matrix, from the start of a subsequence to the start of the next one
inline const xorwow_matrix& xorwow_jump() {
    static const xorwow_matrix jump = [] {
        xorwow_matrix m;
        for (uint32_t c = 0; c < 160; c++) {
            xorwow_state unit = {};
            unit.v[c >> 5] = 1u << (c & 31);
            xorwow_next(unit);
            for (uint32_t w = 0; w < 5; w++) m.col[c][w] = unit.v[w];
        }
        // square the step 67 times
        for (uint32_t sq = 0; sq < XORWOW_SUBSEQUENCE_LOG; sq++) {
            xorwow_matrix m2;
            for (uint32_t c = 0; c < 160; c++) m.apply(m.col[c], m2.col[c]);
            m = m2;
        }
        return m;
    }();
    return jump;
}

class xorwow_generator {
    std::vector<xorwow_state> subsequences_; // subsequences_[s] -> state of the s-th subsequence, past the numbers it already drew
    uint64_t drawn_ = 0; // numbers drawn so far

    public:
    // seed as 'curand_init(seed, s, 0, ...)' does, for every subsequence 's'
    explicit xorwow_generator(const uint64_t seed) : subsequences_(XORWOW_SUBSEQUENCES) {
        const uint32_t s0 = ((uint32_t)seed) ^ 0xaad26b49u;
        const uint32_t s1 = (uint32_t)(seed >> 32) ^ 0xf7dcefddu;
        const uint32_t t0 = 1099087573u * s0;
        const uint32_t t1 = 2591861531u * s1;
        xorwow_state state;
        state.d = 6615241u + t1 + t0;
        state.v[0] = 123456789u + t0;
        state.v[1] = 362436069u ^ t0;
        state.v[2] = 521288629u + t1;
        state.v[3] = 88675123u ^ t1;
        state.v[4] = 5783321u + t0;
        const xorwow_matrix &jump = xorwow_jump();
        for (uint32_t s = 0; s < XORWOW_SUBSEQUENCES; s++) {
            subsequences_[s] = state;
            jump.apply(subsequences_[s].v, state.v); // the weyl sequence does not jump
        }
    }

    // draw the next 'n' numbers, as 'curandGenerate(gen, out, n)'
    void generate(uint32_t* __restrict__ out, const uint32_t n) {
        // STYLE: one subsequence per iteration!
        #pragma omp parallel for schedule(static) if(n > PARALLEL_GRAIN)
        for (uint32_t s = 0; s < XORWOW_SUBSEQUENCES; s++) {
            const uint64_t first = drawn_ + (s + XORWOW_SUBSEQUENCES - drawn_ % XORWOW_SUBSEQUENCES) % XORWOW_SUBSEQUENCES; // first number of the call from this subsequence
            for (uint64_t g = first; g < drawn_ + n; g += XORWOW_SUBSEQUENCES)
                out[g - drawn_] = xorwow_next(subsequences_[s]);
        }
        drawn_ += n;
    }
};
