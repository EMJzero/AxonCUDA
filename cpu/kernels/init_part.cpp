#include <cmath>

#include "init_part.hpp"
#include "utils.hpp"

// PHILOX

// NOTE: in CUDA random numbers come from curand's Philox4x32-10, via "curand_init(seed, subsequence, 0, &state)" and
//       "curand_uniform(&state)", the generator is copied here to draw the very same numbers

#define PHILOX_W32_0   (0x9E3779B9u)
#define PHILOX_W32_1   (0xBB67AE85u)
#define PHILOX_M4x32_0 (0xD2511F53u)
#define PHILOX_M4x32_1 (0xCD9E8D57u)
#define CURAND_2POW32_INV (2.3283064e-10f)

struct philox_ctr { uint32_t x, y, z, w; };

static inline philox_ctr philox4x32round(const philox_ctr ctr, const uint32_t key_x, const uint32_t key_y) {
    const uint64_t product0 = (uint64_t)PHILOX_M4x32_0 * ctr.x;
    const uint64_t product1 = (uint64_t)PHILOX_M4x32_1 * ctr.z;
    const uint32_t hi0 = (uint32_t)(product0 >> 32), lo0 = (uint32_t)product0;
    const uint32_t hi1 = (uint32_t)(product1 >> 32), lo1 = (uint32_t)product1;
    return { hi1 ^ ctr.y ^ key_x, lo1, hi0 ^ ctr.w ^ key_y, lo0 };
}

// first uniform float in (0, 1] of the Philox4x32-10 stream 'subsequence' of 'seed'
static inline float philox_uniform_first(const uint64_t seed, const uint64_t subsequence) {
    philox_ctr ctr { 0u, 0u, (uint32_t)subsequence, (uint32_t)(subsequence >> 32) }; // skipahead_sequence(subsequence)
    uint32_t key_x = (uint32_t)seed, key_y = (uint32_t)(seed >> 32);
    for (int round = 0; round < 9; round++) {
        ctr = philox4x32round(ctr, key_x, key_y);
        key_x += PHILOX_W32_0;
        key_y += PHILOX_W32_1;
    }
    ctr = philox4x32round(ctr, key_x, key_y);
    // NOTE: in CUDA nvcc fuses "x * CURAND_2POW32_INV + CURAND_2POW32_INV/2" into a single FFMA, hence the explicit 'fmaf'
    return std::fmaf((float)ctr.x, CURAND_2POW32_INV, CURAND_2POW32_INV/2.0f);
}


// K-WAY INITIAL PARTITIONING

// compute each hedge's weight over its pins' armonic degree
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void armonic_degree_score_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const dim_t* __restrict__ touching_offsets,
    const float* __restrict__ hedge_weights,
    const uint32_t num_hedges,
    float* __restrict__ hedge_ratio
) {
    // STYLE: one hedge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
        // NOTE: summed lane by lane, then reduced along the same tree as 'warpReduceSumLN0'
        float score_lanes[WARP_SIZE] = {};
        const dim_t hedge_start_idx = hedges_offsets[hedge_idx];
        for (dim_t i = hedge_start_idx; i < hedges_offsets[hedge_idx + 1]; i++) {
            const uint32_t pin = hedges[i];
            const uint32_t deg = (uint32_t)(touching_offsets[pin + 1] - touching_offsets[pin]);
            score_lanes[LANE_OF(i - hedge_start_idx)] += 1/(float)deg;
        }
        const float score = lanesReduceSumLN0<float>(score_lanes);
        hedge_ratio[hedge_idx] = score > 0.0f ? hedge_weights[hedge_idx] / score : 0.0f;
    }
}

// flag each hedge as 'keep' or 'prune' with a probability conditioned on its weight and score
// SEQUENTIAL COMPLEXITY: e
// PARALLEL OVER: e
void prune_hedges_kernel(
    const float* __restrict__ hedge_weights,
    const float* __restrict__ hedge_ratio,
    const uint32_t num_hedges,
    const float threshold,
    const uint32_t seed,
    float* __restrict__ hedge_scaled_weights,
    uint8_t* __restrict__ keep
) {
    // STYLE: one hedge per iteration!
    #pragma omp parallel for schedule(static) if(num_hedges > PARALLEL_GRAIN)
    for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
        // probability of keeping the hedge: p(e) = min(1, threshold * w(e) / score(e))
        const float keep_prob = std::min(1.0f, threshold * hedge_ratio[hedge_idx]);

        // sample random float in (0, 1]
        const float rand = philox_uniform_first(seed, hedge_idx);

        keep[hedge_idx] = keep_prob >= rand;
        // rescale weights (lower probability -> higher weight)
        hedge_scaled_weights[hedge_idx] = keep_prob > 0.0f ? hedge_weights[hedge_idx] / keep_prob : 0.0f;
    }
}
