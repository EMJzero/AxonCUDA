#include "prep.hpp"
#include "utils.hpp"

// count how many hedges touch each node
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void touching_count_kernel(
    const uint32_t* __restrict__ hedges, // stores srcs first, then dsts
    const dim_t* __restrict__ hedges_offsets,
    const uint32_t num_hedges,
    dim_t* __restrict__ touching_offsets // initialized at 0s
) {
    // STYLE: one hedge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
        const uint32_t* hedge = hedges + hedges_offsets[hedge_idx];
        const uint32_t hedge_size = (uint32_t)(hedges_offsets[hedge_idx + 1] - hedges_offsets[hedge_idx]);
        for (uint32_t pin_idx = 0; pin_idx < hedge_size; pin_idx++)
            atomic_add<dim_t>(&touching_offsets[hedge[pin_idx] + 1], 1ull); // leave the first entry to be 0 (offset of the first set)
    }
}

// write incidence sets
// SEQUENTIAL COMPLEXITY: e*d
// PARALLEL OVER: e
void touching_build_kernel(
    const uint32_t* __restrict__ hedges,
    const dim_t* __restrict__ hedges_offsets,
    const dim_t* __restrict__ touching_offsets,
    const uint32_t num_hedges,
    uint32_t* __restrict__ touching,
    uint32_t* __restrict__ inserted_count // initialized at 0s
) {
    // STYLE: one hedge per iteration!
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (uint32_t hedge_idx = 0; hedge_idx < num_hedges; hedge_idx++) {
        const uint32_t* hedge = hedges + hedges_offsets[hedge_idx];
        const uint32_t hedge_size = (uint32_t)(hedges_offsets[hedge_idx + 1] - hedges_offsets[hedge_idx]);
        for (uint32_t pin_idx = 0; pin_idx < hedge_size; pin_idx++) {
            const uint32_t pin = hedge[pin_idx];
            const uint32_t insert_idx = atomic_add<uint32_t>(&inserted_count[pin], 1u);
            touching[touching_offsets[pin] + insert_idx] = hedge_idx;
        }
    }
}
