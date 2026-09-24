#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// USED BY: everyone

// absolute replacement for "size_t"
using dim_t = unsigned long long; // aka uint64_t


// USED BY: grouping kernel

// NOTE: a slot is handled as a packed 64-bit word, high 32 bits = score, low 32 bits = node id
using slot = uint64_t;


// USED BY: refinement constraints checks

// custom comparison logic between size, inbound, and pins events
struct best_move_functor {
    const float* gain;
    const int32_t* valid_moves;
    const int32_t* inbound_valid_moves;
    const int32_t* pins_valid_moves;
    bool operator()(uint32_t a, uint32_t b) const { // return true -> choose b
        // satisfying the inbound constraint is mandatory
        const bool a_inbound_ok = (inbound_valid_moves[a] == 0);
        const bool b_inbound_ok = (inbound_valid_moves[b] == 0);
        // if only one is valid, choose it
        if (a_inbound_ok != b_inbound_ok) return b_inbound_ok;
        // if neither is valid, keep the earlier one
        if (!a_inbound_ok && !b_inbound_ok) return a > b;
        // minimize size and pins constraint violations, size first
        const int32_t va = valid_moves[a];
        const int32_t vb = valid_moves[b];
        if (va != vb) return va > vb; // fewer violations wins
        const int32_t pa = pins_valid_moves[a];
        const int32_t pb = pins_valid_moves[b];
        if (pa != pb) return pa > pb; // fewer violations wins
        // maximize gain
        const float sa = gain[a];
        const float sb = gain[b];
        if (sa != sb) return sa < sb;
        // tie, earlier index wins
        return a > b;
    }
};


// USED BY: pins per partition (sparse bitmap matrix)

#define BITMAP_CAPACITY 64u // number of bits/elements flagged by a bitmap instance
#define BITMAP_CAPLOG 6u // log_2(BITMASK_CAPACITY) -> how many bits are needed to index inside "flg"

struct bitmap {
    uint64_t cnt; // counter of how many entries exist before mines
    uint64_t flg; // i-th bit set to 1 if the cnt+i element is present
};
