#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// TODO: upgrade "uint_32_t"s to "uint64_t"s to handle >4M nodes, >4M touching per node, >4M pins per hedge, >4M neighbors per node!

// USED BY: everyone

// DEVICE CONSTANTS:
extern __constant__ uint32_t max_nodes_per_part;
extern __constant__ uint32_t max_inbound_per_part;

// absolute replacement for "size_t"
using dim_t = unsigned long long; // aka uint64_t

#define WARP_SIZE 32u
// TODO: determine this at runtime w.r.t. the mean and variance of the spike frequency!
#define FIXED_POINT_SCALE 262144u // 256u // used to convert scores to fixed point
// TODO: this is just a good guess on how much more memory give to oversized buffers during deduplication, refine it!
#define OVERSIZED_SIZE_MULTIPLIER 1.5f

#define SAVE_MEMORY_UP_TO_LEVEL 2 // number of coarsening levels for which to spill non-coarse data structures to the host, set to 0 to disable the feature

#define SMALL_PART_MERGE_SIZE_THRESHOLD 15 // number of nodes below which partitions are considered "small" and an attempt is done at merging them with one-another

#define KWAY_INIT_UPPER_THREASHOLD 8192 // number of coarse nodes below which to run the initial partitioning algorithm (KWAY model only)
#define KWAY_INIT_LOWER_THREASHOLD 1024 // number of coarse nodes below which to undo the last coarsening round and run the initial partitioning algorithm (KWAY model only)
#define KWAY_INIT_SHRINK_RATIO_LIMIT 0.95f // coarsening ratio between levels above which to run the initial partitioning algorithm (KWAY model only)

#define INIT_SEED 86 // seed for random initialization (KWAY mode only)


// USED BY: neighborhoods kernel

#define SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE 8192u // 16384 is too big for an A100...
#define GM_MIN_BLOCK_DEDUPE_BUFFER_SIZE 256u


// USED BY: candidates kernel

#define HIST_SIZE 512u // must be a multiple of WARP_SIZE (for the histogram max reduction)
#define MAX_CANDIDATES 16u // => how many candidates are proposed for a node (ranked by score)

#define DETERMINISTIC_SCORE_NOISE 64u // 256u // => adds a +[0, DETERMINISTIC_SCORE_NOISE - 1]/FIXED_POINT_SCALE symmetric noise while calculating pairing scores; set to 0 to disable; keep it a power of 2 otherwise

template <typename T>
struct __align__(8) bin {
    uint32_t payload;
    T val;
};


// USED BY: grouping kernel

#define MAX_GROUP_SIZE 1u // => MAX_GROUP_SIZE - 1 slots per node; 2 means pairs
#define PATH_SIZE 224u // initial slots for nodes to see while traversing the pairs tree, TODO: automatically extend if needed (costly...)
#define MAX_REPEATS 64u // maximum number of nodes a single thread can handle, must be less than 32 (due to using one-hot anti-repeat encoding)

typedef struct __align__(8) {
    uint32_t id; // lower 32 bits (Nvidia GPUs are little-endian)
    uint32_t score; // converted from float to fixed point! higher 32 bits
} slot;

typedef struct __align__(8) {
    uint32_t with; // total score in the traversed subtree assuming the current node will be paired with its targed
    uint32_t wout; // total score in the traversed subtree assuming the current node will NOT be paired with its targed
} dp_score;


// USED BY: coarsening routines (all, touching, hedges, and neighbors)

#define MAX_SM_WARP_DEDUPE_BUFFER_SIZE 3072u // the A100 has 48KB of SM, this is (48KB/4B of uint32s)/4 warps per block
#define MIN_GM_WARP_DEDUPE_BUFFER_SIZE 256u // just for safety, interplays with 'MAX_HASH_PROBE_LENGTH' and 'OVERSIZED_SIZE_MULTIPLIER'


// USED BY: fm refinement kernel

#define REFINE_REPEATS 16u // 64u // 256u // repetitions of FM refinement per uncoarsening level

#define PART_HIST_SIZE 64u // best if it is a multiple of WARP_SIZE, best if partitions_per_thread * WARP_SIZE <= num_partitions


// USED BY: refinement constraints checks

// valid values filtering functor
struct masked_twin_value_functor {
    const float* value;
    const int32_t* valid_1;
    const int32_t* valid_2;
    __host__ __device__ float operator()(uint32_t i) const { return valid_1[i] == 0 && valid_2[i] == 0 ? value[i] : -FLT_MAX; }
};

// valid values filtering functor - one validity entry only
struct masked_value_functor {
    const float* value;
    const int32_t* valid;
    __host__ __device__ float operator()(uint32_t i) const { return valid[i] == 0 ? value[i] : -FLT_MAX; }
};

// custom comparison logic between size and inbound events
struct best_move_functor {
    const float* gain;
    const int32_t* valid_moves;
    const int32_t* inbound_valid_moves;
    __host__ __device__ bool operator()(uint32_t a, uint32_t b) const { // return true -> choose b
        // satisfying the inbound constraint is mandatory
        const bool a_inbound_ok = (inbound_valid_moves[a] == 0);
        const bool b_inbound_ok = (inbound_valid_moves[b] == 0);
        // if only one is valid, choose it
        if (a_inbound_ok != b_inbound_ok) return b_inbound_ok;
        // if neither is valid, keep the earlier one
        if (!a_inbound_ok && !b_inbound_ok) return a > b;
        // minimize size constraint violations
        const int32_t va = valid_moves[a];
        const int32_t vb = valid_moves[b];
        if (va != vb) return va > vb; // fewer violations wins
        // maximize gain
        const float sa = gain[a];
        const float sb = gain[b];
        if (sa != sb) return sa < sb;
        // tie, earlier index wins
        return a > b;
    }
};


// USED BY: final small partitions merging

struct constraints_state {
    dim_t s, i;
    uint32_t g;
};


// HASH-SET

#define HASH_EMPTY 0xFFFFFFFFu
#define MAX_HASH_PROBE_LENGTH 32u


//  HASH-MAP

typedef struct {
    uint32_t key;
    uint32_t value;
} hashmap_entry;