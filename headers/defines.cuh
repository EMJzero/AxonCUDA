#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// NOTE: put here program-wide constants
// => put into each header kernel-specific constants

// USED BY: main

#define DEVICE_ID 0

#define VERBOSE false
#define VERBOSE_LENGTH 20


// USED BY: everyone

#define WARP_SIZE 32u
// TODO: determine this at runtime w.r.t. the mean and variance of the spike frequency!
#define FIXED_POINT_SCALE 262144u // 256u // used to convert scores to fixed point
// TODO: this is just a good guess on how much more memory give to oversized buffers during deduplication, refine it!
#define OVERSIZED_SIZE_MULTIPLIER 1.5f

#define SHRINK_RATIO_LIMIT 0.95f // coarsening ratio between levels above which to stop and start uncoarsening
#define NUMBER_OF_LEVELS_WITH_NO_SHRINK_LIMIT 32u // number of coarsening levels that are performed even if the 'SHRINK_RATIO_LIMIT' was reached

#define SAVE_MEMORY_UP_TO_LEVEL 2 // number of coarsening levels for which to spill non-coarse data structures to the host, set to 0 to disable the feature

#define SMALL_PART_MERGE_SIZE_THRESHOLD 15 // number of nodes below which partitions are considered "small" and an attempt is done at merging them with one-another

#define KWAY_INIT_UPPER_THREASHOLD 8192 // number of coarse nodes below which to run the initial partitioning algorithm (KWAY model only)
#define KWAY_INIT_LOWER_THREASHOLD 1024 // number of coarse nodes below which to undo the last coarsening round and run the initial partitioning algorithm (KWAY model only)
#define KWAY_INIT_SHRINK_RATIO_LIMIT 0.95f // coarsening ratio between levels above which to run the initial partitioning algorithm (KWAY model only)

#define INIT_SEED 86 // seed for random initialization (KWAY mode only)


// USED BY: neighborhoods kernel

#define NEIGHBORS_SAMPLE_SIZE 2400


// USED BY: candidates kernel

#define MAX_CANDIDATES 16u // => how many candidates are proposed for a node (ranked by score)


// USED BY: fm refinement kernel

#define REFINE_REPEATS 16u // 64u // 256u // repetitions of FM refinement per uncoarsening level


// USED BY: hash-sets, hash-maps

#define HASH_EMPTY 0xFFFFFFFFu
#define MAX_HASH_PROBE_LENGTH 32u