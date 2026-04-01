#pragma once
#include <cfloat>
#include <cstdint>
#include <stdint.h>

// USED BY: main

#define DEVICE_ID 0

#define VERBOSE false
#define VERBOSE_LENGTH 20

// USED BY: everyone

#define SEED 86u

#define MAX_ITERATIONS 32 // 1024

#define LEFT 0
#define RIGHT 1
#define UP 2
#define DOWN 3

// TODO: infer this at runtime, make it a device-side constant
//       => infer it especially from the hardware width/height, that determine the manhattan distance range
#define FORCE_FIXED_POINT_SCALE 262144u