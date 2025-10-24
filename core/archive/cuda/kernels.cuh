/**
 * @file kernels.cuh
 * @brief Shared CUDA kernel utilities and definitions
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace wcodec {
namespace cuda {

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Constants
constexpr int TILE_SIZE = 16;
constexpr int TILE_THREADS = 256;  // Threads per tile
constexpr int WARP_SIZE = 32;
constexpr int RANS_PRECISION = 16;  // 16-bit precision
constexpr uint32_t RANS_L = (1u << RANS_PRECISION);

// Predictor modes
enum class PredictorMode : uint8_t {
    LEFT = 0,
    TOP = 1,
    AVG = 2,
    PLANAR = 3
};

// Shared memory structure for tile decode
struct TileSharedMem {
    // Frequency table (shared across tile)
    uint32_t freq_table[256];
    uint32_t cumul_freq[257];
    
    // Decoded residuals
    int8_t residuals[TILE_SIZE * TILE_SIZE];
    
    // Reconstructed values
    int8_t reconstructed[TILE_SIZE * TILE_SIZE];
};

// Device helper functions
__device__ __forceinline__
int clamp_int8(int val) {
    return min(127, max(-128, val));
}

__device__ __forceinline__
int tile_idx_2d(int row, int col, int tile_cols) {
    return row * tile_cols + col;
}

__device__ __forceinline__
int pixel_idx(int row, int col) {
    return row * TILE_SIZE + col;
}

} // namespace cuda
} // namespace wcodec

