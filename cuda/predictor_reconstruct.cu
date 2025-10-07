/**
 * @file predictor_reconstruct.cu
 * @brief GPU-side predictor reconstruction
 */

#include "kernels.cuh"

namespace wcodec {
namespace cuda {

/**
 * Get predictor reference values
 */
__device__ void get_predictor_refs(
    const int8_t* reconstructed,
    int row,
    int col,
    int tile_row,
    int tile_col,
    int tile_cols,
    int8_t& left,
    int8_t& top,
    int8_t& top_left
) {
    left = (col > 0) ? reconstructed[pixel_idx(row, col - 1)] : 0;
    top = (row > 0) ? reconstructed[pixel_idx(row - 1, col)] : 0;
    top_left = (row > 0 && col > 0) ? reconstructed[pixel_idx(row - 1, col - 1)] : 0;
    
    // Handle tile boundaries (would need neighbor tile data)
    // Simplified: assume 0 for now
}

/**
 * Predict value based on mode
 */
__device__ int8_t predict_pixel(
    int8_t left,
    int8_t top,
    int8_t top_left,
    PredictorMode mode
) {
    switch (mode) {
        case PredictorMode::LEFT:
            return left;
        
        case PredictorMode::TOP:
            return top;
        
        case PredictorMode::AVG:
            return clamp_int8((int(left) + int(top)) / 2);
        
        case PredictorMode::PLANAR: {
            // Simplified planar
            int pred = int(left) + int(top) - int(top_left);
            return clamp_int8(pred);
        }
        
        default:
            return 0;
    }
}

/**
 * Reconstruct tile from residuals using predictor
 * Each threadblock = 1 tile
 */
__global__ void reconstruct_tile_kernel(
    const int8_t* residuals,
    const uint8_t* predictor_modes,
    int8_t* output,
    int num_tiles,
    int tile_rows,
    int tile_cols,
    int output_rows,
    int output_cols
) {
    int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;
    
    int tid = threadIdx.x;
    
    // Shared memory for tile reconstruction
    extern __shared__ char shared_mem[];
    TileSharedMem* smem = reinterpret_cast<TileSharedMem*>(shared_mem);
    
    // Load residuals to shared memory
    int tile_row = tile_id / tile_cols;
    int tile_col = tile_id % tile_cols;
    int global_offset = (tile_row * TILE_SIZE * tile_cols * TILE_SIZE) + 
                        (tile_col * TILE_SIZE);
    
    for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
        int local_row = i / TILE_SIZE;
        int local_col = i % TILE_SIZE;
        int global_idx = global_offset + local_row * tile_cols * TILE_SIZE + local_col;
        smem->residuals[i] = residuals[global_idx];
    }
    
    __syncthreads();
    
    // Get predictor mode for this tile
    PredictorMode mode = static_cast<PredictorMode>(predictor_modes[tile_id]);
    
    // Reconstruct in raster order (sequential for dependencies)
    if (tid == 0) {
        for (int row = 0; row < TILE_SIZE; row++) {
            for (int col = 0; col < TILE_SIZE; col++) {
                int idx = pixel_idx(row, col);
                
                // Get prediction references
                int8_t left, top, top_left;
                get_predictor_refs(
                    smem->reconstructed,
                    row, col,
                    tile_row, tile_col, tile_cols,
                    left, top, top_left
                );
                
                // Predict + residual
                int8_t pred = predict_pixel(left, top, top_left, mode);
                int8_t residual = smem->residuals[idx];
                smem->reconstructed[idx] = clamp_int8(int(pred) + int(residual));
            }
        }
    }
    
    __syncthreads();
    
    // Write reconstructed tile to global memory
    for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
        int local_row = i / TILE_SIZE;
        int local_col = i % TILE_SIZE;
        int global_row = tile_row * TILE_SIZE + local_row;
        int global_col = tile_col * TILE_SIZE + local_col;
        
        if (global_row < output_rows && global_col < output_cols) {
            int global_idx = global_row * output_cols + global_col;
            output[global_idx] = smem->reconstructed[i];
        }
    }
}

/**
 * Host function to launch reconstruction
 */
void launch_predictor_reconstruct(
    const int8_t* d_residuals,
    const uint8_t* d_predictor_modes,
    int8_t* d_output,
    int num_tiles,
    int tile_rows,
    int tile_cols,
    int output_rows,
    int output_cols,
    cudaStream_t stream
) {
    int threads = TILE_THREADS;
    int blocks = num_tiles;
    size_t smem_size = sizeof(TileSharedMem);
    
    reconstruct_tile_kernel<<<blocks, threads, smem_size, stream>>>(
        d_residuals,
        d_predictor_modes,
        d_output,
        num_tiles,
        tile_rows,
        tile_cols,
        output_rows,
        output_cols
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace wcodec

