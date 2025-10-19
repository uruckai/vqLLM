/**
 * @file decoder_batched.cu
 * @brief GPU kernel for parallel tile decompression
 */

#include "format_batched.h"
#include "rans.h"  // For RANSSymbol definition
#include <cuda_runtime.h>
#include <cstdio>

namespace codec {

/**
 * GPU kernel: Decode multiple tiles in parallel
 * 
 * Each CUDA block decodes one tile:
 * - Block 0 → Tile 0
 * - Block 1 → Tile 1
 * - etc.
 */
__global__ void decodeTilesBatched(
    const uint8_t* __restrict__ d_tile_data,
    const TileIndexEntry* __restrict__ d_tile_index,
    const RANSSymbol* __restrict__ d_rans_table,
    int8_t* __restrict__ d_output,
    uint32_t tile_size,
    uint32_t output_cols,
    uint32_t num_tiles
) {
    uint32_t tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;
    
    // Get tile info
    const TileIndexEntry& entry = d_tile_index[tile_idx];
    const uint8_t* tile_compressed = d_tile_data + entry.offset;
    uint32_t compressed_size = entry.compressed_size;
    
    // Read size header (first 4 bytes)
    uint32_t uncompressed_size = 
        tile_compressed[0] | 
        (tile_compressed[1] << 8) | 
        (tile_compressed[2] << 16) | 
        (tile_compressed[3] << 24);
    
    // rANS state starts at end of compressed data (last 4 bytes)
    uint32_t state = 
        tile_compressed[compressed_size - 4] |
        (tile_compressed[compressed_size - 3] << 8) |
        (tile_compressed[compressed_size - 2] << 16) |
        (tile_compressed[compressed_size - 1] << 24);
    
    // Decode pointer (just before state)
    int decode_pos = compressed_size - 5;
    
    // Shared memory for decoded differential data
    __shared__ uint8_t s_diff[256 * 256];
    
    // Decode rANS (each thread decodes a portion)
    uint32_t threads_per_block = blockDim.x;
    uint32_t tid = threadIdx.x;
    
    // Decode in parallel chunks
    for (uint32_t i = tid; i < uncompressed_size; i += threads_per_block) {
        // Find symbol by binary search on cumulative frequency
        uint32_t cumul = state & ((1 << 12) - 1);  // RANS_SCALE_BITS = 12
        
        // Binary search (using rans.h RANSSymbol with start/freq fields)
        uint32_t sym = 0;
        for (int bit = 7; bit >= 0; --bit) {
            uint32_t test_sym = sym | (1 << bit);
            if (test_sym < 256 && d_rans_table[test_sym].start <= cumul) {
                sym = test_sym;
            }
        }
        
        // Decode symbol (RANSSymbol from rans.h has start and freq)
        const RANSSymbol& s = d_rans_table[sym];
        state = s.freq * (state >> 12) + (cumul - s.start);
        
        // Renormalize
        while (state < (1 << 23) && decode_pos >= 0) {  // RANS_L
            state = (state << 8) | tile_compressed[decode_pos--];
        }
        
        // Store decoded differential (in reverse order)
        s_diff[uncompressed_size - 1 - i] = sym;
    }
    
    __syncthreads();
    
    // Apply inverse differential encoding
    // Thread 0 handles first pixel, then parallel for rest
    if (tid == 0) {
        int32_t val = static_cast<int32_t>(s_diff[0]) - 128;
        
        // Calculate output position
        uint32_t tile_row = entry.row;
        uint32_t tile_col = entry.col;
        uint32_t out_row = tile_row * tile_size;
        uint32_t out_col = tile_col * tile_size;
        
        if (out_row < output_cols && out_col < output_cols) {  // bounds check
            d_output[out_row * output_cols + out_col] = static_cast<int8_t>(val);
        }
    }
    
    __syncthreads();
    
    // Reconstruct remaining pixels in parallel (LEFT predictor)
    for (uint32_t i = 1 + tid; i < uncompressed_size; i += threads_per_block) {
        uint32_t local_row = i / tile_size;
        uint32_t local_col = i % tile_size;
        
        // Get prediction from left
        int32_t pred_val;
        if (local_col == 0 && local_row > 0) {
            // First column: predict from above (stored in shared mem)
            uint32_t pred_idx = (local_row - 1) * tile_size + local_col;
            int32_t diff = static_cast<int32_t>(s_diff[pred_idx]) - 128;
            pred_val = static_cast<int32_t>(static_cast<int8_t>(diff));
        } else {
            // Use left pixel
            uint32_t pred_idx = i - 1;
            int32_t diff = static_cast<int32_t>(s_diff[pred_idx]) - 128;
            pred_val = static_cast<int32_t>(static_cast<int8_t>(diff));
        }
        
        int32_t residual = static_cast<int32_t>(s_diff[i]) - 128;
        int32_t val = pred_val + residual;
        
        // Write to global output
        uint32_t tile_row = entry.row;
        uint32_t tile_col = entry.col;
        uint32_t out_row = tile_row * tile_size + local_row;
        uint32_t out_col = tile_col * tile_size + local_col;
        
        if (out_row < output_cols && out_col < output_cols) {
            d_output[out_row * output_cols + out_col] = static_cast<int8_t>(val);
        }
    }
}

// Kernel launcher
extern "C" void launchDecodeTilesBatched(
    const uint8_t* d_tile_data,
    const TileIndexEntry* d_tile_index,
    const RANSSymbol* d_rans_table,
    int8_t* d_output,
    uint32_t tile_size,
    uint32_t output_cols,
    uint32_t num_tiles,
    cudaStream_t stream
) {
    // Launch one block per tile for maximum parallelism
    dim3 grid(num_tiles);
    dim3 block(256);  // 256 threads per block
    
    decodeTilesBatched<<<grid, block, 0, stream>>>(
        d_tile_data,
        d_tile_index,
        d_rans_table,
        d_output,
        tile_size,
        output_cols,
        num_tiles
    );
}

} // namespace codec

