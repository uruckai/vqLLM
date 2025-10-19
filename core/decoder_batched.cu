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
    
    // NO shared memory - use global memory directly to avoid 48KB limit
    // Each thread will write directly to global output
    
    // SIMPLIFIED: Single-threaded decode per tile to avoid shared memory
    // Only thread 0 does the work (not optimal but avoids 48KB limit)
    if (threadIdx.x != 0) return;
    
    // Allocate temporary buffer in global memory (slower but works)
    uint8_t* diff_buffer = new uint8_t[uncompressed_size];
    
    // Decode rANS sequentially
    for (int i = uncompressed_size - 1; i >= 0; i--) {
        // Find symbol by binary search
        uint32_t cumul = state & ((1 << 12) - 1);
        
        uint32_t sym = 0;
        for (int bit = 7; bit >= 0; --bit) {
            uint32_t test_sym = sym | (1 << bit);
            if (test_sym < 256 && d_rans_table[test_sym].start <= cumul) {
                sym = test_sym;
            }
        }
        
        const RANSSymbol& s = d_rans_table[sym];
        state = s.freq * (state >> 12) + (cumul - s.start);
        
        // Renormalize
        while (state < (1 << 23) && decode_pos >= 0) {
            state = (state << 8) | tile_compressed[decode_pos--];
        }
        
        diff_buffer[i] = sym;
    }
    
    // Apply inverse differential encoding (LEFT predictor)
    uint32_t tile_row = entry.row;
    uint32_t tile_col = entry.col;
    
    // First pixel
    int32_t val = static_cast<int32_t>(diff_buffer[0]) - 128;
    uint32_t out_row = tile_row * tile_size;
    uint32_t out_col = tile_col * tile_size;
    if (out_row < output_cols && out_col < output_cols) {
        d_output[out_row * output_cols + out_col] = static_cast<int8_t>(val);
    }
    
    // Remaining pixels
    for (uint32_t i = 1; i < uncompressed_size; i++) {
        uint32_t local_row = i / tile_size;
        uint32_t local_col = i % tile_size;
        
        // Get previous reconstructed value
        int32_t pred_val = static_cast<int32_t>(d_output[(tile_row * tile_size + local_row) * output_cols + 
                                                         (tile_col * tile_size + local_col - 1)]);
        
        int32_t residual = static_cast<int32_t>(diff_buffer[i]) - 128;
        val = pred_val + residual;
        
        out_row = tile_row * tile_size + local_row;
        out_col = tile_col * tile_size + local_col;
        
        if (out_row < output_cols && out_col < output_cols) {
            d_output[out_row * output_cols + out_col] = static_cast<int8_t>(val);
        }
    }
    
    delete[] diff_buffer;
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

