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
    
    // Decode rANS sequentially (forward order, not backward!)
    const uint32_t RANS_SCALE = 1 << 12;  // 4096
    const uint32_t RANS_L = 1 << 23;      // 8388608
    
    for (uint32_t i = 0; i < uncompressed_size; i++) {
        // Find symbol using cumulative frequency (LINEAR SEARCH - correct!)
        uint32_t cum_freq = state & (RANS_SCALE - 1);
        
        uint8_t symbol = 0;
        for (int s = 0; s < 256; s++) {
            if (d_rans_table[s].start <= cum_freq && 
                cum_freq < d_rans_table[s].start + d_rans_table[s].freq) {
                symbol = s;
                break;
            }
        }
        
        // Update state
        const RANSSymbol& sym = d_rans_table[symbol];
        state = sym.freq * (state >> 12) + (cum_freq - sym.start);
        
        // Renormalize
        while (state < RANS_L && decode_pos >= 4) {
            state = (state << 8) | tile_compressed[decode_pos--];
        }
        
        diff_buffer[i] = symbol;
    }
    
    // Apply inverse differential encoding (LEFT predictor - simple!)
    uint32_t tile_row = entry.row;
    uint32_t tile_col = entry.col;
    
    // First pixel
    int32_t val = static_cast<int32_t>(diff_buffer[0]) - 128;
    uint32_t out_row = tile_row * tile_size;
    uint32_t out_col = tile_col * tile_size;
    if (out_row < output_cols && out_col < output_cols) {
        d_output[out_row * output_cols + out_col] = static_cast<int8_t>(val);
    }
    
    // Remaining pixels (simple LEFT predictor: use immediately previous pixel)
    for (uint32_t i = 1; i < uncompressed_size; i++) {
        uint32_t local_row = i / tile_size;
        uint32_t local_col = i % tile_size;
        
        // LEFT predictor: always use previous pixel (i-1)
        int32_t residual = static_cast<int32_t>(diff_buffer[i]) - 128;
        val = val + residual;  // val holds the previous reconstructed pixel
        
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

