/**
 * @file decoder_gpu.cu
 * @brief GPU decoder CUDA kernel
 */

#include "format.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace codec {

/**
 * rANS symbol structure (matches host)
 */
struct RANSSymbol {
    uint16_t start;
    uint16_t freq;
};

/**
 * rANS decode on GPU (using shared frequency table)
 */
__device__ void ransDecodeDevice(
    const uint8_t* stream, size_t stream_size,
    const RANSSymbol* symbols, int8_t* output, size_t output_size)
{
    // Stream format: 4 bytes (size header) + rANS encoded data + 4 bytes (state at end)
    if (stream_size < 8) return;

    // Read output size
    uint32_t stored_size = (stream[0] << 0) | (stream[1] << 8) |
                          (stream[2] << 16) | (stream[3] << 24);
    if (stored_size != output_size) return;

    // Initialize rANS state from end of stream
    const uint8_t* state_ptr = stream + stream_size - 4;
    uint32_t state = (state_ptr[0] << 0) | (state_ptr[1] << 8) |
                     (state_ptr[2] << 16) | (state_ptr[3] << 24);
    
    const uint8_t* read_ptr = state_ptr - 1;  // Read backwards from state
    
    // Decode rANS data - need temporary buffer for differential encoded values
    uint8_t diff_data[65536];  // Temporary buffer for decoded differential values (256x256 max)
    size_t decode_count = min(stored_size, static_cast<uint32_t>(output_size));
    decode_count = min(decode_count, static_cast<size_t>(65536));
    
    // rANS decode loop
    for (size_t i = 0; i < decode_count; i++) {
        // Find symbol from cumulative frequency
        uint32_t cum_freq = state & 0xFFF;  // 12-bit scale
        
        uint8_t symbol = 0;
        for (int j = 0; j < 256; j++) {
            if (symbols[j].start <= cum_freq && 
                cum_freq < symbols[j].start + symbols[j].freq) {
                symbol = j;
                break;
            }
        }
        
        const RANSSymbol& s = symbols[symbol];
        
        // Update rANS state
        state = s.freq * (state >> 12) + (cum_freq - s.start);
        
        // Renormalize
        while (state < (1u << 23)) {
            if (read_ptr >= stream + 4) {
                state = (state << 8) | (*read_ptr);
                read_ptr--;
            } else {
                break;
            }
        }
        
        diff_data[i] = symbol;
    }
    
    // Apply differential decoding in-place
    int32_t prev = 0;
    for (size_t i = 0; i < decode_count; i++) {
        // Decode: reverse of (diff + 128) & 0xFF
        int32_t diff = static_cast<int32_t>(diff_data[i]) - 128;
        int32_t current = prev + diff;
        output[i] = static_cast<int8_t>(current);
        prev = current;
    }
}

/**
 * Reconstruct tile from residual using predictor
 */
__device__ void reconstructDevice(
    int8_t* tile, uint32_t rows, uint32_t cols,
    const int8_t* left, const int8_t* top, PredictorMode mode)
{
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            int idx = r * cols + c;
            int8_t pred = 0;
            
            switch (mode) {
                case PRED_LEFT:
                    pred = (c > 0) ? tile[idx - 1] : (left ? left[r] : 0);
                    break;
                    
                case PRED_TOP:
                    pred = (r > 0) ? tile[idx - cols] : (top ? top[c] : 0);
                    break;
                    
                case PRED_AVG: {
                    int8_t left_val = (c > 0) ? tile[idx - 1] : (left ? left[r] : 0);
                    int8_t top_val = (r > 0) ? tile[idx - cols] : (top ? top[c] : 0);
                    pred = (left_val + top_val) / 2;
                    break;
                }
                
                case PRED_PLANAR: {
                    int8_t left_val = (c > 0) ? tile[idx - 1] : (left ? left[r] : 0);
                    int8_t top_val = (r > 0) ? tile[idx - cols] : (top ? top[c] : 0);
                    int8_t tl_val = 0;
                    if (r > 0 && c > 0) {
                        tl_val = tile[idx - cols - 1];
                    }
                    pred = left_val + top_val - tl_val;
                    break;
                }
            }
            
            // Residual -> Original
            tile[idx] = tile[idx] + pred;
        }
    }
}

/**
 * Main decode kernel - one thread block per tile
 */
__global__ void decodeKernel(
    const uint8_t* compressed,
    const Header* header,
    const TileMetadata* tile_metadata,
    int8_t* output,
    const uint8_t* global_freq_table)  // Add global freq table parameter
{
    // Each block handles one tile
    uint32_t tile_idx = blockIdx.x;
    uint32_t num_tiles = header->num_tiles_row * header->num_tiles_col;
    
    if (tile_idx >= num_tiles) return;
    
    // Calculate tile position
    uint32_t ty = tile_idx / header->num_tiles_col;
    uint32_t tx = tile_idx % header->num_tiles_col;
    
    uint32_t row_start = ty * header->tile_size;
    uint32_t col_start = tx * header->tile_size;
    uint32_t tile_rows = min(header->tile_size, header->output_rows - row_start);
    uint32_t tile_cols = min(header->tile_size, header->output_cols - col_start);
    
    // Only use shared memory for frequency table (not tile data - too large for 256x256)
    extern __shared__ RANSSymbol symbols[];
    
    // Thread 0 loads global frequency table into shared memory
    if (threadIdx.x == 0) {
        uint32_t cumul = 0;
        for (int i = 0; i < 256; i++) {
            uint16_t freq = (global_freq_table[i*2 + 0] << 0) | 
                           (global_freq_table[i*2 + 1] << 8);
            symbols[i].start = cumul;
            symbols[i].freq = freq;
            cumul += freq;
        }
    }
    __syncthreads();
    
    // Thread 0 does rANS decode + reconstruction
    if (threadIdx.x == 0) {
        const TileMetadata& meta = tile_metadata[tile_idx];
        const uint8_t* tile_stream = compressed + meta.data_offset;
        
        // Calculate output position for this tile
        int8_t* tile_output = output + tile_idx * header->tile_size * header->tile_size;
        
        // Decode rANS directly to tile position
        ransDecodeDevice(tile_stream, meta.data_size, symbols, 
                        tile_output, tile_rows * tile_cols);
        
        // Reconstruct in-place
        const int8_t* left = nullptr;
        const int8_t* top = nullptr;
        reconstructDevice(tile_output, tile_rows, tile_cols, left, top,
                         static_cast<PredictorMode>(meta.predictor_mode));
    }
}

} // namespace codec

/**
 * Host launch function
 */
extern "C" void launchDecodeKernel(
    const uint8_t* d_compressed,
    const codec::Header* d_header,
    const codec::TileMetadata* d_metadata,
    int8_t* d_output,
    cudaStream_t stream)
{
    // Copy header to host to get dimensions
    codec::Header h_header;
    cudaMemcpy(&h_header, d_header, sizeof(codec::Header), cudaMemcpyDeviceToHost);
    
    uint32_t num_tiles = h_header.num_tiles_row * h_header.num_tiles_col;
    size_t metadata_size = num_tiles * sizeof(codec::TileMetadata);
    
    // Global frequency table is after header + metadata
    const uint8_t* d_global_freq_table = d_compressed + sizeof(codec::Header) + metadata_size;
    
    // Shared memory: tile data + frequency table symbols
    // Only need shared memory for RANSSymbol table (2KB), not the full tile
    size_t shared_mem = 256 * sizeof(codec::RANSSymbol);
    
    // Launch one block per tile, 256 threads per block
    codec::decodeKernel<<<num_tiles, 256, shared_mem, stream>>>(
        d_compressed, d_header, d_metadata, d_output, d_global_freq_table
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

