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
 * rANS decode on GPU
 */
__device__ void ransDecodeDevice(
    const uint8_t* stream, size_t stream_size,
    const uint32_t* freqs, int8_t* output, size_t output_size)
{
    // Minimum size: 4 (size header) + 512 (freq table) + 4 (rANS state)
    if (stream_size < 520) return;

    // Read output size
    uint32_t stored_size = (stream[0] << 0) | (stream[1] << 8) |
                          (stream[2] << 16) | (stream[3] << 24);
    if (stored_size != output_size) return;

    // Read frequency table
    RANSSymbol symbols[256];
    uint32_t cumul = 0;
    const uint8_t* freq_ptr = stream + 4;
    
    for (int i = 0; i < 256; i++) {
        uint16_t freq = (freq_ptr[0] << 0) | (freq_ptr[1] << 8);
        symbols[i].start = cumul;
        symbols[i].freq = freq;
        cumul += freq;
        freq_ptr += 2;
    }

    // Initialize rANS state from end of stream
    const uint8_t* state_ptr = stream + stream_size - 4;
    uint32_t state = (state_ptr[0] << 0) | (state_ptr[1] << 8) |
                     (state_ptr[2] << 16) | (state_ptr[3] << 24);
    
    const uint8_t* read_ptr = state_ptr - 1;  // Start reading backwards from state
    
    // Decode differential data
    uint8_t diff_data[256];  // Buffer for decoded diffs
    size_t decode_count = min(output_size, static_cast<size_t>(256));
    
    for (size_t i = 0; i < decode_count; i++) {
        // Find symbol
        uint32_t cum_freq = state & 0xFFF;  // 12-bit mask
        
        uint8_t symbol = 0;
        for (int j = 0; j < 256; j++) {
            if (symbols[j].start <= cum_freq && 
                cum_freq < symbols[j].start + symbols[j].freq) {
                symbol = j;
                break;
            }
        }
        
        const RANSSymbol& s = symbols[symbol];
        
        // Update state
        state = s.freq * (state >> 12) + (cum_freq - s.start);
        
        // Renormalize
        while (state < (1u << 23)) {
            if (read_ptr >= stream + 516) {  // Don't read past freq table
                state = (state << 8) | (*read_ptr);
                read_ptr--;
            } else {
                break;
            }
        }
        
        diff_data[i] = symbol;
    }
    
    // Apply differential decoding
    int32_t prev = 0;
    for (size_t i = 0; i < decode_count; i++) {
        int32_t diff_temp = static_cast<int32_t>(diff_data[i]) - 128;
        int32_t diff = (diff_temp > 127) ? (diff_temp - 256) : diff_temp;
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
    int8_t* output)
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
    
    // Shared memory for tile
    extern __shared__ int8_t tile_data[];
    
    // Thread 0 does rANS decode
    if (threadIdx.x == 0) {
        const TileMetadata& meta = tile_metadata[tile_idx];
        const uint8_t* tile_stream = compressed + meta.data_offset;
        
        ransDecodeDevice(tile_stream, meta.data_size, meta.freq_table, 
                        tile_data, tile_rows * tile_cols);
    }
    __syncthreads();
    
    // Thread 0 does reconstruction
    if (threadIdx.x == 0) {
        const TileMetadata& meta = tile_metadata[tile_idx];
        
        // Context pointers (simplified - no boundary handling for now)
        const int8_t* left = nullptr;
        const int8_t* top = nullptr;
        
        reconstructDevice(tile_data, tile_rows, tile_cols, left, top,
                         static_cast<PredictorMode>(meta.predictor_mode));
    }
    __syncthreads();
    
    // All threads cooperate to write output
    for (uint32_t i = threadIdx.x; i < tile_rows * tile_cols; i += blockDim.x) {
        uint32_t r = i / tile_cols;
        uint32_t c = i % tile_cols;
        output[(row_start + r) * header->output_cols + col_start + c] = tile_data[i];
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
    size_t shared_mem = h_header.tile_size * h_header.tile_size * sizeof(int8_t);
    
    // Launch one block per tile, 256 threads per block
    codec::decodeKernel<<<num_tiles, 256, shared_mem, stream>>>(
        d_compressed, d_header, d_metadata, d_output
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

