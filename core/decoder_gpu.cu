/**
 * @file decoder_gpu.cu
 * @brief GPU decoder CUDA kernel
 */

#include "format.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace codec {

/**
 * Simple differential decode on GPU (per-tile)
 */
__device__ void ransDecodeDevice(
    const uint8_t* stream, size_t stream_size,
    const uint32_t* freqs, int8_t* output, size_t output_size)
{
    // Read size header (4 bytes)
    if (stream_size < 4) return;

    uint32_t data_size = (stream[0] << 0) | (stream[1] << 8) |
                        (stream[2] << 16) | (stream[3] << 24);

    // Read differential data
    size_t read_size = min(data_size, static_cast<uint32_t>(output_size));
    int8_t prev = 0;

    for (size_t i = 0; i < read_size && (i + 4) < stream_size; i++) {
        uint8_t diff_byte = stream[i + 4];
        int8_t diff = static_cast<int8_t>(diff_byte - 128);  // Uncenter
        int8_t current = prev + diff;
        output[i] = current;
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

