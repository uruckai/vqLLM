/**
 * @file rans_decode_full.cu
 * @brief Complete GPU rANS decode + reconstruction
 */

#include "kernels.cuh"

namespace wcodec {
namespace cuda {

/**
 * Full decode kernel: rANS + reconstruction in one pass
 * Each threadblock handles one tile
 */
__global__ void decode_tile_full_kernel(
    const uint8_t* compressed_data,
    const uint32_t* freq_tables,      // [num_tiles * 256]
    const uint32_t* tile_offsets,
    const uint32_t* tile_sizes,
    const uint8_t* predictor_modes,
    int8_t* output,
    int num_tiles,
    int tile_size,
    int output_rows,
    int output_cols,
    int num_tiles_col
) {
    int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;
    
    int tid = threadIdx.x;
    
    // Shared memory for this tile
    extern __shared__ char shared_mem[];
    int8_t* s_decoded = (int8_t*)shared_mem;  // tile_size * tile_size
    
    // Get tile position
    int tile_row = tile_id / num_tiles_col;
    int tile_col = tile_id % num_tiles_col;
    
    // Get compressed data for this tile
    const uint8_t* tile_data = compressed_data + tile_offsets[tile_id];
    uint32_t data_size = tile_sizes[tile_id];
    
    // Get frequency table for this tile
    const uint32_t* freq_table = freq_tables + tile_id * 256;
    
    // Build cumulative frequency table in shared memory
    __shared__ uint32_t cumul_freq[257];
    
    if (tid < 257) {
        uint32_t sum = 0;
        for (int i = 0; i < tid && i < 256; i++) {
            sum += freq_table[i];
        }
        cumul_freq[tid] = sum;
    }
    __syncthreads();
    
    // Simple rANS decode (thread 0 does sequential decode for now)
    // TODO: Optimize with warp-level parallelism
    if (tid == 0) {
        // Read initial state from end of stream
        uint32_t state = 0;
        if (data_size >= 4) {
            for (int i = 0; i < 4; i++) {
                state = (state << 8) | tile_data[i];
            }
        }
        
        const uint8_t* stream_ptr = tile_data + 4;
        const uint8_t* stream_end = tile_data + data_size;
        
        // Decode all symbols
        for (int i = 0; i < tile_size * tile_size && stream_ptr < stream_end; i++) {
            // Find symbol
            uint32_t cf = state % 4096;  // Freq scale
            
            // Binary search
            int symbol = 0;
            int lo = 0, hi = 256;
            while (lo < hi - 1) {
                int mid = (lo + hi) / 2;
                if (cumul_freq[mid] <= cf) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            symbol = lo;
            
            // Store decoded symbol
            s_decoded[i] = static_cast<int8_t>(symbol);
            
            // Update state
            uint32_t freq = freq_table[symbol];
            uint32_t cumul = cumul_freq[symbol];
            
            if (freq > 0) {
                state = freq * (state / 4096) + (cf - cumul);
                
                // Renormalize
                while (state < (1u << 16) && stream_ptr < stream_end) {
                    state = (state << 8) | *stream_ptr++;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Reconstruct using predictor (still thread 0 for now, due to dependencies)
    uint8_t pred_mode = predictor_modes[tile_id];
    
    if (tid == 0) {
        for (int r = 0; r < tile_size; r++) {
            for (int c = 0; c < tile_size; c++) {
                int idx = r * tile_size + c;
                int8_t residual = s_decoded[idx];
                
                // Get prediction
                int8_t pred = 0;
                int8_t left = (c > 0) ? s_decoded[r * tile_size + (c-1)] : 0;
                int8_t top = (r > 0) ? s_decoded[(r-1) * tile_size + c] : 0;
                int8_t top_left = (r > 0 && c > 0) ? s_decoded[(r-1) * tile_size + (c-1)] : 0;
                
                // Apply predictor based on mode
                switch (pred_mode) {
                    case 0: // LEFT
                        pred = left;
                        break;
                    case 1: // TOP
                        pred = top;
                        break;
                    case 2: // AVG
                        pred = clamp_int8((int(left) + int(top)) / 2);
                        break;
                    case 3: // PLANAR
                        pred = clamp_int8(int(left) + int(top) - int(top_left));
                        break;
                }
                
                // Reconstruct
                s_decoded[idx] = clamp_int8(int(pred) + int(residual));
            }
        }
    }
    
    __syncthreads();
    
    // Write result to global memory (all threads cooperate)
    int global_start_row = tile_row * tile_size;
    int global_start_col = tile_col * tile_size;
    
    for (int i = tid; i < tile_size * tile_size; i += blockDim.x) {
        int local_r = i / tile_size;
        int local_c = i % tile_size;
        int global_r = global_start_row + local_r;
        int global_c = global_start_col + local_c;
        
        if (global_r < output_rows && global_c < output_cols) {
            output[global_r * output_cols + global_c] = s_decoded[i];
        }
    }
}

/**
 * Host function to launch full GPU decode
 */
void launch_rans_decode_full(
    const uint8_t* d_compressed,
    const uint32_t* d_freq_tables,
    const uint32_t* d_tile_offsets,
    const uint32_t* d_tile_sizes,
    const uint8_t* d_predictor_modes,
    int8_t* d_output,
    int num_tiles,
    int tile_size,
    int output_rows,
    int output_cols,
    int num_tiles_col,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = num_tiles;
    size_t smem_size = tile_size * tile_size * sizeof(int8_t);  // Decoded tile
    
    decode_tile_full_kernel<<<blocks, threads, smem_size, stream>>>(
        d_compressed,
        d_freq_tables,
        d_tile_offsets,
        d_tile_sizes,
        d_predictor_modes,
        d_output,
        num_tiles,
        tile_size,
        output_rows,
        output_cols,
        num_tiles_col
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace wcodec

