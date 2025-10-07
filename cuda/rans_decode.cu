/**
 * @file rans_decode.cu
 * @brief Parallel rANS decoder for GPU
 */

#include "kernels.cuh"
#include <cstdio>

namespace wcodec {
namespace cuda {

/**
 * Initialize frequency table in shared memory
 * One warp handles this cooperatively
 */
__device__ void init_freq_table(
    const uint32_t* global_freq,
    TileSharedMem* smem,
    int tid
) {
    // First warp loads frequency table
    if (tid < 256) {
        smem->freq_table[tid] = global_freq[tid];
    }
    
    // Build cumulative frequency table (parallel prefix sum)
    __syncthreads();
    
    if (tid < 257) {
        uint32_t sum = 0;
        for (int i = 0; i < tid; i++) {
            sum += smem->freq_table[i];
        }
        smem->cumul_freq[tid] = sum;
    }
    
    __syncthreads();
}

/**
 * rANS decode single symbol
 */
__device__ uint8_t rans_decode_symbol(
    uint32_t& state,
    const uint8_t*& stream_ptr,
    const TileSharedMem* smem
) {
    // Find symbol from state
    uint32_t freq_scaled = state & (RANS_L - 1);
    
    // Binary search in cumulative frequency table
    int lo = 0, hi = 256;
    while (lo < hi - 1) {
        int mid = (lo + hi) / 2;
        if (smem->cumul_freq[mid] <= freq_scaled) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    uint8_t symbol = lo;
    
    // Update state
    uint32_t freq = smem->freq_table[symbol];
    uint32_t cumul = smem->cumul_freq[symbol];
    
    state = freq * (state >> RANS_PRECISION) + freq_scaled - cumul;
    
    // Renormalize
    while (state < RANS_L) {
        state = (state << 8) | *stream_ptr++;
    }
    
    return symbol;
}

/**
 * Decode tile using parallel rANS
 * Each threadblock = 1 tile
 */
__global__ void decode_tile_kernel(
    const uint8_t* compressed_data,
    const uint32_t* freq_tables,
    const size_t* tile_offsets,
    const size_t* tile_sizes,
    int8_t* residuals_out,
    int num_tiles,
    int tile_rows,
    int tile_cols
) {
    int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;
    
    int tid = threadIdx.x;
    
    // Allocate shared memory
    extern __shared__ char shared_mem[];
    TileSharedMem* smem = reinterpret_cast<TileSharedMem*>(shared_mem);
    
    // Load frequency table for this tile
    const uint32_t* tile_freq = freq_tables + tile_id * 256;
    init_freq_table(tile_freq, smem, tid);
    
    // Initialize rANS state and stream pointer
    const uint8_t* stream = compressed_data + tile_offsets[tile_id];
    size_t stream_size = tile_sizes[tile_id];
    
    // Each thread decodes a portion of the tile
    // Simplified: sequential decode for now (TODO: warp-parallel)
    if (tid == 0) {
        // Read initial state (last 4 bytes)
        uint32_t state = 0;
        for (int i = 0; i < 4; i++) {
            state = (state << 8) | stream[stream_size - 4 + i];
        }
        
        const uint8_t* stream_ptr = stream + stream_size - 5;
        
        // Decode all symbols in reverse order
        for (int i = TILE_SIZE * TILE_SIZE - 1; i >= 0; i--) {
            uint8_t symbol = rans_decode_symbol(state, stream_ptr, smem);
            smem->residuals[i] = static_cast<int8_t>(symbol);
        }
    }
    
    __syncthreads();
    
    // Copy residuals to global memory (all threads cooperate)
    int tile_row = tile_id / tile_cols;
    int tile_col = tile_id % tile_cols;
    int global_offset = (tile_row * TILE_SIZE * tile_cols * TILE_SIZE) + 
                        (tile_col * TILE_SIZE);
    
    for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
        int local_row = i / TILE_SIZE;
        int local_col = i % TILE_SIZE;
        int global_idx = global_offset + local_row * tile_cols * TILE_SIZE + local_col;
        residuals_out[global_idx] = smem->residuals[i];
    }
}

/**
 * Host function to launch rANS decode
 */
void launch_rans_decode(
    const uint8_t* d_compressed,
    const uint32_t* d_freq_tables,
    const size_t* d_tile_offsets,
    const size_t* d_tile_sizes,
    int8_t* d_residuals,
    int num_tiles,
    int tile_rows,
    int tile_cols,
    cudaStream_t stream
) {
    int threads = TILE_THREADS;
    int blocks = num_tiles;
    size_t smem_size = sizeof(TileSharedMem);
    
    decode_tile_kernel<<<blocks, threads, smem_size, stream>>>(
        d_compressed,
        d_freq_tables,
        d_tile_offsets,
        d_tile_sizes,
        d_residuals,
        num_tiles,
        tile_rows,
        tile_cols
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace wcodec

