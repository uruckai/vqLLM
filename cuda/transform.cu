/**
 * @file transform.cu
 * @brief Inverse DCT/ADST transforms on GPU
 */

#include "kernels.cuh"

namespace wcodec {
namespace cuda {

/**
 * Inverse 8-point 1D DCT-II (simplified integer version)
 */
__device__ void idct8(const int16_t* in, int16_t* out) {
    // Simplified butterfly structure for 8-point IDCT
    // Full implementation would use optimized fixed-point constants
    
    const int C1 = 64;  // cos(π/16) * 64
    const int C2 = 64;  // cos(π/8) * 64
    const int C3 = 54;  // cos(3π/16) * 64
    
    // Stage 1: Even part
    int t0 = in[0] + in[4];
    int t1 = in[0] - in[4];
    int t2 = (C2 * in[2]) >> 6;
    int t3 = (C2 * in[6]) >> 6;
    
    // Stage 2: Odd part
    int t4 = (C1 * in[1]) >> 6;
    int t5 = (C3 * in[3]) >> 6;
    int t6 = (C3 * in[5]) >> 6;
    int t7 = (C1 * in[7]) >> 6;
    
    // Combine (simplified)
    out[0] = clamp_int8((t0 + t2 + t4 + t5) >> 6);
    out[1] = clamp_int8((t1 + t3 + t4 + t6) >> 6);
    out[2] = clamp_int8((t1 - t3 + t5 + t6) >> 6);
    out[3] = clamp_int8((t0 - t2 + t5 + t7) >> 6);
    out[4] = clamp_int8((t0 - t2 - t5 - t7) >> 6);
    out[5] = clamp_int8((t1 - t3 - t5 - t6) >> 6);
    out[6] = clamp_int8((t1 + t3 - t4 - t6) >> 6);
    out[7] = clamp_int8((t0 + t2 - t4 - t5) >> 6);
}

/**
 * 2D Inverse DCT for 8x8 block
 */
__device__ void idct8x8(const int16_t* in, int8_t* out) {
    int16_t temp[64];
    
    // Horizontal passes
    for (int i = 0; i < 8; i++) {
        idct8(in + i * 8, temp + i * 8);
    }
    
    // Vertical passes
    int16_t col_in[8];
    int16_t col_out[8];
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            col_in[i] = temp[i * 8 + j];
        }
        idct8(col_in, col_out);
        for (int i = 0; i < 8; i++) {
            out[i * 8 + j] = static_cast<int8_t>(col_out[i]);
        }
    }
}

/**
 * Apply inverse transforms to coefficient blocks
 * Each threadblock processes multiple 8x8 blocks
 */
__global__ void inverse_transform_kernel(
    const int16_t* coeffs,
    const uint8_t* transform_types,  // 0=none, 1=DCT, 2=ADST
    int8_t* output,
    int num_blocks
) {
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= num_blocks) return;
    
    const int16_t* block_in = coeffs + block_id * 64;
    int8_t* block_out = output + block_id * 64;
    uint8_t transform_type = transform_types[block_id];
    
    if (transform_type == 0) {
        // No transform - just copy/cast
        for (int i = 0; i < 64; i++) {
            block_out[i] = clamp_int8(block_in[i]);
        }
    } else if (transform_type == 1) {
        // DCT-II
        idct8x8(block_in, block_out);
    } else {
        // ADST (simplified - use DCT for now)
        idct8x8(block_in, block_out);
    }
}

/**
 * Host function to launch inverse transform
 */
void launch_inverse_transform(
    const int16_t* d_coeffs,
    const uint8_t* d_transform_types,
    int8_t* d_output,
    int num_blocks,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;
    
    inverse_transform_kernel<<<blocks, threads, 0, stream>>>(
        d_coeffs,
        d_transform_types,
        d_output,
        num_blocks
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace wcodec

