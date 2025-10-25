/**
 * @file gpu_decoder_full.cpp
 * @brief Full GPU decoder implementation
 */

#include "wcodec/gpu_decoder.h"
#include "wcodec/encoder_gpu.h"
#include <chrono>
#include <stdexcept>
#include <cstring>

#ifdef WCODEC_CUDA_ENABLED
#include <cuda_runtime.h>

// Forward declare CUDA functions
namespace wcodec {
namespace cuda {
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
    );
}
}
#endif

namespace wcodec {

// Full GPU decode implementation
class GPUDecoderFull {
public:
    explicit GPUDecoderFull(size_t tile_size) : tile_size_(tile_size) {
#ifdef WCODEC_CUDA_ENABLED
        // Check CUDA
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        gpu_available_ = (device_count > 0);
        
        if (gpu_available_) {
            cudaSetDevice(0);
        }
#else
        gpu_available_ = false;
#endif
    }
    
    ~GPUDecoderFull() {
#ifdef WCODEC_CUDA_ENABLED
        if (d_compressed_) cudaFree(d_compressed_);
        if (d_freq_tables_) cudaFree(d_freq_tables_);
        if (d_tile_offsets_) cudaFree(d_tile_offsets_);
        if (d_tile_sizes_) cudaFree(d_tile_sizes_);
        if (d_predictor_modes_) cudaFree(d_predictor_modes_);
        if (d_output_) cudaFree(d_output_);
#endif
    }
    
    GPUDecodeStats decodeLayer(
        const uint8_t* compressed,
        size_t compressed_size,
        size_t rows,
        size_t cols,
        int8_t* output
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        
        GPUDecodeStats stats;
        stats.bytes_decoded = rows * cols;
        
        if (!gpu_available_) {
            throw std::runtime_error("GPU not available");
        }
        
#ifdef WCODEC_CUDA_ENABLED
        // Parse GPU-friendly format
        GPUEncodedLayer layer;
        if (!EncoderGPU::deserialize(compressed, compressed_size, layer)) {
            throw std::runtime_error("Failed to parse compressed data");
        }
        
        int num_tiles = layer.tiles.size();
        int num_tiles_row = layer.num_tiles_row;
        int num_tiles_col = layer.num_tiles_col;
        int tile_size = layer.tile_size;
        
        // Allocate GPU memory
        size_t compressed_data_size = layer.compressed_data.size();
        size_t freq_tables_size = num_tiles * 256 * sizeof(uint32_t);
        size_t offsets_size = num_tiles * sizeof(uint32_t);
        size_t output_size = rows * cols;
        
        allocateGPUMemory(compressed_data_size, freq_tables_size, offsets_size, output_size);
        
        auto transfer_start = std::chrono::high_resolution_clock::now();
        
        // Prepare host data
        std::vector<uint32_t> h_freq_tables(num_tiles * 256);
        std::vector<uint32_t> h_tile_offsets(num_tiles);
        std::vector<uint32_t> h_tile_sizes(num_tiles);
        std::vector<uint8_t> h_predictor_modes(num_tiles);
        
        for (int i = 0; i < num_tiles; ++i) {
            memcpy(&h_freq_tables[i * 256], layer.tiles[i].freq_table, 256 * sizeof(uint32_t));
            h_tile_offsets[i] = layer.tiles[i].compressed_offset;
            h_tile_sizes[i] = layer.tiles[i].compressed_size;
            h_predictor_modes[i] = layer.tiles[i].predictor_mode;
        }
        
        // Transfer to GPU
        cudaMemcpy(d_compressed_, layer.compressed_data.data(), compressed_data_size, 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_freq_tables_, h_freq_tables.data(), freq_tables_size, 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_tile_offsets_, h_tile_offsets.data(), offsets_size, 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_tile_sizes_, h_tile_sizes.data(), offsets_size, 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_predictor_modes_, h_predictor_modes.data(), num_tiles, 
                   cudaMemcpyHostToDevice);
        
        auto transfer_end = std::chrono::high_resolution_clock::now();
        stats.transfer_time_ms = std::chrono::duration<double, std::milli>(
            transfer_end - transfer_start).count();
        
        // Launch GPU kernels
        auto decode_start = std::chrono::high_resolution_clock::now();
        
        cuda::launch_rans_decode_full(
            d_compressed_,
            d_freq_tables_,
            d_tile_offsets_,
            d_tile_sizes_,
            d_predictor_modes_,
            d_output_,
            num_tiles,
            tile_size,
            rows,
            cols,
            num_tiles_col,
            0  // Default stream
        );
        
        cudaDeviceSynchronize();
        
        auto decode_end = std::chrono::high_resolution_clock::now();
        stats.decode_time_ms = std::chrono::duration<double, std::milli>(
            decode_end - decode_start).count();
        
        // Copy result back
        cudaMemcpy(output, d_output_, output_size, cudaMemcpyDeviceToHost);
        
        auto end = std::chrono::high_resolution_clock::now();
        stats.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats.throughput_mbps = (output_size / (1024.0 * 1024.0)) / (stats.total_time_ms / 1000.0);
        
        return stats;
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
private:
    size_t tile_size_;
    bool gpu_available_ = false;
    
#ifdef WCODEC_CUDA_ENABLED
    uint8_t* d_compressed_ = nullptr;
    uint32_t* d_freq_tables_ = nullptr;
    uint32_t* d_tile_offsets_ = nullptr;
    uint32_t* d_tile_sizes_ = nullptr;
    uint8_t* d_predictor_modes_ = nullptr;
    int8_t* d_output_ = nullptr;
    
    size_t allocated_compressed_ = 0;
    size_t allocated_freq_ = 0;
    size_t allocated_offsets_ = 0;
    size_t allocated_output_ = 0;
    
    void allocateGPUMemory(size_t compressed_size, size_t freq_size, 
                          size_t offsets_size, size_t output_size) {
        if (compressed_size > allocated_compressed_) {
            if (d_compressed_) cudaFree(d_compressed_);
            cudaMalloc(&d_compressed_, compressed_size);
            allocated_compressed_ = compressed_size;
        }
        
        if (freq_size > allocated_freq_) {
            if (d_freq_tables_) cudaFree(d_freq_tables_);
            cudaMalloc(&d_freq_tables_, freq_size);
            allocated_freq_ = freq_size;
        }
        
        if (offsets_size > allocated_offsets_) {
            if (d_tile_offsets_) cudaFree(d_tile_offsets_);
            if (d_tile_sizes_) cudaFree(d_tile_sizes_);
            if (d_predictor_modes_) cudaFree(d_predictor_modes_);
            
            cudaMalloc(&d_tile_offsets_, offsets_size);
            cudaMalloc(&d_tile_sizes_, offsets_size);
            cudaMalloc(&d_predictor_modes_, offsets_size / 4);  // 1 byte per tile
            allocated_offsets_ = offsets_size;
        }
        
        if (output_size > allocated_output_) {
            if (d_output_) cudaFree(d_output_);
            cudaMalloc(&d_output_, output_size);
            allocated_output_ = output_size;
        }
    }
#endif
};

} // namespace wcodec

