/**
 * @file gpu_decoder.h
 * @brief GPU-accelerated decoder API
 */

#pragma once

#include "types.h"
#include <vector>
#include <memory>

namespace wcodec {

/**
 * GPU decoder configuration
 */
struct GPUDecoderConfig {
    int device_id = 0;              // CUDA device ID
    int num_streams = 4;            // Number of CUDA streams
    size_t pinned_buffer_size = 256 * 1024 * 1024;  // 256MB pinned memory
    bool fallback_to_cpu = true;    // Fallback to CPU if GPU unavailable
};

/**
 * GPU decoder stats
 */
struct GPUDecodeStats {
    double total_time_ms = 0.0;
    double transfer_time_ms = 0.0;
    double decode_time_ms = 0.0;
    double reconstruct_time_ms = 0.0;
    size_t bytes_decoded = 0;
    double throughput_mbps = 0.0;
};

/**
 * GPU Decoder class
 * 
 * Provides GPU-accelerated decoding of compressed weight layers.
 * Automatically falls back to CPU decoder if CUDA is unavailable.
 */
class GPUDecoder {
public:
    /**
     * Constructor
     * @param config GPU decoder configuration
     */
    explicit GPUDecoder(const GPUDecoderConfig& config = GPUDecoderConfig());
    
    /**
     * Destructor
     */
    ~GPUDecoder();
    
    // Delete copy constructor and assignment
    GPUDecoder(const GPUDecoder&) = delete;
    GPUDecoder& operator=(const GPUDecoder&) = delete;
    
    /**
     * Check if GPU decoding is available
     */
    bool isGPUAvailable() const;
    
    /**
     * Decode layer to GPU memory
     * 
     * @param compressed Compressed layer data
     * @param rows Output rows
     * @param cols Output columns
     * @param d_output Device pointer to output buffer (must be pre-allocated)
     * @return Decode statistics
     */
    GPUDecodeStats decodeLayerToGPU(
        const uint8_t* compressed,
        size_t compressed_size,
        size_t rows,
        size_t cols,
        int8_t* d_output
    );
    
    /**
     * Decode layer to host memory (convenience function)
     * 
     * @param compressed Compressed layer data
     * @param rows Output rows
     * @param cols Output columns
     * @param output Host pointer to output buffer
     * @return Decode statistics
     */
    GPUDecodeStats decodeLayer(
        const uint8_t* compressed,
        size_t compressed_size,
        size_t rows,
        size_t cols,
        int8_t* output
    );
    
    /**
     * Set tile size (must match encoder)
     */
    void setTileSize(size_t tile_rows, size_t tile_cols);
    
    /**
     * Synchronize all GPU operations
     */
    void synchronize();
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Check if CUDA is available at runtime
 */
bool isCUDAAvailable();

/**
 * Get CUDA device properties
 */
struct CUDADeviceInfo {
    std::string name;
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multi_processor_count;
};

std::vector<CUDADeviceInfo> getCUDADevices();

} // namespace wcodec

