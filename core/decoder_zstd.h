/**
 * @file decoder_zstd.h
 * @brief GPU-accelerated Zstd decoder using nvCOMP
 */

#pragma once

#include "format_zstd.h"
#include <cstdint>
#include <vector>

namespace codec {

/**
 * @brief GPU-accelerated Zstd decoder
 * 
 * Uses NVIDIA nvCOMP library for fast GPU decompression.
 * Falls back to CPU Zstd if nvCOMP is not available.
 */
class ZstdGPUDecoder {
public:
    ZstdGPUDecoder();
    ~ZstdGPUDecoder();
    
    /**
     * @brief Check if GPU decoder is available
     */
    static bool isAvailable();
    
    /**
     * @brief Decode a compressed layer to CPU memory
     * @param compressed_data Input compressed data (includes header)
     * @param compressed_size Size of compressed data
     * @param output Output buffer (allocated by caller)
     * @param rows Output parameter for rows
     * @param cols Output parameter for cols
     * @return true on success, false on failure
     */
    bool decodeLayer(const uint8_t* compressed_data, size_t compressed_size,
                     int8_t* output, uint32_t& rows, uint32_t& cols);
    
    /**
     * @brief Decode a compressed layer directly to GPU memory (no CPU copy)
     * @param compressed_data Input compressed data (host memory, includes header)
     * @param compressed_size Size of compressed data
     * @param rows Output parameter for rows
     * @param cols Output parameter for cols
     * @return GPU pointer to decompressed data (caller must cudaFree), or nullptr on failure
     */
    void* decodeLayerToGPU(const uint8_t* compressed_data, size_t compressed_size,
                           uint32_t& rows, uint32_t& cols);
    
    /**
     * @brief Get header info without decompressing
     */
    static bool parseHeader(const uint8_t* compressed_data, size_t compressed_size,
                            LayerHeaderZstd& header);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace codec

