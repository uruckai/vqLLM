/**
 * @file gpu_decode_direct.h
 * @brief Direct GPU decode from encoded layer format
 */

#pragma once

#include "types.h"
#include <vector>
#include <cstdint>

namespace wcodec {

/**
 * Decode layer directly on GPU
 * 
 * Takes the output from CPU encoder and decodes on GPU
 * Simpler than full container format parsing
 */
class GPUDecoder Direct {
public:
    explicit GPUDecoderDirect(size_t tile_size = 16);
    ~GPUDecoderDirect();
    
    /**
     * Decode layer on GPU
     * 
     * @param compressed Compressed data from encoder
     * @param rows Output rows
     * @param cols Output columns
     * @param output Output buffer (host memory)
     * @return Decode statistics
     */
    DecodeStats decodeLayer(
        const uint8_t* compressed,
        size_t compressed_size,
        size_t rows,
        size_t cols,
        int8_t* output
    );
    
    /**
     * Check if GPU is available
     */
    bool isGPUAvailable() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace wcodec

