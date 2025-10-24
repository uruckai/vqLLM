/**
 * @file decoder_host.h
 * @brief GPU decoder host interface
 */

#pragma once

#include "format.h"
#include <vector>
#include <cstdint>

namespace codec {

/**
 * GPU decoder - host side
 */
class GPUDecoder {
public:
    GPUDecoder();
    ~GPUDecoder();
    
    /**
     * Decode compressed data on GPU
     * 
     * @param compressed Compressed input data
     * @param output Output buffer (must be pre-allocated)
     * @return Decode time in milliseconds
     */
    float decode(const std::vector<uint8_t>& compressed, int8_t* output);
    
    /**
     * Check if GPU is available
     */
    static bool isAvailable();
    
private:
    void* d_workspace_;  // GPU workspace
    size_t workspace_size_;
};

} // namespace codec

