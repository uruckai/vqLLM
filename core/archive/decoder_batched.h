/**
 * @file decoder_batched.h
 * @brief Batched GPU decoder for layer-level decompression
 */

#pragma once
#include "format_batched.h"
#include <vector>
#include <cstdint>

namespace codec {

class BatchedGPUDecoder {
public:
    BatchedGPUDecoder();
    ~BatchedGPUDecoder();
    
    /**
     * Decode entire layer on GPU (all tiles in parallel)
     * 
     * @param compressed Compressed layer data
     * @param output Output buffer (must be pre-allocated)
     * @return Decode time in milliseconds
     */
    float decodeLayer(const std::vector<uint8_t>& compressed, int8_t* output);
    
    /**
     * Check if GPU is available
     */
    static bool isAvailable();
    
private:
    void* d_workspace_;       // GPU workspace (reused across calls)
    size_t workspace_size_;
};

} // namespace codec

