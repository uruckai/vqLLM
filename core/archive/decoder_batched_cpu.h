/**
 * @file decoder_batched_cpu.h
 * @brief CPU decoder for batched format (fallback, guaranteed to work)
 */

#pragma once
#include "format_batched.h"
#include <vector>
#include <cstdint>

namespace codec {

class BatchedCPUDecoder {
public:
    BatchedCPUDecoder() = default;
    ~BatchedCPUDecoder() = default;
    
    /**
     * Decode entire layer on CPU
     * 
     * @param compressed Compressed layer data
     * @param output Output buffer (must be pre-allocated)
     * @return Decode time in milliseconds
     */
    float decodeLayer(const std::vector<uint8_t>& compressed, int8_t* output);
    
    /**
     * Always available (CPU only)
     */
    static bool isAvailable() { return true; }
};

} // namespace codec

