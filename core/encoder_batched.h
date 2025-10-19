/**
 * @file encoder_batched.h
 * @brief Batched encoder for layer-level compression
 */

#pragma once
#include "format_batched.h"
#include "rans.h"
#include <vector>
#include <cstdint>

namespace codec {

class BatchedEncoder {
public:
    explicit BatchedEncoder(uint16_t tile_size = 256);
    ~BatchedEncoder() = default;
    
    /**
     * Encode entire layer at once
     * 
     * @param data Input int8 matrix (row-major)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param output Compressed layer data
     * @return Compression ratio (original/compressed)
     */
    float encodeLayer(const int8_t* data, uint32_t rows, uint32_t cols, 
                      std::vector<uint8_t>& output);
    
private:
    uint16_t tile_size_;
    
    // Encode single tile (used internally)
    void encodeTile(const int8_t* tile_data, uint32_t tile_rows, uint32_t tile_cols,
                   std::vector<uint8_t>& output);
    
    // Apply differential encoding
    void applyDifferentialEncoding(const int8_t* input, uint32_t size,
                                   std::vector<uint8_t>& output);
};

} // namespace codec

