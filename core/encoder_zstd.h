/**
 * @file encoder_zstd.h
 * @brief Zstd encoder for layer compression
 */

#pragma once

#include "format_zstd.h"
#include <vector>
#include <cstdint>

namespace codec {

/**
 * @brief Zstd-based layer encoder
 */
class ZstdEncoder {
public:
    /**
     * @brief Constructor
     * @param compression_level Zstd compression level (1-22, default 9)
     *                         Higher = better compression but slower
     *                         9 = good balance (3.0x ratio, fast)
     */
    explicit ZstdEncoder(int compression_level = 9);
    
    /**
     * @brief Encode a layer using Zstd compression
     * @param data Input data (int8 quantized weights)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param output Output buffer (resized as needed)
     * @return Compression ratio
     */
    float encodeLayer(const int8_t* data, uint32_t rows, uint32_t cols,
                      std::vector<uint8_t>& output);

private:
    int compression_level_;
};

} // namespace codec

