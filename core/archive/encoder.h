/**
 * @file encoder.h
 * @brief Simple CPU encoder
 */

#pragma once

#include "format.h"
#include <vector>
#include <cstdint>

namespace codec {

/**
 * Simple encoder for INT8 weight matrices
 */
class Encoder {
public:
    explicit Encoder(uint16_t tile_size = DEFAULT_TILE_SIZE);
    
    /**
     * Encode INT8 weight matrix
     * 
     * @param data Input data (row-major, INT8)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param output Compressed output
     * @return Compression ratio
     */
    float encode(const int8_t* data, uint32_t rows, uint32_t cols, 
                 std::vector<uint8_t>& output);
    
private:
    uint16_t tile_size_;
    
    // Helper functions
    void encodeTile(const int8_t* tile, uint32_t tile_rows, uint32_t tile_cols,
                   const int8_t* left, const int8_t* top,
                   std::vector<uint8_t>& output, TileMetadata& metadata);
    
    PredictorMode selectPredictor(const int8_t* tile, uint32_t rows, uint32_t cols,
                                  const int8_t* left, const int8_t* top);
    
    void predict(const int8_t* tile, int8_t* residual, uint32_t rows, uint32_t cols,
                const int8_t* left, const int8_t* top, PredictorMode mode);
    
    void buildFrequencyTable(const int8_t* data, size_t size, uint32_t* freqs);
    void normalizeFrequencies(uint32_t* freqs, size_t num_symbols, uint32_t scale);
    
    void ransEncode(const int8_t* data, size_t size, const uint32_t* freqs,
                   std::vector<uint8_t>& output);
};

} // namespace codec

