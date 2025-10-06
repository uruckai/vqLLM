/**
 * @file decoder.h
 * @brief High-level decoder interface
 */

#pragma once

#include "types.h"
#include <vector>

namespace wcodec {

class Decoder {
public:
    explicit Decoder(const TileConfig& config = TileConfig());
    ~Decoder();

    /**
     * @brief Decode a single layer
     * @param input Encoded input data
     * @param size Size of input data
     * @param rows Expected number of rows
     * @param cols Expected number of columns
     * @param output Decoded output (caller allocates)
     * @return Decoding statistics
     */
    DecodeStats decodeLayer(
        const uint8_t* input,
        size_t size,
        size_t rows,
        size_t cols,
        int8_t* output
    );

    /**
     * @brief Set tile size (must match encoder)
     */
    void setTileSize(size_t rows, size_t cols);

private:
    TileConfig config_;
    
    // Helper: decode single tile
    void decodeTile(
        const uint8_t* input,
        size_t input_size,
        const TileMetadata& metadata,
        int8_t* tile,
        size_t rows,
        size_t cols,
        const int8_t* left_col,
        const int8_t* top_row
    );
};

} // namespace wcodec

