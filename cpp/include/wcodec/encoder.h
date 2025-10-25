/**
 * @file encoder.h
 * @brief High-level encoder interface
 */

#pragma once

#include "types.h"
#include <vector>
#include <string>

namespace wcodec {

class Encoder {
public:
    explicit Encoder(const TileConfig& config = TileConfig());
    ~Encoder();

    /**
     * @brief Encode a single layer
     * @param data Layer weight data (int8, row-major)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param output Encoded output
     * @return Encoding statistics
     */
    EncodeStats encodeLayer(
        const int8_t* data,
        size_t rows,
        size_t cols,
        std::vector<uint8_t>& output
    );

    /**
     * @brief Set tile size
     */
    void setTileSize(size_t rows, size_t cols);

protected:
    TileConfig config_;

private:
    
    // Helper: encode single tile
    void encodeTile(
        const int8_t* tile,
        size_t rows,
        size_t cols,
        const int8_t* left_col,
        const int8_t* top_row,
        std::vector<uint8_t>& output,
        TileMetadata& metadata
    );
};

} // namespace wcodec

