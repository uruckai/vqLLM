/**
 * @file encoder_gpu.h
 * @brief GPU-friendly encoder that outputs parseable metadata
 */

#pragma once

#include "types.h"
#include "encoder.h"
#include <vector>
#include <cstdint>

namespace wcodec {

/**
 * GPU-friendly tile data
 */
struct GPUTileData {
    uint8_t predictor_mode;
    uint32_t freq_table[256];  // Frequency table for rANS
    uint32_t compressed_offset;  // Offset in compressed stream
    uint32_t compressed_size;    // Size of compressed data
};

/**
 * GPU-friendly encoded output
 */
struct GPUEncodedLayer {
    uint32_t num_tiles_row;
    uint32_t num_tiles_col;
    uint32_t tile_size;
    std::vector<GPUTileData> tiles;
    std::vector<uint8_t> compressed_data;  // All tile streams concatenated
};

/**
 * Encoder that outputs GPU-parseable format
 */
class EncoderGPU : public Encoder {
public:
    explicit EncoderGPU(const TileConfig& config = TileConfig());
    
    /**
     * Encode layer with GPU-friendly output
     */
    EncodeStats encodeLayerGPU(
        const int8_t* data,
        size_t rows,
        size_t cols,
        GPUEncodedLayer& output
    );
    
    /**
     * Serialize GPU-friendly format to bytes (for storage/transfer)
     */
    static void serialize(const GPUEncodedLayer& layer, std::vector<uint8_t>& output);
    
    /**
     * Deserialize from bytes
     */
    static bool deserialize(const uint8_t* data, size_t size, GPUEncodedLayer& layer);
};

} // namespace wcodec

