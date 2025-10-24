/**
 * @file format.h
 * @brief Simple binary format for compressed weights
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace codec {

// Simple format designed for GPU parsing
// No complex metadata, just what's needed for decode

/**
 * File header
 */
struct Header {
    uint32_t magic;           // 'WCDC' = 0x43444357
    uint16_t version;         // 1
    uint16_t tile_size;       // 16
    uint32_t num_tiles_row;
    uint32_t num_tiles_col;
    uint32_t output_rows;
    uint32_t output_cols;
};

/**
 * Per-tile metadata
 */
struct TileMetadata {
    uint8_t predictor_mode;   // 0=LEFT, 1=TOP, 2=AVG, 3=PLANAR
    uint32_t data_offset;     // Offset in compressed stream
    uint32_t data_size;       // Size of compressed data
    uint32_t freq_table[256]; // rANS frequency table
};

/**
 * Complete compressed layer
 * 
 * Layout:
 *   [Header]
 *   [TileMetadata] x num_tiles
 *   [Compressed data for all tiles]
 */

// Constants
constexpr uint32_t MAGIC = 0x43444357;  // 'WCDC'
constexpr uint16_t VERSION = 1;
constexpr uint16_t DEFAULT_TILE_SIZE = 16;
constexpr uint32_t FREQ_SCALE = 4096;  // rANS frequency scale

// Predictor modes
enum PredictorMode : uint8_t {
    PRED_LEFT = 0,
    PRED_TOP = 1,
    PRED_AVG = 2,
    PRED_PLANAR = 3
};

} // namespace codec

