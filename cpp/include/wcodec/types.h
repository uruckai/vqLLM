/**
 * @file types.h
 * @brief Common types and constants for Weight Codec
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace wcodec {

// Predictor modes
enum class PredictorMode : uint8_t {
    NONE = 0,
    LEFT = 1,
    TOP = 2,
    AVG = 3,
    PLANAR = 4
};

// Transform types
enum class TransformType : uint8_t {
    NONE = 0,
    DCT = 1,
    ADST = 2
};

// Data types
enum class DType : uint8_t {
    INT8 = 1,
    INT4 = 2,
    FP16 = 3,
    FP8 = 4
};

// Tile structure
struct TileConfig {
    size_t tile_rows = 16;
    size_t tile_cols = 16;
};

// Layer metadata
struct LayerInfo {
    std::string name;
    size_t rows;
    size_t cols;
    DType dtype;
    size_t num_tiles_row;
    size_t num_tiles_col;
};

// Tile metadata
struct TileMetadata {
    PredictorMode predictor;
    uint8_t transform_map[4];  // 2 bits per 8x8 sub-block, packed
    size_t stream_offset;
    uint32_t stream_length;
};

// Encoding statistics
struct EncodeStats {
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
    double compression_ratio = 0.0;
    double encode_time_ms = 0.0;
    size_t num_tiles = 0;
};

// Decoding statistics
struct DecodeStats {
    size_t compressed_bytes = 0;
    size_t decompressed_bytes = 0;
    double decode_time_ms = 0.0;
    size_t num_tiles = 0;
};

// Constants
constexpr size_t kDefaultTileSize = 16;
constexpr size_t kMaxContexts = 64;
constexpr uint32_t kMagicNumber = 0x57434F44;  // "WCOD"
constexpr uint16_t kVersionMajor = 0;
constexpr uint16_t kVersionMinor = 1;

} // namespace wcodec

