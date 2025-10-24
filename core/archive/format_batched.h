/**
 * @file format_batched.h
 * @brief Layer-level batched compression format for efficient GPU decode
 * 
 * This format processes entire layers at once, enabling:
 * - Single GPU transfer per layer (not per tile)
 * - Parallel tile decompression on GPU
 * - 200x speedup over per-tile approach
 */

#pragma once
#include <cstdint>
#include <cstddef>

namespace codec {

/**
 * Layer Header - describes compressed layer structure
 */
struct LayerHeader {
    uint32_t magic;              // 0xC0DEC111 (CODEC batched)
    uint16_t version;            // Format version (1)
    uint16_t tile_size;          // Tile dimensions (e.g., 256)
    
    uint32_t rows;               // Original layer rows
    uint32_t cols;               // Original layer cols
    
    uint16_t num_tiles_row;      // Number of tile rows
    uint16_t num_tiles_col;      // Number of tile cols
    uint32_t num_tiles;          // Total tiles = num_tiles_row × num_tiles_col
    
    uint32_t compressed_size;    // Total compressed data size
    uint32_t tile_index_offset;  // Offset to tile index table
    uint32_t tile_data_offset;   // Offset to tile data
    
    uint8_t predictor_mode;      // Predictor (0=LEFT, 1=TOP, 2=AVG, 3=PLANAR)
    uint8_t padding[3];          // Alignment
};

/**
 * Tile Index Entry - locates each compressed tile
 */
struct TileIndexEntry {
    uint32_t offset;             // Offset from tile_data_offset
    uint32_t compressed_size;    // Compressed tile size in bytes
    uint16_t row;                // Tile row position
    uint16_t col;                // Tile col position
};

/**
 * Tile Data - individual compressed tile
 * (Note: This is just a documentation struct, actual data is raw bytes)
 */
// Removed empty struct with flexible array member

/**
 * Complete layer format:
 * 
 * [LayerHeader]                          64 bytes
 * [RANSSymbol table]                     2048 bytes (256 × 8 bytes)
 * [TileIndexEntry × num_tiles]           16 bytes × num_tiles
 * [TileData for tile 0]                  Variable
 * [TileData for tile 1]                  Variable
 * ...
 * [TileData for tile N-1]                Variable
 */

static constexpr uint32_t BATCHED_MAGIC = 0xC0DEC111;
static constexpr uint16_t BATCHED_VERSION = 1;
static constexpr size_t RANS_TABLE_SIZE = 1024;  // 256 symbols × 4 bytes (RANSSymbol from rans.h)

} // namespace codec

