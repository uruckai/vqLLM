/**
 * @file decoder.cpp
 * @brief High-level decoder implementation
 */

#include "wcodec/decoder.h"
#include "wcodec/predictor.h"
#include "wcodec/rans.h"
#include <cstring>
#include <chrono>

namespace wcodec {

Decoder::Decoder(const TileConfig& config) : config_(config) {}

Decoder::~Decoder() {}

void Decoder::setTileSize(size_t rows, size_t cols) {
    config_.tile_rows = rows;
    config_.tile_cols = cols;
}

DecodeStats Decoder::decodeLayer(
    const uint8_t* input,
    size_t size,
    size_t rows,
    size_t cols,
    int8_t* output
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    DecodeStats stats;
    stats.compressed_bytes = size;
    stats.decompressed_bytes = rows * cols;
    
    // Calculate tile grid (must match encoder)
    size_t num_tiles_row = (rows + config_.tile_rows - 1) / config_.tile_rows;
    size_t num_tiles_col = (cols + config_.tile_cols - 1) / config_.tile_cols;
    stats.num_tiles = num_tiles_row * num_tiles_col;
    
    // Parse header and metadata (simplified for now)
    size_t offset = 64;  // Skip header
    
    // Read tile metadata
    std::vector<TileMetadata> tile_metadata(stats.num_tiles);
    for (size_t i = 0; i < stats.num_tiles; ++i) {
        tile_metadata[i].predictor = static_cast<PredictorMode>(input[offset++]);
        tile_metadata[i].transform_map[0] = input[offset++];
    }
    
    // For now, assume streams are contiguous (improve in Week 3)
    size_t stream_offset = offset;
    
    // Decode each tile
    std::vector<int8_t> tile_buf(config_.tile_rows * config_.tile_cols);
    std::vector<int8_t> left_col(config_.tile_rows);
    std::vector<int8_t> top_row(config_.tile_cols);
    
    for (size_t tr = 0; tr < num_tiles_row; ++tr) {
        for (size_t tc = 0; tc < num_tiles_col; ++tc) {
            size_t tile_idx = tr * num_tiles_col + tc;
            
            size_t tile_rows = std::min(config_.tile_rows, rows - tr * config_.tile_rows);
            size_t tile_cols = std::min(config_.tile_cols, cols - tc * config_.tile_cols);
            
            // Get neighbors (from already decoded tiles)
            const int8_t* left_ptr = nullptr;
            const int8_t* top_ptr = nullptr;
            
            if (tc > 0) {
                for (size_t r = 0; r < tile_rows; ++r) {
                    size_t src_r = tr * config_.tile_rows + r;
                    size_t src_c = tc * config_.tile_cols - 1;
                    left_col[r] = output[src_r * cols + src_c];
                }
                left_ptr = left_col.data();
            }
            
            if (tr > 0) {
                for (size_t c = 0; c < tile_cols; ++c) {
                    size_t src_r = tr * config_.tile_rows - 1;
                    size_t src_c = tc * config_.tile_cols + c;
                    top_row[c] = output[src_r * cols + src_c];
                }
                top_ptr = top_row.data();
            }
            
            // Decode tile (simplified - needs proper stream length)
            size_t tile_stream_size = (size - stream_offset) / (stats.num_tiles - tile_idx);
            decodeTile(input + stream_offset, tile_stream_size,
                      tile_metadata[tile_idx],
                      tile_buf.data(), tile_rows, tile_cols,
                      left_ptr, top_ptr);
            stream_offset += tile_stream_size;
            
            // Write tile to output
            for (size_t r = 0; r < tile_rows; ++r) {
                for (size_t c = 0; c < tile_cols; ++c) {
                    size_t dst_r = tr * config_.tile_rows + r;
                    size_t dst_c = tc * config_.tile_cols + c;
                    output[dst_r * cols + dst_c] = tile_buf[r * config_.tile_cols + c];
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    stats.decode_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return stats;
}

void Decoder::decodeTile(
    const uint8_t* input,
    size_t input_size,
    const TileMetadata& metadata,
    int8_t* tile,
    size_t rows,
    size_t cols,
    const int8_t* left_col,
    const int8_t* top_row
) {
    size_t size = rows * cols;
    std::vector<int8_t> residual(size);
    
    // Build frequency table (in real implementation, should be stored)
    // For now, assume uniform distribution (will fix in Week 3)
    uint32_t freqs[256];
    for (int i = 0; i < 256; ++i) {
        freqs[i] = 16;  // 256 * 16 = 4096
    }
    
    // Decode with rANS
    RansDecoder decoder;
    decoder.init(input, input_size, freqs, 256);
    
    for (size_t i = 0; i < size && decoder.hasMore(); ++i) {
        residual[i] = static_cast<int8_t>(decoder.decode());
    }
    
    // Reconstruct from residual
    Predictor::reconstruct(residual.data(), tile, rows, cols,
                          left_col, top_row, metadata.predictor);
}

} // namespace wcodec

