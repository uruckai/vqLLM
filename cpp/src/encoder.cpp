/**
 * @file encoder.cpp
 * @brief High-level encoder implementation
 */

#include "wcodec/encoder.h"
#include "wcodec/predictor.h"
#include "wcodec/rans.h"
#include <cstring>
#include <chrono>

namespace wcodec {

Encoder::Encoder(const TileConfig& config) : config_(config) {}

Encoder::~Encoder() {}

void Encoder::setTileSize(size_t rows, size_t cols) {
    config_.tile_rows = rows;
    config_.tile_cols = cols;
}

EncodeStats Encoder::encodeLayer(
    const int8_t* data,
    size_t rows,
    size_t cols,
    std::vector<uint8_t>& output
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    EncodeStats stats;
    stats.original_bytes = rows * cols;
    
    // Calculate tile grid
    size_t num_tiles_row = (rows + config_.tile_rows - 1) / config_.tile_rows;
    size_t num_tiles_col = (cols + config_.tile_cols - 1) / config_.tile_cols;
    stats.num_tiles = num_tiles_row * num_tiles_col;
    
    // Temporary storage for tile data
    size_t tile_size = config_.tile_rows * config_.tile_cols;
    std::vector<int8_t> tile_buf(tile_size);
    std::vector<int8_t> left_col(config_.tile_rows);
    std::vector<int8_t> top_row(config_.tile_cols);
    std::vector<int8_t> residual(tile_size);
    
    // Storage for all tile outputs and metadata
    std::vector<std::vector<uint8_t>> tile_outputs(stats.num_tiles);
    std::vector<TileMetadata> tile_metadata(stats.num_tiles);
    
    // Encode each tile
    for (size_t tr = 0; tr < num_tiles_row; ++tr) {
        for (size_t tc = 0; tc < num_tiles_col; ++tc) {
            size_t tile_idx = tr * num_tiles_col + tc;
            
            // Extract tile
            size_t tile_rows = std::min(config_.tile_rows, rows - tr * config_.tile_rows);
            size_t tile_cols = std::min(config_.tile_cols, cols - tc * config_.tile_cols);
            
            for (size_t r = 0; r < tile_rows; ++r) {
                for (size_t c = 0; c < tile_cols; ++c) {
                    size_t src_r = tr * config_.tile_rows + r;
                    size_t src_c = tc * config_.tile_cols + c;
                    tile_buf[r * config_.tile_cols + c] = data[src_r * cols + src_c];
                }
            }
            
            // Get neighbors
            const int8_t* left_ptr = nullptr;
            const int8_t* top_ptr = nullptr;
            
            if (tc > 0) {
                // Get left neighbor's right column
                for (size_t r = 0; r < tile_rows; ++r) {
                    size_t src_r = tr * config_.tile_rows + r;
                    size_t src_c = tc * config_.tile_cols - 1;
                    left_col[r] = data[src_r * cols + src_c];
                }
                left_ptr = left_col.data();
            }
            
            if (tr > 0) {
                // Get top neighbor's bottom row
                for (size_t c = 0; c < tile_cols; ++c) {
                    size_t src_r = tr * config_.tile_rows - 1;
                    size_t src_c = tc * config_.tile_cols + c;
                    top_row[c] = data[src_r * cols + src_c];
                }
                top_ptr = top_row.data();
            }
            
            // Encode tile
            encodeTile(tile_buf.data(), tile_rows, tile_cols,
                      left_ptr, top_ptr,
                      tile_outputs[tile_idx], tile_metadata[tile_idx]);
        }
    }
    
    // Assemble output: header + metadata + streams
    output.clear();
    
    // Simple header (for now)
    size_t header_offset = 0;
    output.resize(64);  // Reserve space for header
    
    // Write metadata
    size_t metadata_offset = output.size();
    for (auto& meta : tile_metadata) {
        output.push_back(static_cast<uint8_t>(meta.predictor));
        // Pack transform map (4 x 2 bits)
        output.push_back(meta.transform_map[0]);
    }
    
    // Write tile stream offsets and data
    size_t streams_offset = output.size();
    for (size_t i = 0; i < tile_outputs.size(); ++i) {
        tile_metadata[i].stream_offset = output.size();
        tile_metadata[i].stream_length = tile_outputs[i].size();
        output.insert(output.end(), tile_outputs[i].begin(), tile_outputs[i].end());
    }
    
    stats.compressed_bytes = output.size();
    stats.compression_ratio = static_cast<double>(stats.original_bytes) / 
                             static_cast<double>(stats.compressed_bytes);
    
    auto end = std::chrono::high_resolution_clock::now();
    stats.encode_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return stats;
}

void Encoder::encodeTile(
    const int8_t* tile,
    size_t rows,
    size_t cols,
    const int8_t* left_col,
    const int8_t* top_row,
    std::vector<uint8_t>& output,
    TileMetadata& metadata
) {
    size_t size = rows * cols;
    std::vector<int8_t> residual(size);
    
    // Select best predictor
    metadata.predictor = Predictor::selectMode(tile, rows, cols, left_col, top_row);
    
    // Compute residual
    Predictor::predict(tile, residual.data(), rows, cols, left_col, top_row, 
                      metadata.predictor);
    
    // For now, no transform (Week 3)
    std::memset(metadata.transform_map, 0, sizeof(metadata.transform_map));
    
    // Build frequency table
    uint32_t freqs[256];
    buildFrequencyTable(residual.data(), size, freqs);
    normalizeFrequencies(freqs, 256, 4096);
    
    // Encode with rANS
    RansEncoder encoder;
    encoder.init(freqs, 256);
    
    for (size_t i = 0; i < size; ++i) {
        encoder.encode(static_cast<uint8_t>(residual[i]));
    }
    
    encoder.finish(output);
}

} // namespace wcodec

