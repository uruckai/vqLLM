/**
 * @file encoder_gpu.cpp
 * @brief GPU-friendly encoder implementation
 */

#include "wcodec/encoder_gpu.h"
#include "wcodec/predictor.h"
#include "wcodec/rans.h"
#include <cstring>
#include <chrono>

namespace wcodec {

EncoderGPU::EncoderGPU(const TileConfig& config) : Encoder(config) {}

EncodeStats EncoderGPU::encodeLayerGPU(
    const int8_t* data,
    size_t rows,
    size_t cols,
    GPUEncodedLayer& output
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    EncodeStats stats;
    stats.original_bytes = rows * cols;
    
    // Calculate tile grid
    size_t num_tiles_row = (rows + config_.tile_rows - 1) / config_.tile_rows;
    size_t num_tiles_col = (cols + config_.tile_cols - 1) / config_.tile_cols;
    stats.num_tiles = num_tiles_row * num_tiles_col;
    
    // Set output metadata
    output.num_tiles_row = num_tiles_row;
    output.num_tiles_col = num_tiles_col;
    output.tile_size = config_.tile_rows;  // Assuming square tiles
    output.tiles.resize(stats.num_tiles);
    output.compressed_data.clear();
    
    // Temporary storage
    size_t tile_size = config_.tile_rows * config_.tile_cols;
    std::vector<int8_t> tile_buf(tile_size);
    std::vector<int8_t> left_col(config_.tile_rows);
    std::vector<int8_t> top_row(config_.tile_cols);
    std::vector<int8_t> residual(tile_size);
    
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
                for (size_t r = 0; r < tile_rows; ++r) {
                    size_t src_r = tr * config_.tile_rows + r;
                    size_t src_c = tc * config_.tile_cols - 1;
                    left_col[r] = data[src_r * cols + src_c];
                }
                left_ptr = left_col.data();
            }
            
            if (tr > 0) {
                for (size_t c = 0; c < tile_cols; ++c) {
                    size_t src_r = tr * config_.tile_rows - 1;
                    size_t src_c = tc * config_.tile_cols + c;
                    top_row[c] = data[src_r * cols + src_c];
                }
                top_ptr = top_row.data();
            }
            
            // Select predictor
            PredictorMode pred = Predictor::selectMode(tile_buf.data(), tile_rows, tile_cols, 
                                                       left_ptr, top_ptr);
            output.tiles[tile_idx].predictor_mode = static_cast<uint8_t>(pred);
            
            // Compute residual
            Predictor::predict(tile_buf.data(), residual.data(), tile_rows, tile_cols,
                             left_ptr, top_ptr, pred);
            
            // Build and normalize frequency table
            uint32_t freqs[256];
            buildFrequencyTable(residual.data(), tile_size, freqs);
            normalizeFrequencies(freqs, 256, 4096);
            
            // Store frequency table
            std::memcpy(output.tiles[tile_idx].freq_table, freqs, sizeof(freqs));
            
            // Encode with rANS
            RansEncoder encoder;
            encoder.init(freqs, 256);
            
            for (size_t i = 0; i < tile_size; ++i) {
                encoder.encode(static_cast<uint8_t>(residual[i]));
            }
            
            std::vector<uint8_t> tile_compressed;
            encoder.finish(tile_compressed);
            
            // Store in output
            output.tiles[tile_idx].compressed_offset = output.compressed_data.size();
            output.tiles[tile_idx].compressed_size = tile_compressed.size();
            output.compressed_data.insert(output.compressed_data.end(),
                                         tile_compressed.begin(),
                                         tile_compressed.end());
        }
    }
    
    stats.compressed_bytes = output.compressed_data.size() + 
                            output.tiles.size() * (1 + 256 * 4 + 8);  // Metadata size
    stats.compression_ratio = static_cast<double>(stats.original_bytes) / 
                             static_cast<double>(stats.compressed_bytes);
    
    auto end = std::chrono::high_resolution_clock::now();
    stats.encode_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return stats;
}

void EncoderGPU::serialize(const GPUEncodedLayer& layer, std::vector<uint8_t>& output) {
    output.clear();
    
    // Header
    output.resize(12);
    std::memcpy(&output[0], &layer.num_tiles_row, 4);
    std::memcpy(&output[4], &layer.num_tiles_col, 4);
    std::memcpy(&output[8], &layer.tile_size, 4);
    
    // Tile metadata
    for (const auto& tile : layer.tiles) {
        output.push_back(tile.predictor_mode);
        
        // Frequency table
        size_t freq_offset = output.size();
        output.resize(output.size() + 256 * 4);
        std::memcpy(&output[freq_offset], tile.freq_table, 256 * 4);
        
        // Offsets/sizes
        size_t offset_pos = output.size();
        output.resize(output.size() + 8);
        std::memcpy(&output[offset_pos], &tile.compressed_offset, 4);
        std::memcpy(&output[offset_pos + 4], &tile.compressed_size, 4);
    }
    
    // Compressed data
    output.insert(output.end(), layer.compressed_data.begin(), layer.compressed_data.end());
}

bool EncoderGPU::deserialize(const uint8_t* data, size_t size, GPUEncodedLayer& layer) {
    if (size < 12) return false;
    
    // Read header
    std::memcpy(&layer.num_tiles_row, &data[0], 4);
    std::memcpy(&layer.num_tiles_col, &data[4], 4);
    std::memcpy(&layer.tile_size, &data[8], 4);
    
    size_t num_tiles = layer.num_tiles_row * layer.num_tiles_col;
    layer.tiles.resize(num_tiles);
    
    size_t offset = 12;
    
    // Read tile metadata
    for (size_t i = 0; i < num_tiles; ++i) {
        if (offset + 1 + 256 * 4 + 8 > size) return false;
        
        layer.tiles[i].predictor_mode = data[offset++];
        
        std::memcpy(layer.tiles[i].freq_table, &data[offset], 256 * 4);
        offset += 256 * 4;
        
        std::memcpy(&layer.tiles[i].compressed_offset, &data[offset], 4);
        offset += 4;
        std::memcpy(&layer.tiles[i].compressed_size, &data[offset], 4);
        offset += 4;
    }
    
    // Read compressed data
    layer.compressed_data.assign(data + offset, data + size);
    
    return true;
}

} // namespace wcodec

