/**
 * @file encoder_safe.cpp
 * @brief Safer encoder with sampled frequency table building
 * 
 * Instead of accumulating ALL differential data (which can be huge),
 * we sample a subset to build the frequency table
 */

#include "encoder.h"
#include "rans.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdio>

namespace codec {

Encoder::Encoder(uint16_t tile_size) : tile_size_(tile_size) {}

float Encoder::encode(const int8_t* data, uint32_t rows, uint32_t cols,
                     std::vector<uint8_t>& output) {
    // Calculate tiling
    uint32_t num_tiles_row = (rows + tile_size_ - 1) / tile_size_;
    uint32_t num_tiles_col = (cols + tile_size_ - 1) / tile_size_;
    uint32_t num_tiles = num_tiles_row * num_tiles_col;

    // Write header
    Header header;
    header.magic = MAGIC;
    header.version = VERSION;
    header.tile_size = tile_size_;
    header.num_tiles_row = num_tiles_row;
    header.num_tiles_col = num_tiles_col;
    header.output_rows = rows;
    header.output_cols = cols;

    output.clear();
    output.insert(output.end(),
                 reinterpret_cast<uint8_t*>(&header),
                 reinterpret_cast<uint8_t*>(&header) + sizeof(Header));

    // Create temporary metadata storage
    std::vector<TileMetadata> tile_metadata(num_tiles);
    
    // Reserve space for tile metadata in output (will fill later)
    size_t metadata_offset = output.size();
    output.resize(metadata_offset + num_tiles * sizeof(TileMetadata));

    // SAFE PASS 1: Sample tiles to build frequency table (max 10 tiles or 10%)
    uint32_t max_sample_tiles = std::max(10u, num_tiles / 10);
    std::vector<uint8_t> sampled_diffs;
    sampled_diffs.reserve(max_sample_tiles * tile_size_ * tile_size_);
    
    uint32_t sample_stride = std::max(1u, num_tiles / max_sample_tiles);
    
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx += sample_stride) {
        uint32_t ty = tile_idx / num_tiles_col;
        uint32_t tx = tile_idx % num_tiles_col;

        if (ty >= num_tiles_row) break;

        // Calculate tile bounds
        uint32_t row_start = ty * tile_size_;
        uint32_t col_start = tx * tile_size_;
        uint32_t tile_rows = std::min(static_cast<uint32_t>(tile_size_), rows - row_start);
        uint32_t tile_cols = std::min(static_cast<uint32_t>(tile_size_), cols - col_start);
        size_t tile_size = tile_rows * tile_cols;

        // Extract tile data
        std::vector<int8_t> tile_data(tile_size);
        for (uint32_t r = 0; r < tile_rows; r++) {
            memcpy(tile_data.data() + r * tile_cols,
                  data + (row_start + r) * cols + col_start,
                  tile_cols);
        }

        // Get context
        const int8_t* left = (tx > 0) ? data + row_start * cols + col_start - 1 : nullptr;
        const int8_t* top = (ty > 0) ? data + (row_start - 1) * cols + col_start : nullptr;

        // Select predictor and compute residual
        PredictorMode mode = selectPredictor(tile_data.data(), tile_rows, tile_cols, left, top);
        
        std::vector<int8_t> residual(tile_size);
        predict(tile_data.data(), residual.data(), tile_rows, tile_cols, left, top, mode);

        // Differential encoding
        int32_t prev = 0;
        for (size_t i = 0; i < tile_size; i++) {
            int32_t current = static_cast<int32_t>(residual[i]);
            int32_t diff = current - prev;
            uint8_t diff_byte = static_cast<uint8_t>((diff + 128) & 0xFF);
            sampled_diffs.push_back(diff_byte);
            prev = current;
        }
        
        if (sampled_diffs.size() >= max_sample_tiles * tile_size_ * tile_size_) {
            break; // Don't accumulate too much
        }
    }
    
    // Build ONE global frequency table from sampled data
    RANSEncoder global_rans;
    global_rans.buildFrequencies(sampled_diffs.data(), sampled_diffs.size());
    
    // Write global frequency table (512 bytes) once
    const auto* symbols = global_rans.getSymbolTable();
    for (int i = 0; i < 256; i++) {
        output.push_back((symbols[i].freq >> 0) & 0xFF);
        output.push_back((symbols[i].freq >> 8) & 0xFF);
    }
    
    // PASS 2: Encode all tiles using the global frequency table
    for (uint32_t ty = 0; ty < num_tiles_row; ty++) {
        for (uint32_t tx = 0; tx < num_tiles_col; tx++) {
            uint32_t tile_idx = ty * num_tiles_col + tx;

            // Calculate tile bounds
            uint32_t row_start = ty * tile_size_;
            uint32_t col_start = tx * tile_size_;
            uint32_t tile_rows = std::min(static_cast<uint32_t>(tile_size_), rows - row_start);
            uint32_t tile_cols = std::min(static_cast<uint32_t>(tile_size_), cols - col_start);
            size_t tile_size = tile_rows * tile_cols;

            // Extract tile data
            std::vector<int8_t> tile_data(tile_size);
            for (uint32_t r = 0; r < tile_rows; r++) {
                memcpy(tile_data.data() + r * tile_cols,
                      data + (row_start + r) * cols + col_start,
                      tile_cols);
            }

            // Get context
            const int8_t* left = (tx > 0) ? data + row_start * cols + col_start - 1 : nullptr;
            const int8_t* top = (ty > 0) ? data + (row_start - 1) * cols + col_start : nullptr;

            // Select predictor and compute residual
            tile_metadata[tile_idx].predictor_mode = selectPredictor(tile_data.data(), tile_rows, tile_cols, left, top);
            
            std::vector<int8_t> residual(tile_size);
            predict(tile_data.data(), residual.data(), tile_rows, tile_cols, left, top,
                   static_cast<PredictorMode>(tile_metadata[tile_idx].predictor_mode));

            // Differential encoding
            std::vector<uint8_t> diff_data(tile_size);
            int32_t prev = 0;
            for (size_t i = 0; i < tile_size; i++) {
                int32_t current = static_cast<int32_t>(residual[i]);
                int32_t diff = current - prev;
                diff_data[i] = static_cast<uint8_t>((diff + 128) & 0xFF);
                prev = current;
            }
            
            // Record tile offset
            tile_metadata[tile_idx].offset = output.size();
            
            // Encode using global frequency table
            global_rans.resetState();
            std::vector<uint8_t> compressed = global_rans.encodeWithoutFreqTable(diff_data.data(), diff_data.size());
            
            // Record tile size
            tile_metadata[tile_idx].compressed_size = compressed.size();
            tile_metadata[tile_idx].original_size = tile_size;
            
            // Append compressed data
            output.insert(output.end(), compressed.begin(), compressed.end());
        }
    }

    // Write metadata
    memcpy(output.data() + metadata_offset, tile_metadata.data(),
           num_tiles * sizeof(TileMetadata));

    // Calculate compression ratio
    float ratio = (rows * cols) / static_cast<float>(output.size());
    return ratio;
}

// Existing methods...
uint8_t Encoder::selectPredictor(const int8_t* tile, uint32_t rows, uint32_t cols,
                                const int8_t* left, const int8_t* top) {
    // Always use PLANAR for simplicity
    return static_cast<uint8_t>(PredictorMode::PLANAR);
}

void Encoder::predict(const int8_t* tile, int8_t* residual,
                     uint32_t rows, uint32_t cols,
                     const int8_t* left, const int8_t* top,
                     PredictorMode mode) {
    for (uint32_t i = 0; i < rows * cols; i++) {
        int prediction = 0;
        
        uint32_t y = i / cols;
        uint32_t x = i % cols;
        
        if (x == 0 && y == 0) {
            prediction = 0;
        } else if (y == 0) {
            prediction = tile[i - 1];
        } else if (x == 0) {
            prediction = tile[i - cols];
        } else {
            int L = tile[i - 1];
            int T = tile[i - cols];
            int TL = tile[i - cols - 1];
            prediction = L + T - TL;
        }
        
        residual[i] = tile[i] - static_cast<int8_t>(prediction);
    }
}

} // namespace codec

