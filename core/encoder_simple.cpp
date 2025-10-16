/**
 * @file encoder_simple.cpp
 * @brief Simplified encoder with rANS compression
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
    fprintf(stderr, "[ENC] start encode rows=%u cols=%u tile=%u\n",
            rows, cols, tile_size_);
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

    // PASS 1: Collect all differential data to build global frequency table
    std::vector<std::vector<uint8_t>> all_tile_diffs(num_tiles);
    std::vector<uint8_t> all_diffs_combined;
    
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
            if (tile_idx == 0) {
                fprintf(stderr, "[ENC] tile0 size=%zu diff_data[0..3]=%u %u %u %u\n",
                        diff_data.size(),
                        diff_data.size() > 0 ? diff_data[0] : 0,
                        diff_data.size() > 1 ? diff_data[1] : 0,
                        diff_data.size() > 2 ? diff_data[2] : 0,
                        diff_data.size() > 3 ? diff_data[3] : 0);
            }
            
            all_tile_diffs[tile_idx] = diff_data;
            
            // Only accumulate up to 1MB of data to avoid memory issues
            if (all_diffs_combined.size() < 1024 * 1024) {
                all_diffs_combined.insert(all_diffs_combined.end(), diff_data.begin(), diff_data.end());
            }
        }
    }
    
    // Ensure we have some data for frequency table
    if (all_diffs_combined.empty() && !all_tile_diffs.empty()) {
        all_diffs_combined = all_tile_diffs[0];
    }
    
    // Build ONE global frequency table
    RANSEncoder global_rans;
    global_rans.buildFrequencies(all_diffs_combined.data(), all_diffs_combined.size());
    
    // Write global frequency table (512 bytes) once
    const auto* symbols = global_rans.getSymbolTable();
    for (int i = 0; i < 256; i++) {
        output.push_back((symbols[i].freq >> 0) & 0xFF);
        output.push_back((symbols[i].freq >> 8) & 0xFF);
    }
    
    // PASS 2: Encode each tile with rANS using shared frequency table
    RANSEncoder tile_rans;
    tile_rans.buildFrequencies(all_diffs_combined.data(), all_diffs_combined.size());
    
    size_t total_before_tiles = output.size();
    size_t total_input_bytes = 0;
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        tile_metadata[tile_idx].data_offset = output.size();
        
        total_input_bytes += all_tile_diffs[tile_idx].size();
        
        // Encode with rANS (WITHOUT frequency table - it's global)
        std::vector<uint8_t> compressed = tile_rans.encodeWithoutFreqTable(
            all_tile_diffs[tile_idx].data(), 
            all_tile_diffs[tile_idx].size());
        
        // Write compressed tile data
        output.insert(output.end(), compressed.begin(), compressed.end());
        
        tile_metadata[tile_idx].data_size = output.size() - tile_metadata[tile_idx].data_offset;
        
        if (tile_idx == 0) {
            fprintf(stderr, "Tile 0: input=%zu bytes, compressed=%zu bytes (ratio=%.2fx)\n", 
                    all_tile_diffs[tile_idx].size(), compressed.size(),
                    (float)all_tile_diffs[tile_idx].size() / compressed.size());
        }
    }
    size_t total_tile_data = output.size() - total_before_tiles;
    fprintf(stderr, "rANS compression: %u tiles, input=%zu bytes, compressed=%zu bytes (ratio=%.2fx)\n",
            num_tiles, total_input_bytes, total_tile_data, 
            (float)total_input_bytes / total_tile_data);
    
    // Copy metadata into output buffer
    memcpy(output.data() + metadata_offset, tile_metadata.data(), num_tiles * sizeof(TileMetadata));

    // Calculate compression ratio
    size_t original_size = rows * cols;
    float ratio = static_cast<float>(original_size) / output.size();
    return ratio;
}


PredictorMode Encoder::selectPredictor(const int8_t* tile, uint32_t rows, uint32_t cols,
                                       const int8_t* left, const int8_t* top) {
    // Simple selection: try LEFT and TOP, pick better
    std::vector<int8_t> residual_left(rows * cols);
    std::vector<int8_t> residual_top(rows * cols);

    predict(tile, residual_left.data(), rows, cols, left, top, PRED_LEFT);
    predict(tile, residual_top.data(), rows, cols, left, top, PRED_TOP);

    // Compute energy
    float energy_left = 0, energy_top = 0;
    for (size_t i = 0; i < rows * cols; i++) {
        energy_left += std::abs(residual_left[i]);
        energy_top += std::abs(residual_top[i]);
    }

    return (energy_left <= energy_top) ? PRED_LEFT : PRED_TOP;
}

void Encoder::predict(const int8_t* tile, int8_t* residual, uint32_t rows, uint32_t cols,
                     const int8_t* left, const int8_t* top, PredictorMode mode) {
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            int idx = r * cols + c;
            int8_t pred = 0;

            switch (mode) {
                case PRED_LEFT:
                    pred = (c > 0) ? tile[idx - 1] : (left ? left[r] : 0);
                    break;

                case PRED_TOP:
                    pred = (r > 0) ? tile[idx - cols] : (top ? top[c] : 0);
                    break;

                case PRED_AVG:
                    // Simplified - just use left for now
                    pred = (c > 0) ? tile[idx - 1] : (left ? left[r] : 0);
                    break;

                case PRED_PLANAR:
                    // Simplified - just use left for now
                    pred = (c > 0) ? tile[idx - 1] : (left ? left[r] : 0);
                    break;
            }

            residual[idx] = tile[idx] - pred;
        }
    }
}

void Encoder::buildFrequencyTable(const int8_t* data, size_t size, uint32_t* freqs) {
    memset(freqs, 0, 256 * sizeof(uint32_t));

    for (size_t i = 0; i < size; i++) {
        uint8_t symbol = static_cast<uint8_t>(data[i]);
        freqs[symbol]++;
    }

    // Ensure no zero frequencies
    for (int i = 0; i < 256; i++) {
        if (freqs[i] == 0) freqs[i] = 1;
    }
}

void Encoder::normalizeFrequencies(uint32_t* freqs, size_t num_symbols, uint32_t scale) {
    // Simple normalization
    uint32_t max_freq = 0;
    for (size_t i = 0; i < num_symbols; i++) {
        if (freqs[i] > max_freq) max_freq = freqs[i];
    }

    for (size_t i = 0; i < num_symbols; i++) {
        freqs[i] = std::max(1u, (freqs[i] * scale) / max_freq);
    }
}

void Encoder::ransEncode(const int8_t* data, size_t size, const uint32_t* freqs,
                        std::vector<uint8_t>& output) {
    // Simplified differential encoding (use int32 to avoid overflow)
    int32_t prev = 0;

    // Write size header
    uint32_t data_size = static_cast<uint32_t>(size);
    output.push_back((data_size >> 0) & 0xFF);
    output.push_back((data_size >> 8) & 0xFF);
    output.push_back((data_size >> 16) & 0xFF);
    output.push_back((data_size >> 24) & 0xFF);

    // Write differential data
    for (size_t i = 0; i < size; i++) {
        int32_t current = static_cast<int32_t>(data[i]);
        int32_t diff = current - prev;
        output.push_back(static_cast<uint8_t>((diff + 128) & 0xFF));
        prev = current;
    }
}

} // namespace codec
