/**
 * @file encoder.cpp
 * @brief Encoder implementation
 */

#include "encoder.h"
#include <algorithm>
#include <cstring>
#include <cmath>

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
    
    // Reserve space for tile metadata (will fill in later)
    size_t metadata_offset = output.size();
    output.resize(metadata_offset + num_tiles * sizeof(TileMetadata));
    auto* tile_metadata = reinterpret_cast<TileMetadata*>(output.data() + metadata_offset);
    
    // Encode each tile
    for (uint32_t ty = 0; ty < num_tiles_row; ty++) {
        for (uint32_t tx = 0; tx < num_tiles_col; tx++) {
            uint32_t tile_idx = ty * num_tiles_col + tx;
            
            // Calculate tile bounds
            uint32_t row_start = ty * tile_size_;
            uint32_t col_start = tx * tile_size_;
            uint32_t tile_rows = std::min(tile_size_, rows - row_start);
            uint32_t tile_cols = std::min(tile_size_, cols - col_start);
            
            // Extract tile data
            std::vector<int8_t> tile_data(tile_rows * tile_cols);
            for (uint32_t r = 0; r < tile_rows; r++) {
                memcpy(tile_data.data() + r * tile_cols,
                      data + (row_start + r) * cols + col_start,
                      tile_cols);
            }
            
            // Get context (left and top tiles)
            const int8_t* left = (tx > 0) ? data + row_start * cols + col_start - 1 : nullptr;
            const int8_t* top = (ty > 0) ? data + (row_start - 1) * cols + col_start : nullptr;
            
            // Encode tile
            tile_metadata[tile_idx].data_offset = output.size();
            encodeTile(tile_data.data(), tile_rows, tile_cols, left, top,
                      output, tile_metadata[tile_idx]);
            tile_metadata[tile_idx].data_size = output.size() - tile_metadata[tile_idx].data_offset;
        }
    }
    
    // Calculate compression ratio
    size_t original_size = rows * cols;
    float ratio = static_cast<float>(original_size) / output.size();
    return ratio;
}

void Encoder::encodeTile(const int8_t* tile, uint32_t tile_rows, uint32_t tile_cols,
                        const int8_t* left, const int8_t* top,
                        std::vector<uint8_t>& output, TileMetadata& metadata) {
    size_t tile_size = tile_rows * tile_cols;
    
    // Select best predictor
    metadata.predictor_mode = selectPredictor(tile, tile_rows, tile_cols, left, top);
    
    // Compute residual
    std::vector<int8_t> residual(tile_size);
    predict(tile, residual.data(), tile_rows, tile_cols, left, top, 
           static_cast<PredictorMode>(metadata.predictor_mode));
    
    // Build frequency table
    buildFrequencyTable(residual.data(), tile_size, metadata.freq_table);
    normalizeFrequencies(metadata.freq_table, 256, FREQ_SCALE);
    
    // rANS encode
    ransEncode(residual.data(), tile_size, metadata.freq_table, output);
}

PredictorMode Encoder::selectPredictor(const int8_t* tile, uint32_t rows, uint32_t cols,
                                       const int8_t* left, const int8_t* top) {
    // Simple selection: try all, pick lowest residual energy
    PredictorMode best_mode = PRED_LEFT;
    float best_energy = 1e30f;
    
    std::vector<int8_t> residual(rows * cols);
    
    for (int mode = 0; mode < 4; mode++) {
        predict(tile, residual.data(), rows, cols, left, top, static_cast<PredictorMode>(mode));
        
        // Compute energy
        float energy = 0;
        for (size_t i = 0; i < rows * cols; i++) {
            energy += std::abs(residual[i]);
        }
        
        if (energy < best_energy) {
            best_energy = energy;
            best_mode = static_cast<PredictorMode>(mode);
        }
    }
    
    return best_mode;
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
                    
                case PRED_AVG: {
                    int8_t left_val = (c > 0) ? tile[idx - 1] : (left ? left[r] : 0);
                    int8_t top_val = (r > 0) ? tile[idx - cols] : (top ? top[c] : 0);
                    pred = (left_val + top_val) / 2;
                    break;
                }
                
                case PRED_PLANAR: {
                    int8_t left_val = (c > 0) ? tile[idx - 1] : (left ? left[r] : 0);
                    int8_t top_val = (r > 0) ? tile[idx - cols] : (top ? top[c] : 0);
                    int8_t tl_val = 0;
                    if (r > 0 && c > 0) {
                        tl_val = tile[idx - cols - 1];
                    }
                    pred = left_val + top_val - tl_val;
                    break;
                }
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
    
    // Ensure no zero frequencies (add 1 to all)
    for (int i = 0; i < 256; i++) {
        freqs[i]++;
    }
}

void Encoder::normalizeFrequencies(uint32_t* freqs, size_t num_symbols, uint32_t scale) {
    // Calculate total
    uint64_t total = 0;
    for (size_t i = 0; i < num_symbols; i++) {
        total += freqs[i];
    }
    
    // Normalize to scale
    uint32_t normalized_sum = 0;
    for (size_t i = 0; i < num_symbols; i++) {
        freqs[i] = std::max(1u, static_cast<uint32_t>((freqs[i] * scale) / total));
        normalized_sum += freqs[i];
    }
    
    // Adjust to exactly match scale
    if (normalized_sum != scale) {
        // Find largest frequency and adjust
        uint32_t max_idx = 0;
        for (size_t i = 1; i < num_symbols; i++) {
            if (freqs[i] > freqs[max_idx]) max_idx = i;
        }
        freqs[max_idx] += (scale - normalized_sum);
    }
}

void Encoder::ransEncode(const int8_t* data, size_t size, const uint32_t* freqs,
                        std::vector<uint8_t>& output) {
    // Build cumulative frequency table
    uint32_t cumul[257];
    cumul[0] = 0;
    for (int i = 0; i < 256; i++) {
        cumul[i + 1] = cumul[i] + freqs[i];
    }
    
    // Simple rANS encoding
    uint64_t state = 1u << 31;  // Initial state
    std::vector<uint8_t> temp_output;
    
    // Encode backwards
    for (int64_t i = size - 1; i >= 0; i--) {
        uint8_t symbol = static_cast<uint8_t>(data[i]);
        
        // Renormalize if needed
        while (state >= (1ull << 32)) {
            temp_output.push_back(state & 0xFF);
            state >>= 8;
        }
        
        // Encode symbol
        uint32_t freq = freqs[symbol];
        uint32_t start = cumul[symbol];
        
        state = ((state / freq) * FREQ_SCALE) + (state % freq) + start;
    }
    
    // Final renormalization
    while (state >= (1u << 8)) {
        temp_output.push_back(state & 0xFF);
        state >>= 8;
    }
    temp_output.push_back(state & 0xFF);
    
    // Reverse and append
    output.insert(output.end(), temp_output.rbegin(), temp_output.rend());
}

} // namespace codec

