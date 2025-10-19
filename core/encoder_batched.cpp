/**
 * @file encoder_batched.cpp
 * @brief Batched encoder implementation
 */

#include "encoder_batched.h"
#include <cstring>
#include <algorithm>
#include <cstdio>

namespace codec {

BatchedEncoder::BatchedEncoder(uint16_t tile_size) : tile_size_(tile_size) {}

float BatchedEncoder::encodeLayer(const int8_t* data, uint32_t rows, uint32_t cols,
                                  std::vector<uint8_t>& output) {
    // Calculate tiling
    uint32_t num_tiles_row = (rows + tile_size_ - 1) / tile_size_;
    uint32_t num_tiles_col = (cols + tile_size_ - 1) / tile_size_;
    uint32_t num_tiles = num_tiles_row * num_tiles_col;
    
    // Create header
    LayerHeader header;
    header.magic = BATCHED_MAGIC;
    header.version = BATCHED_VERSION;
    header.tile_size = tile_size_;
    header.rows = rows;
    header.cols = cols;
    header.num_tiles_row = num_tiles_row;
    header.num_tiles_col = num_tiles_col;
    header.num_tiles = num_tiles;
    header.predictor_mode = 0;  // LEFT predictor
    
    // Reserve space for output
    output.clear();
    output.reserve(rows * cols);  // Upper bound
    
    // Write header placeholder (will update later)
    size_t header_offset = output.size();
    output.resize(output.size() + sizeof(LayerHeader));
    
    // PASS 1: Collect all differential data to build global frequency table
    std::vector<std::vector<uint8_t>> all_tile_diffs(num_tiles);
    std::vector<uint8_t> all_diffs_combined;
    all_diffs_combined.reserve(std::min(rows * cols, 1024u * 1024u));  // Limit to 1MB
    
    for (uint32_t tile_row = 0; tile_row < num_tiles_row; ++tile_row) {
        for (uint32_t tile_col = 0; tile_col < num_tiles_col; ++tile_col) {
            uint32_t tile_idx = tile_row * num_tiles_col + tile_col;
            
            // Calculate tile bounds
            uint32_t row_start = tile_row * tile_size_;
            uint32_t col_start = tile_col * tile_size_;
            uint32_t row_end = std::min(row_start + tile_size_, rows);
            uint32_t col_end = std::min(col_start + tile_size_, cols);
            
            // Extract and pad tile
            std::vector<int8_t> tile_data(tile_size_ * tile_size_, 0);
            for (uint32_t r = row_start; r < row_end; ++r) {
                for (uint32_t c = col_start; c < col_end; ++c) {
                    uint32_t tile_r = r - row_start;
                    uint32_t tile_c = c - col_start;
                    tile_data[tile_r * tile_size_ + tile_c] = data[r * cols + c];
                }
            }
            
            // Apply differential encoding
            std::vector<uint8_t> diff_data;
            applyDifferentialEncoding(tile_data.data(), tile_size_ * tile_size_, diff_data);
            
            all_tile_diffs[tile_idx] = diff_data;
            
            // Sample for global frequency table (limit to prevent OOM)
            if (all_diffs_combined.size() < 1024 * 1024) {
                all_diffs_combined.insert(all_diffs_combined.end(), diff_data.begin(), diff_data.end());
            }
        }
    }
    
    // Build global frequency table
    RANSEncoder global_rans;
    global_rans.buildFrequencies(all_diffs_combined.data(), all_diffs_combined.size());
    
    // Write RANSSymbol table
    size_t rans_table_offset = output.size();
    output.resize(output.size() + RANS_TABLE_SIZE);
    memcpy(output.data() + rans_table_offset, global_rans.getSymbolTable(), RANS_TABLE_SIZE);
    
    // Write tile index placeholder
    size_t tile_index_offset = output.size();
    header.tile_index_offset = tile_index_offset;
    output.resize(output.size() + num_tiles * sizeof(TileIndexEntry));
    
    // PASS 2: Encode each tile with shared frequency table
    std::vector<TileIndexEntry> tile_index(num_tiles);
    size_t tile_data_offset = output.size();
    header.tile_data_offset = tile_data_offset;
    
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        size_t tile_start = output.size();
        
        // Encode with rANS using global frequency table
        RANSEncoder tile_rans;
        tile_rans.copyFrequencies(global_rans);  // Reuse global frequencies
        tile_rans.resetState();
        
        std::vector<uint8_t> rans_encoded;
        tile_rans.encodeWithoutFreqTable(all_tile_diffs[tile_idx].data(), 
                                         all_tile_diffs[tile_idx].size(), 
                                         rans_encoded);
        
        // Write tile data
        output.insert(output.end(), rans_encoded.begin(), rans_encoded.end());
        
        // Update index
        tile_index[tile_idx].offset = tile_start - tile_data_offset;
        tile_index[tile_idx].compressed_size = rans_encoded.size();
        tile_index[tile_idx].row = tile_idx / num_tiles_col;
        tile_index[tile_idx].col = tile_idx % num_tiles_col;
    }
    
    // Update header
    header.compressed_size = output.size();
    memcpy(output.data() + header_offset, &header, sizeof(LayerHeader));
    
    // Write tile index
    memcpy(output.data() + tile_index_offset, tile_index.data(), 
           num_tiles * sizeof(TileIndexEntry));
    
    // Calculate compression ratio
    size_t original_size = rows * cols;
    float ratio = static_cast<float>(original_size) / output.size();
    
    return ratio;
}

void BatchedEncoder::applyDifferentialEncoding(const int8_t* input, uint32_t size,
                                               std::vector<uint8_t>& output) {
    output.resize(size);
    
    // First pixel: no prediction, just shift to unsigned
    output[0] = static_cast<uint8_t>(static_cast<int32_t>(input[0]) + 128);
    
    // LEFT predictor for rest
    for (uint32_t i = 1; i < size; ++i) {
        int32_t prediction = input[i - 1];
        int32_t residual = static_cast<int32_t>(input[i]) - prediction;
        output[i] = static_cast<uint8_t>(residual + 128);
    }
}

} // namespace codec

