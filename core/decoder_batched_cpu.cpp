/**
 * @file decoder_batched_cpu.cpp
 * @brief CPU decoder implementation for batched format
 */

#include "decoder_batched_cpu.h"
#include "rans.h"
#include <cstring>
#include <chrono>
#include <iostream>

namespace codec {

float BatchedCPUDecoder::decodeLayer(const std::vector<uint8_t>& compressed, int8_t* output) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Parse header
    const LayerHeader* header = reinterpret_cast<const LayerHeader*>(compressed.data());
    
    if (header->magic != BATCHED_MAGIC) {
        std::cerr << "Invalid magic number: " << std::hex << header->magic << std::endl;
        return -1.0f;
    }
    
    // Get pointers to data sections
    const uint8_t* rans_table_ptr = compressed.data() + sizeof(LayerHeader);
    const RANSSymbol* rans_table = reinterpret_cast<const RANSSymbol*>(rans_table_ptr);
    
    const TileIndexEntry* tile_index = reinterpret_cast<const TileIndexEntry*>(
        compressed.data() + header->tile_index_offset);
    
    const uint8_t* tile_data_base = compressed.data() + header->tile_data_offset;
    
    // Decode each tile
    for (uint32_t tile_idx = 0; tile_idx < header->num_tiles; ++tile_idx) {
        const TileIndexEntry& entry = tile_index[tile_idx];
        const uint8_t* tile_compressed = tile_data_base + entry.offset;
        uint32_t tile_compressed_size = entry.compressed_size;
        
        // Read uncompressed size from tile header (first 4 bytes)
        uint32_t uncompressed_size = 
            tile_compressed[0] |
            (tile_compressed[1] << 8) |
            (tile_compressed[2] << 16) |
            (tile_compressed[3] << 24);
        
        // Manually decode rANS using global frequency table
        // The tile data is: [4-byte size][rANS encoded data][4-byte state]
        
        // Initialize state from end of tile
        uint32_t state = 
            tile_compressed[tile_compressed_size - 4] |
            (tile_compressed[tile_compressed_size - 3] << 8) |
            (tile_compressed[tile_compressed_size - 2] << 16) |
            (tile_compressed[tile_compressed_size - 1] << 24);
        
        // Data pointer (starts after size, ends before state)
        const uint8_t* data_ptr = tile_compressed + tile_compressed_size - 5;
        
        // Decode symbols
        std::vector<uint8_t> diff_data(uncompressed_size);
        const uint32_t RANS_SCALE = 1 << 12;  // 4096
        const uint32_t RANS_L = 1 << 23;      // 8388608
        
        for (size_t i = 0; i < uncompressed_size; i++) {
            // Find symbol using cumulative frequency
            uint32_t cum_freq = state & (RANS_SCALE - 1);
            
            // Linear search for symbol (same as rans.cpp)
            uint8_t symbol = 0;
            for (int s = 0; s < 256; s++) {
                if (rans_table[s].start <= cum_freq && 
                    cum_freq < rans_table[s].start + rans_table[s].freq) {
                    symbol = s;
                    break;
                }
            }
            
            // Update state
            const RANSSymbol& sym = rans_table[symbol];
            state = sym.freq * (state >> 12) + (cum_freq - sym.start);
            
            // Renormalize
            while (state < RANS_L) {
                state = (state << 8) | (*data_ptr--);
            }
            
            diff_data[i] = symbol;
        }
        
        // Apply inverse differential encoding (LEFT predictor)
        uint32_t tile_row = entry.row;
        uint32_t tile_col = entry.col;
        uint32_t tile_size = header->tile_size;
        
        // First pixel
        int32_t val = static_cast<int32_t>(diff_data[0]) - 128;
        uint32_t out_row = tile_row * tile_size;
        uint32_t out_col = tile_col * tile_size;
        
        if (out_row < header->rows && out_col < header->cols) {
            output[out_row * header->cols + out_col] = static_cast<int8_t>(val);
        }
        
        // Remaining pixels (simple LEFT predictor: use immediately previous pixel)
        for (uint32_t i = 1; i < uncompressed_size; i++) {
            uint32_t local_row = i / tile_size;
            uint32_t local_col = i % tile_size;
            
            // LEFT predictor: always use previous pixel (i-1)
            int32_t residual = static_cast<int32_t>(diff_data[i]) - 128;
            val = val + residual;  // val holds the previous reconstructed pixel
            
            out_row = tile_row * tile_size + local_row;
            out_col = tile_col * tile_size + local_col;
            
            if (out_row < header->rows && out_col < header->cols) {
                output[out_row * header->cols + out_col] = static_cast<int8_t>(val);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    float milliseconds = std::chrono::duration<float, std::milli>(end - start).count();
    
    return milliseconds;
}

} // namespace codec

