/**
 * @file container_writer.cpp
 * @brief Implementation of container writer
 */

#include "wcodec/container_writer.h"
#include <cstring>
#include <stdexcept>

namespace wcodec {

// Magic bytes
static const char MAGIC_HEADER[4] = {'W', 'C', 'D', 'C'};
static const char MAGIC_FOOTER[4] = {'C', 'D', 'C', 'W'};

// Simple CRC32 implementation
uint32_t ContainerWriter::computeCRC32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < size; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return ~crc;
}

ContainerWriter::ContainerWriter(const std::string& path) {
    file_.open(path, std::ios::binary | std::ios::out);
    if (!file_.is_open()) {
        return;
    }
    
    valid_ = true;
    current_offset_ = 0;
    
    // Reserve space for header (will write after layers are added)
    // For now, just mark position
}

ContainerWriter::~ContainerWriter() {
    if (valid_ && !finalized_) {
        finalize();
    }
}

void ContainerWriter::setMetadata(const ContainerMetadata& metadata) {
    metadata_ = metadata;
}

void ContainerWriter::addLayer(const LayerInfo& info, const std::vector<uint8_t>& compressed_data) {
    if (!valid_) {
        throw std::runtime_error("Invalid container writer");
    }
    
    LayerInfo layer_info = info;
    layer_info.compressed_size = compressed_data.size();
    layer_info.crc32 = computeCRC32(compressed_data.data(), compressed_data.size());
    
    // Store layer info
    layers_.push_back(layer_info);
    
    // Update metadata stats
    metadata_.total_uncompressed += info.uncompressed_size;
    metadata_.total_compressed += compressed_data.size();
}

void ContainerWriter::writeHeader() {
    // Magic bytes
    file_.write(MAGIC_HEADER, 4);
    
    // Version
    file_.write(reinterpret_cast<const char*>(&metadata_.version), sizeof(uint16_t));
    
    // Metadata as simple binary format (simplified - in production use JSON)
    uint32_t metadata_size = metadata_.model_name.size() + metadata_.quantization_type.size() + 100;
    file_.write(reinterpret_cast<const char*>(&metadata_size), sizeof(uint32_t));
    
    // Model name
    uint32_t name_len = metadata_.model_name.size();
    file_.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
    file_.write(metadata_.model_name.data(), name_len);
    
    // Quantization type
    uint32_t quant_len = metadata_.quantization_type.size();
    file_.write(reinterpret_cast<const char*>(&quant_len), sizeof(uint32_t));
    file_.write(metadata_.quantization_type.data(), quant_len);
    
    // Tile size
    file_.write(reinterpret_cast<const char*>(&metadata_.default_tile_size), sizeof(size_t));
    
    // Stats
    file_.write(reinterpret_cast<const char*>(&metadata_.total_uncompressed), sizeof(uint64_t));
    file_.write(reinterpret_cast<const char*>(&metadata_.total_compressed), sizeof(uint64_t));
    
    double ratio = static_cast<double>(metadata_.total_uncompressed) / 
                   std::max(1UL, metadata_.total_compressed);
    file_.write(reinterpret_cast<const char*>(&ratio), sizeof(double));
    
    current_offset_ = file_.tellp();
}

void ContainerWriter::writeLayerIndex() {
    // Number of layers
    uint32_t num_layers = layers_.size();
    file_.write(reinterpret_cast<const char*>(&num_layers), sizeof(uint32_t));
    
    // Calculate offsets (after index)
    uint64_t data_offset = current_offset_ + sizeof(uint32_t);  // After num_layers
    
    // Estimate index size
    for (const auto& layer : layers_) {
        data_offset += sizeof(uint32_t);  // name length
        data_offset += layer.name.size();
        data_offset += sizeof(size_t) * 2;  // rows, cols
        data_offset += sizeof(uint64_t) * 3;  // offset, compressed_size, uncompressed_size
        data_offset += sizeof(uint32_t);  // crc32
        data_offset += sizeof(size_t) * 2;  // tile_rows, tile_cols
        
        size_t num_tiles = ((layer.rows + layer.tile_rows - 1) / layer.tile_rows) *
                           ((layer.cols + layer.tile_cols - 1) / layer.tile_cols);
        
        data_offset += sizeof(uint32_t);  // num_tiles
        data_offset += num_tiles * sizeof(uint8_t);  // predictor_modes
        data_offset += num_tiles * sizeof(uint8_t);  // transform_types
        data_offset += num_tiles * 256 * sizeof(uint32_t);  // frequency tables
        data_offset += num_tiles * sizeof(uint32_t) * 2;  // offsets, sizes
    }
    
    // Write each layer's metadata
    uint64_t current_data_offset = data_offset;
    
    for (auto& layer : layers_) {
        layer.offset = current_data_offset;
        
        // Name
        uint32_t name_len = layer.name.size();
        file_.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
        file_.write(layer.name.data(), name_len);
        
        // Shape
        file_.write(reinterpret_cast<const char*>(&layer.rows), sizeof(size_t));
        file_.write(reinterpret_cast<const char*>(&layer.cols), sizeof(size_t));
        
        // Offsets and sizes
        file_.write(reinterpret_cast<const char*>(&layer.offset), sizeof(uint64_t));
        file_.write(reinterpret_cast<const char*>(&layer.compressed_size), sizeof(uint64_t));
        file_.write(reinterpret_cast<const char*>(&layer.uncompressed_size), sizeof(uint64_t));
        file_.write(reinterpret_cast<const char*>(&layer.crc32), sizeof(uint32_t));
        
        // Tile info
        file_.write(reinterpret_cast<const char*>(&layer.tile_rows), sizeof(size_t));
        file_.write(reinterpret_cast<const char*>(&layer.tile_cols), sizeof(size_t));
        
        size_t num_tiles = layer.predictor_modes.size();
        file_.write(reinterpret_cast<const char*>(&num_tiles), sizeof(uint32_t));
        
        // Per-tile metadata
        file_.write(reinterpret_cast<const char*>(layer.predictor_modes.data()), 
                    num_tiles * sizeof(uint8_t));
        file_.write(reinterpret_cast<const char*>(layer.transform_types.data()), 
                    num_tiles * sizeof(uint8_t));
        
        // Frequency tables
        for (const auto& freq_table : layer.frequency_tables) {
            file_.write(reinterpret_cast<const char*>(freq_table.data()), 
                        256 * sizeof(uint32_t));
        }
        
        // Tile offsets and sizes
        file_.write(reinterpret_cast<const char*>(layer.tile_offsets.data()), 
                    num_tiles * sizeof(uint32_t));
        file_.write(reinterpret_cast<const char*>(layer.tile_sizes.data()), 
                    num_tiles * sizeof(uint32_t));
        
        current_data_offset += layer.compressed_size;
    }
    
    current_offset_ = file_.tellp();
}

void ContainerWriter::writeFooter() {
    // Compute file CRC (simplified - skip for now)
    file_crc32_ = 0x12345678;  // Placeholder
    
    file_.write(reinterpret_cast<const char*>(&file_crc32_), sizeof(uint32_t));
    file_.write(MAGIC_FOOTER, 4);
}

void ContainerWriter::finalize() {
    if (!valid_ || finalized_) {
        return;
    }
    
    // Go back to beginning and write header
    file_.seekp(0);
    writeHeader();
    
    // Write layer index
    writeLayerIndex();
    
    // Write actual layer data (stored in memory for now - in production stream directly)
    // This is a simplified version - real implementation would write data as layers are added
    
    // Write footer
    writeFooter();
    
    file_.close();
    finalized_ = true;
}

} // namespace wcodec

