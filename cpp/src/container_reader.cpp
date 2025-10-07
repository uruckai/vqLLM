/**
 * @file container_reader.cpp
 * @brief Implementation of container reader
 */

#include "wcodec/container_reader.h"
#include <cstring>
#include <stdexcept>

namespace wcodec {

// Magic bytes
static const char MAGIC_HEADER[4] = {'W', 'C', 'D', 'C'};
static const char MAGIC_FOOTER[4] = {'C', 'D', 'C', 'W'};

uint32_t ContainerReader::computeCRC32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < size; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return ~crc;
}

ContainerReader::ContainerReader(const std::string& path) {
    file_.open(path, std::ios::binary | std::ios::in);
    if (!file_.is_open()) {
        return;
    }
    
    // Parse header
    if (!parseHeader()) {
        return;
    }
    
    // Parse layer index
    if (!parseLayerIndex()) {
        return;
    }
    
    valid_ = true;
}

ContainerReader::~ContainerReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

bool ContainerReader::parseHeader() {
    // Check magic bytes
    char magic[4];
    file_.read(magic, 4);
    if (std::memcmp(magic, MAGIC_HEADER, 4) != 0) {
        return false;
    }
    
    // Read version
    file_.read(reinterpret_cast<char*>(&metadata_.version), sizeof(uint16_t));
    
    // Read metadata size
    uint32_t metadata_size;
    file_.read(reinterpret_cast<char*>(&metadata_size), sizeof(uint32_t));
    
    // Model name
    uint32_t name_len;
    file_.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
    metadata_.model_name.resize(name_len);
    file_.read(&metadata_.model_name[0], name_len);
    
    // Quantization type
    uint32_t quant_len;
    file_.read(reinterpret_cast<char*>(&quant_len), sizeof(uint32_t));
    metadata_.quantization_type.resize(quant_len);
    file_.read(&metadata_.quantization_type[0], quant_len);
    
    // Tile size
    file_.read(reinterpret_cast<char*>(&metadata_.default_tile_size), sizeof(size_t));
    
    // Stats
    file_.read(reinterpret_cast<char*>(&metadata_.total_uncompressed), sizeof(uint64_t));
    file_.read(reinterpret_cast<char*>(&metadata_.total_compressed), sizeof(uint64_t));
    file_.read(reinterpret_cast<char*>(&metadata_.compression_ratio), sizeof(double));
    
    return true;
}

bool ContainerReader::parseLayerIndex() {
    // Number of layers
    uint32_t num_layers;
    file_.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));
    
    layers_.resize(num_layers);
    
    // Read each layer's metadata
    for (size_t i = 0; i < num_layers; i++) {
        if (!parseLayerMetadata(layers_[i])) {
            return false;
        }
    }
    
    return true;
}

bool ContainerReader::parseLayerMetadata(LayerInfo& info) {
    // Name
    uint32_t name_len;
    file_.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
    info.name.resize(name_len);
    file_.read(&info.name[0], name_len);
    
    // Shape
    file_.read(reinterpret_cast<char*>(&info.rows), sizeof(size_t));
    file_.read(reinterpret_cast<char*>(&info.cols), sizeof(size_t));
    
    // Offsets and sizes
    file_.read(reinterpret_cast<char*>(&info.offset), sizeof(uint64_t));
    file_.read(reinterpret_cast<char*>(&info.compressed_size), sizeof(uint64_t));
    file_.read(reinterpret_cast<char*>(&info.uncompressed_size), sizeof(uint64_t));
    file_.read(reinterpret_cast<char*>(&info.crc32), sizeof(uint32_t));
    
    // Tile info
    file_.read(reinterpret_cast<char*>(&info.tile_rows), sizeof(size_t));
    file_.read(reinterpret_cast<char*>(&info.tile_cols), sizeof(size_t));
    
    uint32_t num_tiles;
    file_.read(reinterpret_cast<char*>(&num_tiles), sizeof(uint32_t));
    
    // Per-tile metadata
    info.predictor_modes.resize(num_tiles);
    file_.read(reinterpret_cast<char*>(info.predictor_modes.data()), 
               num_tiles * sizeof(uint8_t));
    
    info.transform_types.resize(num_tiles);
    file_.read(reinterpret_cast<char*>(info.transform_types.data()), 
               num_tiles * sizeof(uint8_t));
    
    // Frequency tables
    info.frequency_tables.resize(num_tiles);
    for (auto& freq_table : info.frequency_tables) {
        freq_table.resize(256);
        file_.read(reinterpret_cast<char*>(freq_table.data()), 
                   256 * sizeof(uint32_t));
    }
    
    // Tile offsets and sizes
    info.tile_offsets.resize(num_tiles);
    file_.read(reinterpret_cast<char*>(info.tile_offsets.data()), 
               num_tiles * sizeof(uint32_t));
    
    info.tile_sizes.resize(num_tiles);
    file_.read(reinterpret_cast<char*>(info.tile_sizes.data()), 
               num_tiles * sizeof(uint32_t));
    
    return file_.good();
}

std::vector<std::string> ContainerReader::getLayerNames() const {
    std::vector<std::string> names;
    for (const auto& layer : layers_) {
        names.push_back(layer.name);
    }
    return names;
}

const LayerInfo& ContainerReader::getLayerInfo(size_t index) const {
    if (index >= layers_.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return layers_[index];
}

const LayerInfo* ContainerReader::getLayerInfo(const std::string& name) const {
    for (const auto& layer : layers_) {
        if (layer.name == name) {
            return &layer;
        }
    }
    return nullptr;
}

bool ContainerReader::readLayerData(size_t index, std::vector<uint8_t>& output) {
    if (index >= layers_.size()) {
        return false;
    }
    
    const auto& layer = layers_[index];
    
    // Seek to layer data
    file_.seekg(layer.offset);
    
    // Read compressed data
    output.resize(layer.compressed_size);
    file_.read(reinterpret_cast<char*>(output.data()), layer.compressed_size);
    
    // Verify CRC
    uint32_t crc = computeCRC32(output.data(), output.size());
    if (crc != layer.crc32) {
        return false;
    }
    
    return file_.good();
}

bool ContainerReader::readLayerData(const std::string& name, std::vector<uint8_t>& output) {
    for (size_t i = 0; i < layers_.size(); i++) {
        if (layers_[i].name == name) {
            return readLayerData(i, output);
        }
    }
    return false;
}

bool ContainerReader::validateIntegrity() {
    // Validate each layer's CRC
    for (size_t i = 0; i < layers_.size(); i++) {
        std::vector<uint8_t> data;
        if (!readLayerData(i, data)) {
            return false;
        }
    }
    
    // Validate footer
    return validateFooter();
}

bool ContainerReader::validateFooter() {
    // Seek to end - 8 bytes (crc32 + magic)
    file_.seekg(-8, std::ios::end);
    
    uint32_t file_crc;
    file_.read(reinterpret_cast<char*>(&file_crc), sizeof(uint32_t));
    
    char magic[4];
    file_.read(magic, 4);
    
    return std::memcmp(magic, MAGIC_FOOTER, 4) == 0;
}

} // namespace wcodec

