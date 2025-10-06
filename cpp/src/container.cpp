/**
 * @file container.cpp
 * @brief Implementation of container format
 */

#include "wcodec/container.h"
#include <cstring>
#include <stdexcept>

namespace wcodec {

// ContainerWriter implementation

ContainerWriter::ContainerWriter(const std::string& path)
    : current_offset_(0) {
    file_.open(path, std::ios::binary | std::ios::out);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
}

ContainerWriter::~ContainerWriter() {
    if (file_.is_open()) {
        file_.close();
    }
}

void ContainerWriter::writeHeader(const ContainerHeader& header) {
    file_.write(reinterpret_cast<const char*>(&header), sizeof(header));
    current_offset_ += sizeof(header);
}

void ContainerWriter::beginLayer(const LayerInfo& info) {
    LayerRecord record;
    std::memset(&record, 0, sizeof(record));
    
    std::strncpy(record.name, info.name.c_str(), sizeof(record.name) - 1);
    record.rows = info.rows;
    record.cols = info.cols;
    record.dtype = static_cast<uint8_t>(info.dtype);
    record.num_tiles_row = info.num_tiles_row;
    record.num_tiles_col = info.num_tiles_col;
    record.metadata_offset = current_offset_;
    
    layers_.push_back(record);
}

void ContainerWriter::writeTileMetadata(const TileMetadata& meta) {
    file_.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
    current_offset_ += sizeof(meta);
}

void ContainerWriter::writeTileData(const std::vector<uint8_t>& data) {
    file_.write(reinterpret_cast<const char*>(data.data()), data.size());
    current_offset_ += data.size();
}

void ContainerWriter::writeFrequencyTable(const FrequencyTable& table) {
    file_.write(reinterpret_cast<const char*>(&table), sizeof(table));
    current_offset_ += sizeof(table);
}

void ContainerWriter::endLayer() {
    if (!layers_.empty()) {
        layers_.back().compressed_size = current_offset_ - layers_.back().metadata_offset;
    }
}

void ContainerWriter::finalize() {
    // Write layer index at end
    uint64_t index_offset = current_offset_;
    
    for (const auto& layer : layers_) {
        file_.write(reinterpret_cast<const char*>(&layer), sizeof(layer));
    }
    
    file_.close();
}

// ContainerReader implementation

ContainerReader::ContainerReader(const std::string& path) {
    file_.open(path, std::ios::binary | std::ios::in);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    
    // Read header
    file_.read(reinterpret_cast<char*>(&header_), sizeof(header_));
    
    // Verify magic
    if (header_.magic != kMagicNumber) {
        throw std::runtime_error("Invalid magic number");
    }
    
    // Read layer index
    file_.seekg(header_.layer_index_offset);
    layers_.resize(header_.num_layers);
    
    for (uint32_t i = 0; i < header_.num_layers; ++i) {
        file_.read(reinterpret_cast<char*>(&layers_[i]), sizeof(LayerRecord));
    }
}

ContainerReader::~ContainerReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

ContainerHeader ContainerReader::readHeader() {
    return header_;
}

size_t ContainerReader::numLayers() const {
    return layers_.size();
}

LayerInfo ContainerReader::readLayerInfo(size_t layer_idx) {
    if (layer_idx >= layers_.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    
    const auto& record = layers_[layer_idx];
    
    LayerInfo info;
    info.name = record.name;
    info.rows = record.rows;
    info.cols = record.cols;
    info.dtype = static_cast<DType>(record.dtype);
    info.num_tiles_row = record.num_tiles_row;
    info.num_tiles_col = record.num_tiles_col;
    
    return info;
}

std::vector<TileMetadata> ContainerReader::readTileMetadata(size_t layer_idx) {
    if (layer_idx >= layers_.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    
    const auto& record = layers_[layer_idx];
    size_t num_tiles = record.num_tiles_row * record.num_tiles_col;
    
    std::vector<TileMetadata> metadata(num_tiles);
    
    file_.seekg(record.metadata_offset);
    for (size_t i = 0; i < num_tiles; ++i) {
        file_.read(reinterpret_cast<char*>(&metadata[i]), sizeof(TileMetadata));
    }
    
    return metadata;
}

std::vector<uint8_t> ContainerReader::readTileData(size_t layer_idx, size_t tile_idx) {
    auto metadata = readTileMetadata(layer_idx);
    
    if (tile_idx >= metadata.size()) {
        throw std::out_of_range("Tile index out of range");
    }
    
    const auto& meta = metadata[tile_idx];
    std::vector<uint8_t> data(meta.stream_length);
    
    file_.seekg(meta.stream_offset);
    file_.read(reinterpret_cast<char*>(data.data()), meta.stream_length);
    
    return data;
}

FrequencyTable ContainerReader::readFrequencyTable(size_t layer_idx) {
    if (layer_idx >= layers_.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    
    FrequencyTable table;
    file_.seekg(layers_[layer_idx].data_offset);
    file_.read(reinterpret_cast<char*>(&table), sizeof(table));
    
    return table;
}

} // namespace wcodec

