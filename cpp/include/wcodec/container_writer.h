/**
 * @file container_writer.h
 * @brief Writer for .wcodec container format
 */

#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

namespace wcodec {

/**
 * Container metadata
 */
struct ContainerMetadata {
    uint16_t version = 1;
    std::string model_name;
    std::string quantization_type;  // "int8", "int4", etc.
    size_t default_tile_size = 16;
    
    // Compression stats
    uint64_t total_uncompressed = 0;
    uint64_t total_compressed = 0;
    double compression_ratio = 1.0;
};

/**
 * Writer for .wcodec files
 */
class ContainerWriter {
public:
    /**
     * Open file for writing
     */
    explicit ContainerWriter(const std::string& path);
    
    /**
     * Close and finalize file
     */
    ~ContainerWriter();
    
    /**
     * Set container metadata
     */
    void setMetadata(const ContainerMetadata& metadata);
    
    /**
     * Add a compressed layer
     * 
     * @param info Layer metadata
     * @param compressed_data Compressed bitstream
     */
    void addLayer(const LayerInfo& info, const std::vector<uint8_t>& compressed_data);
    
    /**
     * Finalize and write footer
     */
    void finalize();
    
    /**
     * Check if write was successful
     */
    bool isValid() const { return valid_; }
    
private:
    std::ofstream file_;
    bool valid_ = false;
    bool finalized_ = false;
    
    ContainerMetadata metadata_;
    std::vector<LayerInfo> layers_;
    
    uint64_t current_offset_ = 0;
    uint32_t file_crc32_ = 0;
    
    void writeHeader();
    void writeLayerIndex();
    void writeFooter();
    uint32_t computeCRC32(const uint8_t* data, size_t size);
};

} // namespace wcodec

