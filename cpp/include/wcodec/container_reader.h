/**
 * @file container_reader.h
 * @brief Reader for .wcodec container format
 */

#pragma once

#include "container_writer.h"  // Reuse LayerInfo, ContainerMetadata
#include <string>
#include <vector>
#include <fstream>
#include <memory>

namespace wcodec {

/**
 * Reader for .wcodec files
 */
class ContainerReader {
public:
    /**
     * Open file for reading
     */
    explicit ContainerReader(const std::string& path);
    
    /**
     * Destructor
     */
    ~ContainerReader();
    
    /**
     * Check if file is valid
     */
    bool isValid() const { return valid_; }
    
    /**
     * Get container metadata
     */
    const ContainerMetadata& getMetadata() const { return metadata_; }
    
    /**
     * Get number of layers
     */
    size_t getNumLayers() const { return layers_.size(); }
    
    /**
     * Get layer names
     */
    std::vector<std::string> getLayerNames() const;
    
    /**
     * Get layer info by index
     */
    const LayerInfo& getLayerInfo(size_t index) const;
    
    /**
     * Get layer info by name
     */
    const LayerInfo* getLayerInfo(const std::string& name) const;
    
    /**
     * Read compressed layer data
     * 
     * @param index Layer index
     * @param output Output buffer (will be resized)
     * @return True if successful
     */
    bool readLayerData(size_t index, std::vector<uint8_t>& output);
    
    /**
     * Read compressed layer data by name
     */
    bool readLayerData(const std::string& name, std::vector<uint8_t>& output);
    
    /**
     * Validate file integrity (checksums)
     */
    bool validateIntegrity();
    
private:
    std::ifstream file_;
    bool valid_ = false;
    
    ContainerMetadata metadata_;
    std::vector<LayerInfo> layers_;
    
    bool parseHeader();
    bool parseLayerIndex();
    bool parseLayerMetadata(LayerInfo& info);
    bool validateFooter();
    uint32_t computeCRC32(const uint8_t* data, size_t size);
};

} // namespace wcodec

