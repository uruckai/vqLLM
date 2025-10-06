/**
 * @file container.h
 * @brief .wcodec container format
 */

#pragma once

#include "types.h"
#include <vector>
#include <string>
#include <fstream>

namespace wcodec {

/**
 * @brief Container header
 */
struct ContainerHeader {
    uint32_t magic;           // "WCOD"
    uint16_t version_major;
    uint16_t version_minor;
    uint8_t model_hash[32];   // SHA256 of original model
    uint8_t base_quant;       // DType
    uint8_t default_tile_size;
    uint32_t num_layers;
    uint64_t layer_index_offset;
    uint8_t reserved[64];
};

/**
 * @brief Layer record in container
 */
struct LayerRecord {
    char name[256];
    uint32_t rows;
    uint32_t cols;
    uint8_t dtype;
    uint16_t num_tiles_row;
    uint16_t num_tiles_col;
    uint64_t metadata_offset;
    uint64_t data_offset;
    uint32_t compressed_size;
    uint32_t checksum;  // CRC32
};

/**
 * @brief Frequency table for entropy coding
 */
struct FrequencyTable {
    uint32_t freqs[256];
    uint32_t scale;  // Total frequency
};

/**
 * @brief Container writer
 */
class ContainerWriter {
public:
    explicit ContainerWriter(const std::string& path);
    ~ContainerWriter();

    /**
     * @brief Write container header
     */
    void writeHeader(const ContainerHeader& header);

    /**
     * @brief Begin writing a layer
     */
    void beginLayer(const LayerInfo& info);

    /**
     * @brief Write tile metadata
     */
    void writeTileMetadata(const TileMetadata& meta);

    /**
     * @brief Write tile data
     */
    void writeTileData(const std::vector<uint8_t>& data);

    /**
     * @brief Write frequency table
     */
    void writeFrequencyTable(const FrequencyTable& table);

    /**
     * @brief End current layer
     */
    void endLayer();

    /**
     * @brief Finalize container
     */
    void finalize();

private:
    std::ofstream file_;
    std::vector<LayerRecord> layers_;
    uint64_t current_offset_;
};

/**
 * @brief Container reader
 */
class ContainerReader {
public:
    explicit ContainerReader(const std::string& path);
    ~ContainerReader();

    /**
     * @brief Read container header
     */
    ContainerHeader readHeader();

    /**
     * @brief Get number of layers
     */
    size_t numLayers() const;

    /**
     * @brief Read layer info
     */
    LayerInfo readLayerInfo(size_t layer_idx);

    /**
     * @brief Read tile metadata for a layer
     */
    std::vector<TileMetadata> readTileMetadata(size_t layer_idx);

    /**
     * @brief Read tile data
     */
    std::vector<uint8_t> readTileData(size_t layer_idx, size_t tile_idx);

    /**
     * @brief Read frequency table
     */
    FrequencyTable readFrequencyTable(size_t layer_idx);

private:
    std::ifstream file_;
    ContainerHeader header_;
    std::vector<LayerRecord> layers_;
};

} // namespace wcodec

