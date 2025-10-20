/**
 * @file encoder_zstd.cpp
 * @brief Zstd encoder implementation
 */

#include "encoder_zstd.h"
#include <zstd.h>
#include <stdexcept>
#include <cstring>

namespace codec {

ZstdEncoder::ZstdEncoder(int compression_level)
    : compression_level_(compression_level) {
    if (compression_level < 1 || compression_level > 22) {
        throw std::invalid_argument("Zstd compression level must be 1-22");
    }
}

float ZstdEncoder::encodeLayer(const int8_t* data, uint32_t rows, uint32_t cols,
                                std::vector<uint8_t>& output) {
    uint32_t uncompressed_size = rows * cols;
    
    // Estimate compressed size
    size_t max_compressed_size = ZSTD_compressBound(uncompressed_size);
    
    // Allocate output buffer
    output.clear();
    output.reserve(sizeof(LayerHeaderZstd) + max_compressed_size);
    
    // Write placeholder header
    size_t header_offset = output.size();
    output.resize(output.size() + sizeof(LayerHeaderZstd));
    
    // Compress data
    size_t compressed_data_offset = output.size();
    output.resize(compressed_data_offset + max_compressed_size);
    
    size_t compressed_size = ZSTD_compress(
        output.data() + compressed_data_offset,
        max_compressed_size,
        data,
        uncompressed_size,
        compression_level_
    );
    
    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error(std::string("Zstd compression failed: ") + 
                                 ZSTD_getErrorName(compressed_size));
    }
    
    // Resize to actual compressed size
    output.resize(compressed_data_offset + compressed_size);
    
    // Calculate checksum (simple XOR for now, can upgrade to XXH64)
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < uncompressed_size; i++) {
        checksum ^= static_cast<uint32_t>(data[i]) << ((i % 4) * 8);
    }
    
    // Write header
    LayerHeaderZstd header;
    header.magic = ZSTD_MAGIC;
    header.version = ZSTD_VERSION;
    header.rows = rows;
    header.cols = cols;
    header.uncompressed_size = uncompressed_size;
    header.compressed_size = compressed_size;
    header.compression_level = compression_level_;
    header.checksum = checksum;
    memset(header.reserved, 0, sizeof(header.reserved));
    
    memcpy(output.data() + header_offset, &header, sizeof(LayerHeaderZstd));
    
    // Calculate compression ratio
    float ratio = static_cast<float>(uncompressed_size) / output.size();
    
    return ratio;
}

} // namespace codec

