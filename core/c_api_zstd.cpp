/**
 * @file c_api_zstd.cpp
 * @brief C API wrapper for Zstd encoder/decoder
 */

#include "encoder_zstd.h"
#include "decoder_zstd.h"
#include <cstdlib>
#include <cstring>

using namespace codec;

extern "C" {

// ============================================================================
// Encoder API
// ============================================================================

void* zstd_encoder_create(int compression_level) {
    try {
        return new ZstdEncoder(compression_level);
    } catch (...) {
        return nullptr;
    }
}

void zstd_encoder_destroy(void* encoder) {
    if (encoder) {
        delete static_cast<ZstdEncoder*>(encoder);
    }
}

float zstd_encoder_encode_layer(void* encoder,
                                 const int8_t* data,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint8_t** output,
                                 size_t* output_size) {
    if (!encoder || !data || !output || !output_size) {
        return -1.0f;
    }
    
    try {
        ZstdEncoder* enc = static_cast<ZstdEncoder*>(encoder);
        std::vector<uint8_t> compressed;
        
        float ratio = enc->encodeLayer(data, rows, cols, compressed);
        
        // Allocate output buffer
        *output = (uint8_t*)malloc(compressed.size());
        if (!*output) {
            return -1.0f;
        }
        
        memcpy(*output, compressed.data(), compressed.size());
        *output_size = compressed.size();
        
        return ratio;
    } catch (...) {
        return -1.0f;
    }
}

// ============================================================================
// Decoder API
// ============================================================================

void* zstd_decoder_create() {
    try {
        return new ZstdGPUDecoder();
    } catch (...) {
        return nullptr;
    }
}

void zstd_decoder_destroy(void* decoder) {
    if (decoder) {
        delete static_cast<ZstdGPUDecoder*>(decoder);
    }
}

int zstd_decoder_is_available() {
    return ZstdGPUDecoder::isAvailable() ? 1 : 0;
}

int zstd_decoder_decode_layer(void* decoder,
                               const uint8_t* compressed_data,
                               size_t compressed_size,
                               int8_t* output,
                               uint32_t* rows,
                               uint32_t* cols) {
    if (!decoder || !compressed_data || !output || !rows || !cols) {
        return 0;
    }
    
    try {
        ZstdGPUDecoder* dec = static_cast<ZstdGPUDecoder*>(decoder);
        return dec->decodeLayer(compressed_data, compressed_size, output, *rows, *cols) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int zstd_decoder_parse_header(const uint8_t* compressed_data,
                               size_t compressed_size,
                               uint32_t* rows,
                               uint32_t* cols,
                               uint32_t* uncompressed_size) {
    if (!compressed_data || !rows || !cols || !uncompressed_size) {
        return 0;
    }
    
    LayerHeaderZstd header;
    if (!ZstdGPUDecoder::parseHeader(compressed_data, compressed_size, header)) {
        return 0;
    }
    
    *rows = header.rows;
    *cols = header.cols;
    *uncompressed_size = header.uncompressed_size;
    
    return 1;
}

} // extern "C"

