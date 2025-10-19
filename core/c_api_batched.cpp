/**
 * @file c_api_batched.cpp
 * @brief C API for batched encoder/decoder
 */

#include "encoder_batched.h"
#include "decoder_batched_cpu.h"  // Using CPU decoder (reliable fallback)
#include <cstdlib>
#include <cstring>

using namespace codec;

extern "C" {

// ====================
// Batched Encoder API
// ====================

void* batched_encoder_create(uint16_t tile_size) {
    try {
        return new BatchedEncoder(tile_size);
    } catch (...) {
        return nullptr;
    }
}

void batched_encoder_destroy(void* encoder) {
    delete static_cast<BatchedEncoder*>(encoder);
}

float batched_encoder_encode_layer(
    void* encoder,
    const int8_t* data,
    uint32_t rows,
    uint32_t cols,
    uint8_t** output,
    size_t* output_size
) {
    try {
        auto* enc = static_cast<BatchedEncoder*>(encoder);
        std::vector<uint8_t> compressed;
        
        float ratio = enc->encodeLayer(data, rows, cols, compressed);
        
        // Allocate output buffer
        *output_size = compressed.size();
        *output = static_cast<uint8_t*>(malloc(*output_size));
        if (*output == nullptr) {
            return -1.0f;
        }
        memcpy(*output, compressed.data(), *output_size);
        
        return ratio;
    } catch (...) {
        return -1.0f;
    }
}

// ====================
// Batched Decoder API
// ====================

void* batched_decoder_create() {
    try {
        return new BatchedCPUDecoder();
    } catch (...) {
        return nullptr;
    }
}

void batched_decoder_destroy(void* decoder) {
    delete static_cast<BatchedCPUDecoder*>(decoder);
}

float batched_decoder_decode_layer(
    void* decoder,
    const uint8_t* compressed,
    size_t compressed_size,
    int8_t* output
) {
    try {
        auto* dec = static_cast<BatchedCPUDecoder*>(decoder);
        std::vector<uint8_t> compressed_vec(compressed, compressed + compressed_size);
        
        return dec->decodeLayer(compressed_vec, output);
    } catch (...) {
        return -1.0f;
    }
}

bool batched_decoder_is_available() {
    return BatchedCPUDecoder::isAvailable();
}

// Memory management
void batched_free_buffer(uint8_t* buffer) {
    free(buffer);
}

} // extern "C"

