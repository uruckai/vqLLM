/**
 * @file c_api.cpp
 * @brief C API wrapper for Python bindings
 */

#include "encoder.h"
#include "decoder_host.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

using namespace codec;

extern "C" {

// Encoder API
void* encoder_create(uint16_t tile_size) {
    try {
        auto* enc = new Encoder(tile_size);
        fprintf(stderr, "[C_API] encoder_create tile=%u -> %p\n", tile_size, (void*)enc);
        return enc;
    } catch (...) {
        fprintf(stderr, "[C_API] encoder_create failed\n");
        return nullptr;
    }
}

void encoder_destroy(void* encoder) {
    delete static_cast<Encoder*>(encoder);
}

float encoder_encode(void* encoder, const int8_t* data, uint32_t rows, uint32_t cols,
                    uint8_t** output, size_t* output_size) {
    fprintf(stderr, "[C_API] encoder_encode called enc=%p rows=%u cols=%u\n", encoder, rows, cols);
    fflush(stderr);
    try {
        auto* enc = static_cast<Encoder*>(encoder);
        std::vector<uint8_t> compressed;
        
        fprintf(stderr, "[C_API] calling enc->encode\n");
        fflush(stderr);
        float ratio = enc->encode(data, rows, cols, compressed);
        
        // Allocate output buffer
        *output_size = compressed.size();
        *output = static_cast<uint8_t*>(malloc(*output_size));
        memcpy(*output, compressed.data(), *output_size);
        
        return ratio;
    } catch (...) {
        return -1.0f;
    }
}

// Decoder API
void* decoder_create() {
    try {
        return new GPUDecoder();
    } catch (...) {
        return nullptr;
    }
}

void decoder_destroy(void* decoder) {
    delete static_cast<GPUDecoder*>(decoder);
}

float decoder_decode(void* decoder, const uint8_t* compressed, size_t compressed_size,
                    int8_t* output) {
    try {
        auto* dec = static_cast<GPUDecoder*>(decoder);
        std::vector<uint8_t> compressed_vec(compressed, compressed + compressed_size);
        
        return dec->decode(compressed_vec, output);
    } catch (...) {
        return -1.0f;
    }
}

bool decoder_is_available() {
    return GPUDecoder::isAvailable();
}

// Memory management
void free_buffer(uint8_t* buffer) {
    free(buffer);
}

} // extern "C"

