/**
 * @file c_api.cpp
 * @brief C API wrapper for Python ctypes bindings
 */

#include "wcodec/encoder.h"
#include "wcodec/decoder.h"
#include "wcodec/gpu_decoder.h"
#include <cstring>
#include <vector>

// Export C symbols
extern "C" {

// Opaque handles
struct WCodecEncoder {
    wcodec::Encoder* encoder;
};

struct WCodecDecoder {
    wcodec::Decoder* decoder;
};

// Create/destroy encoder
WCodecEncoder* wcodec_encoder_create(size_t tile_rows, size_t tile_cols) {
    wcodec::TileConfig config;
    config.tile_rows = tile_rows;
    config.tile_cols = tile_cols;
    
    WCodecEncoder* handle = new WCodecEncoder();
    handle->encoder = new wcodec::Encoder(config);
    return handle;
}

void wcodec_encoder_destroy(WCodecEncoder* encoder) {
    if (encoder) {
        delete encoder->encoder;
        delete encoder;
    }
}

// Encode layer
int wcodec_encode_layer(
    WCodecEncoder* encoder,
    const int8_t* data,
    size_t rows,
    size_t cols,
    uint8_t** output,
    size_t* output_size,
    double* compression_ratio,
    double* encode_time_ms
) {
    if (!encoder || !data || !output || !output_size) {
        return -1;
    }
    
    try {
        std::vector<uint8_t> encoded;
        wcodec::EncodeStats stats = encoder->encoder->encodeLayer(
            data, rows, cols, encoded
        );
        
        // Allocate output buffer
        *output_size = encoded.size();
        *output = new uint8_t[*output_size];
        std::memcpy(*output, encoded.data(), *output_size);
        
        // Return stats
        if (compression_ratio) {
            *compression_ratio = stats.compression_ratio;
        }
        if (encode_time_ms) {
            *encode_time_ms = stats.encode_time_ms;
        }
        
        return 0;
    } catch (...) {
        return -1;
    }
}

// Create/destroy decoder
WCodecDecoder* wcodec_decoder_create(size_t tile_rows, size_t tile_cols) {
    wcodec::TileConfig config;
    config.tile_rows = tile_rows;
    config.tile_cols = tile_cols;
    
    WCodecDecoder* handle = new WCodecDecoder();
    handle->decoder = new wcodec::Decoder(config);
    return handle;
}

void wcodec_decoder_destroy(WCodecDecoder* decoder) {
    if (decoder) {
        delete decoder->decoder;
        delete decoder;
    }
}

// Decode layer
int wcodec_decode_layer(
    WCodecDecoder* decoder,
    const uint8_t* input,
    size_t input_size,
    size_t rows,
    size_t cols,
    int8_t* output,
    double* decode_time_ms
) {
    if (!decoder || !input || !output) {
        return -1;
    }
    
    try {
        wcodec::DecodeStats stats = decoder->decoder->decodeLayer(
            input, input_size, rows, cols, output
        );
        
        if (decode_time_ms) {
            *decode_time_ms = stats.decode_time_ms;
        }
        
        return 0;
    } catch (...) {
        return -1;
    }
}

// Free buffer allocated by encode
void wcodec_free_buffer(uint8_t* buffer) {
    delete[] buffer;
}

// GPU decoder functions
struct WCodecGPUDecoder {
    wcodec::GPUDecoder* decoder;
};

WCodecGPUDecoder* wcodec_gpu_decoder_create(size_t tile_rows, size_t tile_cols) {
    try {
        wcodec::GPUDecoderConfig config;
        config.device_id = 0;
        config.fallback_to_cpu = true;
        
        WCodecGPUDecoder* handle = new WCodecGPUDecoder();
        handle->decoder = new wcodec::GPUDecoder(config);
        return handle;
    } catch (...) {
        return nullptr;
    }
}

void wcodec_gpu_decoder_destroy(WCodecGPUDecoder* decoder) {
    if (decoder) {
        delete decoder->decoder;
        delete decoder;
    }
}

int wcodec_gpu_decode_layer(
    WCodecGPUDecoder* decoder,
    const uint8_t* input,
    size_t input_size,
    size_t rows,
    size_t cols,
    int8_t* output,
    double* decode_time_ms,
    int* used_gpu
) {
    if (!decoder || !input || !output) {
        return -1;
    }
    
    try {
        wcodec::GPUDecodeStats stats = decoder->decoder->decodeLayer(
            input, input_size, rows, cols, output
        );
        
        if (decode_time_ms) {
            *decode_time_ms = stats.total_time_ms;
        }
        
        if (used_gpu) {
            *used_gpu = decoder->decoder->isGPUAvailable() ? 1 : 0;
        }
        
        return 0;
    } catch (...) {
        return -1;
    }
}

int wcodec_gpu_is_available() {
    return wcodec::isCUDAAvailable() ? 1 : 0;
}

} // extern "C"

