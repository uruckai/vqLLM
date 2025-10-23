/**
 * @file encoder_zstd_v3.cpp
 * @brief Zstd encoder implementation using nvCOMP 3.0.6 API
 */

#include "encoder_zstd.h"
#include "format_zstd.h"

extern "C" {
#include <zstd.h>
}

#include <stdexcept>
#include <cstring>

// Define simplified header for now
#define ZSTD_LAYER_MAGIC 0x5A535444
struct ZstdLayerHeader {
    uint32_t magic;
    uint32_t rows;
    uint32_t cols;
    uint32_t uncompressed_size;
    uint32_t payload_size;
    uint8_t dtype;
} __attribute__((packed));

#ifdef NVCOMP_AVAILABLE
#include <cuda_runtime.h>
#if __has_include(<nvcomp/zstd.h>)
    #include <nvcomp/zstd.h>
    #define USE_NVCOMP_ZSTD
#endif
#endif

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
    
    size_t compressed_size = 0;
    std::vector<uint8_t> compressed_payload;
    
#ifdef USE_NVCOMP_ZSTD
    // Use nvCOMP 3.0.6 GPU compression
    try {
        fprintf(stderr, "[ENCODER] Starting nvCOMP 3.0.6 GPU compression for %u bytes\n", uncompressed_size);
        
        // Upload data to GPU
        void* d_uncompressed = nullptr;
        cudaError_t err = cudaMalloc(&d_uncompressed, uncompressed_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc failed");
        }
        
        err = cudaMemcpy(d_uncompressed, data, uncompressed_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA memcpy failed");
        }
        
        // nvCOMP 3.0.6 API
        nvcompBatchedZstdOpts_t opts = nvcompBatchedZstdDefaultOpts;
        size_t temp_size = 0;
        size_t max_comp_size = 0;

        // Get max compressed size
        nvcompStatus_t status = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
            uncompressed_size, opts, &max_comp_size);
        if (status != nvcompSuccess) {
            cudaFree(d_uncompressed);
            throw std::runtime_error("GetMaxOutputChunkSize failed");
        }

        // Get temp size
        status = nvcompBatchedZstdCompressGetTempSize(
            1, uncompressed_size, opts, &temp_size);
        if (status != nvcompSuccess) {
            cudaFree(d_uncompressed);
            throw std::runtime_error("GetTempSize failed");
        }
        
        fprintf(stderr, "[ENCODER] Temp: %zu bytes, Max output: %zu bytes\n", 
                temp_size, max_comp_size);

        // Allocate temp and output buffers
        void* d_temp = nullptr;
        void* d_compressed = nullptr;
        
        err = cudaMalloc(&d_temp, temp_size);
        if (err != cudaSuccess) {
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA malloc temp failed");
        }

        err = cudaMalloc(&d_compressed, max_comp_size);
        if (err != cudaSuccess) {
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA malloc compressed failed");
        }

        // Prepare HOST pointer/size arrays (v3.0.6 uses host arrays!)
        const void* h_uncompressed_ptrs[1] = {d_uncompressed};
        size_t h_uncompressed_sizes[1] = {uncompressed_size};
        void* h_compressed_ptrs[1] = {d_compressed};
        size_t h_compressed_sizes[1] = {0};

        // Create CUDA stream
        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            cudaFree(d_compressed);
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            throw std::runtime_error("Stream creation failed");
        }

        // Compress! (v3.0.6 API)
        status = nvcompBatchedZstdCompressAsync(
            h_uncompressed_ptrs,       // const void* const* (host array of device pointers)
            h_uncompressed_sizes,      // const size_t* (host array)
            uncompressed_size,         // max_uncompressed_chunk_bytes
            1,                         // batch_size
            d_temp,                    // device_temp_ptr
            temp_size,                 // temp_bytes
            h_compressed_ptrs,         // void* const* (host array of device pointers)
            h_compressed_sizes,        // size_t* (host array - output)
            opts,                      // nvcompBatchedZstdOpts_t
            stream);                   // cudaStream_t

        if (status != nvcompSuccess) {
            fprintf(stderr, "[ENCODER] CompressAsync failed: %d\n", status);
            cudaStreamDestroy(stream);
            cudaFree(d_compressed);
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            throw std::runtime_error("CompressAsync failed");
        }

        // Wait for compression to complete
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] Stream sync failed: %s\n", cudaGetErrorString(err));
        }

        // Get compressed size from host array
        compressed_size = h_compressed_sizes[0];
        
        fprintf(stderr, "[ENCODER] Compressed size: %zu bytes\n", compressed_size);
        
        if (compressed_size > 0 && compressed_size <= max_comp_size) {
            // Copy compressed data back to host
            compressed_payload.resize(compressed_size);
            err = cudaMemcpy(compressed_payload.data(), d_compressed, 
                           compressed_size, cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                fprintf(stderr, "[ENCODER] ✓ GPU compression SUCCESS: %u -> %zu bytes (%.2fx)\n", 
                       uncompressed_size, compressed_size, 
                       (float)uncompressed_size / compressed_size);
            } else {
                fprintf(stderr, "[ENCODER] Failed to copy result\n");
                compressed_size = 0;
            }
        } else {
            compressed_size = 0;
        }
        
        // Cleanup
        cudaStreamDestroy(stream);
        cudaFree(d_compressed);
        cudaFree(d_temp);
        cudaFree(d_uncompressed);
        
    } catch (const std::exception& e) {
        fprintf(stderr, "[ENCODER] GPU compression failed: %s\n", e.what());
        compressed_size = 0;
    }
#endif
    
    // Fallback to CPU Zstd if GPU failed or not available
    if (compressed_size == 0) {
        fprintf(stderr, "[ENCODER] Using CPU Zstd compression...\n");
        
        size_t max_comp_size = ZSTD_compressBound(uncompressed_size);
        compressed_payload.resize(max_comp_size);
        
        compressed_size = ZSTD_compress(
            compressed_payload.data(), max_comp_size,
            data, uncompressed_size,
            compression_level_
        );
        
        if (ZSTD_isError(compressed_size)) {
            throw std::runtime_error("CPU Zstd compression failed");
        }
        
        compressed_payload.resize(compressed_size);
    }
    
    // Write header + compressed data
    ZstdLayerHeader header;
    header.magic = ZSTD_LAYER_MAGIC;
    header.rows = rows;
    header.cols = cols;
    header.uncompressed_size = uncompressed_size;
    header.payload_size = static_cast<uint32_t>(compressed_size);
    header.dtype = 0; // int8
    
    output.resize(sizeof(ZstdLayerHeader) + compressed_size);
    std::memcpy(output.data(), &header, sizeof(header));
    std::memcpy(output.data() + sizeof(header), compressed_payload.data(), compressed_size);
    
    float ratio = static_cast<float>(uncompressed_size) / compressed_size;
    return ratio;
}

} // namespace codec

