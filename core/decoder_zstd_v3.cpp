/**
 * @file decoder_zstd_v3.cpp
 * @brief Zstd decoder implementation using nvCOMP 3.0.6 API
 */

#include "decoder_zstd.h"
#include "format_zstd.h"
#include <stdexcept>
#include <cstring>

#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

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

ZstdGPUDecoder::ZstdGPUDecoder() {
#ifdef USE_NVCOMP_ZSTD
    fprintf(stderr, "[DECODER] nvCOMP 3.0.6 Zstd GPU decoder initialized\n");
#else
    fprintf(stderr, "[DECODER] nvCOMP not available, using CPU decoding\n");
#endif
}

ZstdGPUDecoder::~ZstdGPUDecoder() {
}

bool ZstdGPUDecoder::isAvailable() {
#ifdef USE_NVCOMP_ZSTD
    return true;
#else
    return false;
#endif
}

bool ZstdGPUDecoder::parseHeader(const uint8_t* compressed_data, size_t compressed_size,
                                  LayerHeaderZstd& header) {
    // We're actually using ZstdLayerHeader (simplified format)
    if (compressed_size < sizeof(ZstdLayerHeader)) {
        return false;
    }
    
    ZstdLayerHeader simple_header;
    memcpy(&simple_header, compressed_data, sizeof(ZstdLayerHeader));
    
    // Validate magic number
    if (simple_header.magic != ZSTD_LAYER_MAGIC) {
        return false;
    }
    
    // Convert to LayerHeaderZstd format for API compatibility
    header.magic = simple_header.magic;
    header.version = 1;  // Default version
    header.rows = simple_header.rows;
    header.cols = simple_header.cols;
    header.uncompressed_size = simple_header.uncompressed_size;
    header.compressed_size = simple_header.payload_size;
    header.compression_level = 0;  // Unknown
    header.checksum = 0;  // Not used
    memset(header.reserved, 0, sizeof(header.reserved));
    
    return true;
}

bool ZstdGPUDecoder::decodeLayer(const uint8_t* compressed_data, size_t compressed_size,
                                  int8_t* output, uint32_t& rows, uint32_t& cols) {
    // Parse header
    if (compressed_size < sizeof(ZstdLayerHeader)) {
        return false;
    }
    
    ZstdLayerHeader header;
    std::memcpy(&header, compressed_data, sizeof(header));
    
    if (header.magic != ZSTD_LAYER_MAGIC) {
        return false;
    }
    
    rows = header.rows;
    cols = header.cols;
    const uint8_t* payload = compressed_data + sizeof(ZstdLayerHeader);
    size_t payload_size = header.payload_size;
    
#ifdef USE_NVCOMP_ZSTD
    try {
        // Get temp size for decompression (v3.0.6 API)
        size_t temp_size = 0;
        nvcompStatus_t status = nvcompBatchedZstdDecompressGetTempSize(
            1,                           // num_chunks
            header.uncompressed_size,    // max_uncompressed_chunk_size
            &temp_size);
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "[DECODER] GetTempSize failed: %d\n", status);
            throw std::runtime_error("GetTempSize failed");
        }
        
        // Allocate GPU memory
        void* d_compressed = nullptr;
        void* d_uncompressed = nullptr;
        void* d_temp = nullptr;
        
        cudaError_t err = cudaMalloc(&d_compressed, payload_size);
        if (err != cudaSuccess) throw std::runtime_error("cudaMalloc compressed failed");
        
        err = cudaMalloc(&d_uncompressed, header.uncompressed_size);
        if (err != cudaSuccess) {
            cudaFree(d_compressed);
            throw std::runtime_error("cudaMalloc uncompressed failed");
        }
        
        err = cudaMalloc(&d_temp, temp_size);
        if (err != cudaSuccess) {
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            throw std::runtime_error("cudaMalloc temp failed");
        }
        
        // Copy compressed data to GPU
        err = cudaMemcpy(d_compressed, payload, payload_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            throw std::runtime_error("cudaMemcpy H2D failed");
        }
        
        // Allocate device memory for statuses (nvCOMP requires this)
        nvcompStatus_t* d_statuses = nullptr;
        err = cudaMalloc(&d_statuses, sizeof(nvcompStatus_t));
        if (err != cudaSuccess) {
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            throw std::runtime_error("Failed to allocate device statuses");
        }
        
        // Prepare HOST arrays (v3.0.6 uses host arrays!)
        const void* h_compressed_ptrs[1] = {d_compressed};
        size_t h_compressed_sizes[1] = {payload_size};
        size_t h_uncompressed_sizes[1] = {header.uncompressed_size};
        void* h_uncompressed_ptrs[1] = {d_uncompressed};
        size_t h_actual_uncompressed_sizes[1] = {0};
        
        // Create stream
        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            cudaFree(d_statuses);
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            throw std::runtime_error("Stream creation failed");
        }
        
        // Decompress! (v3.0.6 API - includes device_statuses parameter)
        // nvcompStatus_t nvcompBatchedZstdDecompressAsync(
        //     const void* const* device_compressed_ptrs,
        //     const size_t* device_compressed_bytes,
        //     const size_t* device_uncompressed_bytes,
        //     size_t* device_actual_uncompressed_bytes,
        //     size_t batch_size,
        //     void* device_temp_ptr,
        //     size_t temp_bytes,
        //     void* const* device_uncompressed_ptrs,
        //     nvcompStatus_t* device_statuses,
        //     cudaStream_t stream);
        status = nvcompBatchedZstdDecompressAsync(
            h_compressed_ptrs,              // const void* const* (host array of device pointers)
            h_compressed_sizes,             // const size_t* (host array)
            h_uncompressed_sizes,           // const size_t* (host array) 
            h_actual_uncompressed_sizes,    // size_t* (host array - output)
            1,                              // batch_size
            d_temp,                         // device_temp_ptr
            temp_size,                      // temp_bytes
            h_uncompressed_ptrs,            // void* const* (host array of device pointers)
            d_statuses,                     // device_statuses (required!)
            stream);                        // cudaStream_t
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "[DECODER] DecompressAsync failed: %d\n", status);
            cudaStreamDestroy(stream);
            cudaFree(d_statuses);
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            throw std::runtime_error("DecompressAsync failed");
        }
        
        // Wait for decompression
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[DECODER] Stream sync failed\n");
        }
        
        // Copy result back to host
        err = cudaMemcpy(output, d_uncompressed, header.uncompressed_size, 
                        cudaMemcpyDeviceToHost);
        bool success = (err == cudaSuccess);
        
        if (success) {
            fprintf(stderr, "[DECODER] ✓ GPU decompression SUCCESS: %zu -> %u bytes\n",
                   payload_size, header.uncompressed_size);
        }
        
        // Cleanup
        cudaStreamDestroy(stream);
        cudaFree(d_statuses);
        cudaFree(d_temp);
        cudaFree(d_uncompressed);
        cudaFree(d_compressed);
        
        return success;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "[DECODER] GPU decompression failed: %s, falling back to CPU\n", e.what());
    }
#endif
    
    // CPU fallback
    fprintf(stderr, "[DECODER] Using CPU Zstd decompression...\n");
    size_t result = ZSTD_decompress(output, header.uncompressed_size, 
                                    payload, payload_size);
    
    if (ZSTD_isError(result)) {
        fprintf(stderr, "[DECODER] CPU decompression failed: %s\n", 
               ZSTD_getErrorName(result));
        return false;
    }
    
    return true;
}

void* ZstdGPUDecoder::decodeLayerToGPU(const uint8_t* compressed_data, size_t compressed_size,
                                       uint32_t& rows, uint32_t& cols) {
    // Parse header
    if (compressed_size < sizeof(ZstdLayerHeader)) {
        return nullptr;
    }
    
    ZstdLayerHeader header;
    std::memcpy(&header, compressed_data, sizeof(header));
    
    if (header.magic != ZSTD_LAYER_MAGIC) {
        return nullptr;
    }
    
    rows = header.rows;
    cols = header.cols;
    
#ifdef USE_NVCOMP_ZSTD
    const uint8_t* payload = compressed_data + sizeof(ZstdLayerHeader);
    size_t payload_size = header.payload_size;
    
    try {
        // Get temp size
        size_t temp_size = 0;
        nvcompStatus_t status = nvcompBatchedZstdDecompressGetTempSize(
            1, header.uncompressed_size, &temp_size);
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "[DECODER] GetTempSize failed: %d\n", status);
            return nullptr;
        }
        
        // Allocate GPU memory
        void* d_compressed = nullptr;
        void* d_uncompressed = nullptr;
        void* d_temp = nullptr;
        
        cudaError_t err = cudaMalloc(&d_compressed, payload_size);
        if (err != cudaSuccess) return nullptr;
        
        err = cudaMalloc(&d_uncompressed, header.uncompressed_size);
        if (err != cudaSuccess) {
            cudaFree(d_compressed);
            return nullptr;
        }
        
        err = cudaMalloc(&d_temp, temp_size);
        if (err != cudaSuccess) {
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            return nullptr;
        }
        
        // Copy compressed data to GPU
        err = cudaMemcpy(d_compressed, payload, payload_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            return nullptr;
        }
        
        // Allocate device statuses (nvCOMP requires this)
        nvcompStatus_t* d_statuses = nullptr;
        err = cudaMalloc(&d_statuses, sizeof(nvcompStatus_t));
        if (err != cudaSuccess) {
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            return nullptr;
        }
        
        // Prepare host arrays
        const void* h_compressed_ptrs[1] = {d_compressed};
        size_t h_compressed_sizes[1] = {payload_size};
        size_t h_uncompressed_sizes[1] = {header.uncompressed_size};
        void* h_uncompressed_ptrs[1] = {d_uncompressed};
        size_t h_actual_sizes[1] = {0};
        
        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            cudaFree(d_statuses);
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            return nullptr;
        }
        
        // Decompress (with device_statuses parameter)
        status = nvcompBatchedZstdDecompressAsync(
            h_compressed_ptrs, h_compressed_sizes, h_uncompressed_sizes,
            h_actual_sizes, 1, d_temp, temp_size, h_uncompressed_ptrs, 
            d_statuses, stream);  // device_statuses, stream
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "[DECODER] GPU decompress failed: %d\n", status);
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            cudaFree(d_statuses);
            cudaFree(d_temp);
            cudaFree(d_uncompressed);
            cudaFree(d_compressed);
            return nullptr;
        }
        
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(d_statuses);
        cudaFree(d_temp);
        cudaFree(d_compressed);
        
        fprintf(stderr, "[DECODER] ✓ GPU direct decode SUCCESS\n");
        return d_uncompressed;  // Return GPU pointer - caller must free!
        
    } catch (...) {
        return nullptr;
    }
#endif
    
    return nullptr;
}

} // namespace codec

