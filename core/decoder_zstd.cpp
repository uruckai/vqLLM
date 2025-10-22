/**
 * @file decoder_zstd_v5.cpp
 * @brief GPU-accelerated Zstd decoder for nvCOMP 5.0+
 */

#include "decoder_zstd.h"
#include <zstd.h>
#include <cstring>
#include <stdexcept>

#ifdef NVCOMP_AVAILABLE
#include <cuda_runtime.h>
#include <nvcomp/zstd.h>
#endif

namespace codec {

struct ZstdGPUDecoder::Impl {
#ifdef NVCOMP_AVAILABLE
    // nvCOMP 5.0 doesn't need persistent temp buffers per instance
    // We'll allocate on-demand
#endif
};

ZstdGPUDecoder::ZstdGPUDecoder() : impl_(new Impl()) {}

ZstdGPUDecoder::~ZstdGPUDecoder() {
    delete impl_;
}

bool ZstdGPUDecoder::isAvailable() {
#ifdef NVCOMP_AVAILABLE
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

bool ZstdGPUDecoder::parseHeader(const uint8_t* compressed_data, size_t compressed_size,
                                  LayerHeaderZstd& header) {
    if (compressed_size < sizeof(LayerHeaderZstd)) {
        return false;
    }
    
    memcpy(&header, compressed_data, sizeof(LayerHeaderZstd));
    
    // Validate magic number
    if (header.magic != ZSTD_MAGIC) {
        return false;
    }
    
    // Validate version
    if (header.version != ZSTD_VERSION) {
        return false;
    }
    
    return true;
}

bool ZstdGPUDecoder::decodeLayer(const uint8_t* compressed_data, size_t compressed_size,
                                  int8_t* output, uint32_t& rows, uint32_t& cols) {
    // Parse header
    LayerHeaderZstd header;
    if (!parseHeader(compressed_data, compressed_size, header)) {
        return false;
    }
    
    rows = header.rows;
    cols = header.cols;
    
    const uint8_t* payload = compressed_data + sizeof(LayerHeaderZstd);
    size_t payload_size = header.compressed_size;
    
#ifdef NVCOMP_AVAILABLE
    // Try GPU decompression first
    if (isAvailable()) {
        try {
            // Allocate GPU buffers
            void* d_compressed = nullptr;
            void* d_decompressed = nullptr;
            
            cudaError_t err = cudaMalloc(&d_compressed, payload_size);
            if (err != cudaSuccess) {
                goto cpu_fallback;
            }
            
            err = cudaMalloc(&d_decompressed, header.uncompressed_size);
            if (err != cudaSuccess) {
                cudaFree(d_compressed);
                goto cpu_fallback;
            }
            
            // Copy compressed data to GPU
            err = cudaMemcpy(d_compressed, payload, payload_size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(d_compressed);
                cudaFree(d_decompressed);
                goto cpu_fallback;
            }
            
            // nvCOMP 5.0: Use simple single-buffer API
            // Allocate GPU arrays for batched API (batch of 1)
            void** d_compressed_ptrs_dev = nullptr;
            size_t* d_compressed_sizes_dev = nullptr;
            void** d_decompressed_ptrs_dev = nullptr;
            size_t* d_actual_decompressed_sizes_dev = nullptr;
            nvcompStatus_t* d_statuses_dev = nullptr;
            
            cudaMalloc(&d_compressed_ptrs_dev, sizeof(void*));
            cudaMalloc(&d_compressed_sizes_dev, sizeof(size_t));
            cudaMalloc(&d_decompressed_ptrs_dev, sizeof(void*));
            cudaMalloc(&d_actual_decompressed_sizes_dev, sizeof(size_t));
            cudaMalloc(&d_statuses_dev, sizeof(nvcompStatus_t));
            
            // Copy pointer values to GPU
            cudaMemcpy(d_compressed_ptrs_dev, &d_compressed, sizeof(void*), cudaMemcpyHostToDevice);
            cudaMemcpy(d_compressed_sizes_dev, &payload_size, sizeof(size_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_decompressed_ptrs_dev, &d_decompressed, sizeof(void*), cudaMemcpyHostToDevice);
            
            size_t decompressed_size_array[1] = {header.uncompressed_size};
            
            // Get temp size
            size_t temp_size = 0;
            nvcompBatchedZstdDecompressOpts_t opts = {};  // Default options
            
            nvcompStatus_t status = nvcompBatchedZstdDecompressGetTempSizeSync(
                (const void* const*)d_compressed_ptrs_dev,
                d_compressed_sizes_dev,
                header.uncompressed_size,  // max_uncompressed_chunk_bytes
                1,                         // batch_size
                &temp_size,
                0,                         // max_temp_bytes (0 = query)
                opts,
                d_statuses_dev,
                0                          // stream
            );
            
            if (status != nvcompSuccess) {
                // Cleanup and fallback
                cudaFree(d_statuses_dev);
                cudaFree(d_actual_decompressed_sizes_dev);
                cudaFree(d_decompressed_ptrs_dev);
                cudaFree(d_compressed_sizes_dev);
                cudaFree(d_compressed_ptrs_dev);
                cudaFree(d_compressed);
                cudaFree(d_decompressed);
                goto cpu_fallback;
            }
            
            // Allocate temp buffer
            void* d_temp = nullptr;
            cudaMalloc(&d_temp, temp_size);
            
            // Decompress
            status = nvcompBatchedZstdDecompressAsync(
                (const void* const*)d_compressed_ptrs_dev,
                d_compressed_sizes_dev,
                decompressed_size_array,
                d_actual_decompressed_sizes_dev,
                1,  // batch_size
                d_temp,
                temp_size,
                (void* const*)d_decompressed_ptrs_dev,
                opts,
                d_statuses_dev,
                0  // stream
            );
            
            cudaDeviceSynchronize();
            
            // Cleanup temp buffers
            cudaFree(d_temp);
            cudaFree(d_statuses_dev);
            cudaFree(d_actual_decompressed_sizes_dev);
            cudaFree(d_decompressed_ptrs_dev);
            cudaFree(d_compressed_sizes_dev);
            cudaFree(d_compressed_ptrs_dev);
            
            if (status != nvcompSuccess) {
                cudaFree(d_compressed);
                cudaFree(d_decompressed);
                goto cpu_fallback;
            }
            
            // Copy result back to host
            err = cudaMemcpy(output, d_decompressed, header.uncompressed_size,
                            cudaMemcpyDeviceToHost);
            
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            
            if (err != cudaSuccess) {
                goto cpu_fallback;
            }
            
            return true;
        } catch (...) {
            // Fall through to CPU fallback
        }
    }
    
cpu_fallback:
#endif
    
    // CPU fallback using standard Zstd
    size_t decompressed_size = ZSTD_decompress(
        output,
        header.uncompressed_size,
        payload,
        payload_size
    );
    
    if (ZSTD_isError(decompressed_size)) {
        return false;
    }
    
    if (decompressed_size != header.uncompressed_size) {
        return false;
    }
    
    return true;
}

void* ZstdGPUDecoder::decodeLayerToGPU(const uint8_t* compressed_data, size_t compressed_size,
                                        uint32_t& rows, uint32_t& cols) {
    // Parse header
    LayerHeaderZstd header;
    if (!parseHeader(compressed_data, compressed_size, header)) {
        fprintf(stderr, "ERROR: Failed to parse header\n");
        return nullptr;
    }
    
    rows = header.rows;
    cols = header.cols;
    
    const uint8_t* payload = compressed_data + sizeof(LayerHeaderZstd);
    size_t payload_size = header.compressed_size;
    
    fprintf(stderr, "DEBUG: rows=%u, cols=%u, payload_size=%zu, uncompressed_size=%u\n",
            rows, cols, payload_size, header.uncompressed_size);
    
#ifdef NVCOMP_AVAILABLE
    // GPU decompression ONLY - no CPU fallback
    if (!isAvailable()) {
        fprintf(stderr, "ERROR: GPU not available\n");
        return nullptr;
    }
    
    try {
        // Allocate GPU buffers for compressed and decompressed data
        void* d_compressed = nullptr;
        void* d_decompressed = nullptr;
        
        cudaError_t err = cudaMalloc(&d_compressed, payload_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMalloc d_compressed failed: %s\n", cudaGetErrorString(err));
            return nullptr;
        }
        
        err = cudaMalloc(&d_decompressed, header.uncompressed_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMalloc d_decompressed failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_compressed);
            return nullptr;
        }
        
        // Copy compressed data to GPU
        err = cudaMemcpy(d_compressed, payload, payload_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        // nvCOMP 5.0: Allocate GPU arrays for batched API
        void** d_compressed_ptrs_dev = nullptr;
        size_t* d_compressed_sizes_dev = nullptr;
        void** d_decompressed_ptrs_dev = nullptr;
        size_t* d_actual_sizes_dev = nullptr;
        nvcompStatus_t* d_statuses_dev = nullptr;
        
        cudaMalloc(&d_compressed_ptrs_dev, sizeof(void*));
        cudaMalloc(&d_compressed_sizes_dev, sizeof(size_t));
        cudaMalloc(&d_decompressed_ptrs_dev, sizeof(void*));
        cudaMalloc(&d_actual_sizes_dev, sizeof(size_t));
        cudaMalloc(&d_statuses_dev, sizeof(nvcompStatus_t));
        
        // Copy values to GPU arrays
        cudaMemcpy(d_compressed_ptrs_dev, &d_compressed, sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_compressed_sizes_dev, &payload_size, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_decompressed_ptrs_dev, &d_decompressed, sizeof(void*), cudaMemcpyHostToDevice);
        
        // Get temp size
        size_t temp_size = 0;
        nvcompBatchedZstdDecompressOpts_t opts = {};
        
        fprintf(stderr, "DEBUG: Getting temp size...\n");
        
        nvcompStatus_t status = nvcompBatchedZstdDecompressGetTempSizeSync(
            (const void* const*)d_compressed_ptrs_dev,
            d_compressed_sizes_dev,
            header.uncompressed_size,
            1,
            &temp_size,
            0,
            opts,
            d_statuses_dev,
            0
        );
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "ERROR: nvcompBatchedZstdDecompressGetTempSizeSync failed: %d\n", status);
            cudaFree(d_statuses_dev);
            cudaFree(d_actual_sizes_dev);
            cudaFree(d_decompressed_ptrs_dev);
            cudaFree(d_compressed_sizes_dev);
            cudaFree(d_compressed_ptrs_dev);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        fprintf(stderr, "DEBUG: temp_size=%zu\n", temp_size);
        
        // Allocate temp buffer
        void* d_temp = nullptr;
        err = cudaMalloc(&d_temp, temp_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMalloc temp failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_statuses_dev);
            cudaFree(d_actual_sizes_dev);
            cudaFree(d_decompressed_ptrs_dev);
            cudaFree(d_compressed_sizes_dev);
            cudaFree(d_compressed_ptrs_dev);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        // Decompress
        size_t decompressed_size_array[1] = {header.uncompressed_size};
        
        fprintf(stderr, "DEBUG: Decompressing...\n");
        
        status = nvcompBatchedZstdDecompressAsync(
            (const void* const*)d_compressed_ptrs_dev,
            d_compressed_sizes_dev,
            decompressed_size_array,
            d_actual_sizes_dev,
            1,
            d_temp,
            temp_size,
            (void* const*)d_decompressed_ptrs_dev,
            opts,
            d_statuses_dev,
            0
        );
        
        cudaDeviceSynchronize();
        
        fprintf(stderr, "DEBUG: Decompress status: %d\n", status);
        
        // Cleanup temp buffers
        cudaFree(d_temp);
        cudaFree(d_statuses_dev);
        cudaFree(d_actual_sizes_dev);
        cudaFree(d_decompressed_ptrs_dev);
        cudaFree(d_compressed_sizes_dev);
        cudaFree(d_compressed_ptrs_dev);
        cudaFree(d_compressed);
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "ERROR: nvcompBatchedZstdDecompressAsync failed: %d\n", status);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        fprintf(stderr, "DEBUG: Success! Returning GPU pointer 0x%p\n", d_decompressed);
        
        // Return GPU pointer (caller must free)
        return d_decompressed;
        
    } catch (...) {
        fprintf(stderr, "ERROR: Exception caught\n");
        return nullptr;
    }
#else
    fprintf(stderr, "ERROR: NVCOMP not available at compile time\n");
    return nullptr;
#endif
}

} // namespace codec

