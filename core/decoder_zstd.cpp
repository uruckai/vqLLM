/**
 * @file decoder_zstd.cpp
 * @brief GPU-accelerated Zstd decoder implementation
 */

#include "decoder_zstd.h"
#include <zstd.h>
#include <cstring>
#include <stdexcept>

#ifdef NVCOMP_AVAILABLE
#include <cuda_runtime.h>
// nvCOMP v4.x uses different headers
#if __has_include(<nvcomp/zstd.h>)
    #include <nvcomp/zstd.h>
#else
    #include <nvcomp.h>
#endif
#endif

namespace codec {

struct ZstdGPUDecoder::Impl {
#ifdef NVCOMP_AVAILABLE
    void* d_temp_buffer;
    size_t temp_buffer_size;
    
    Impl() : d_temp_buffer(nullptr), temp_buffer_size(0) {}
    
    ~Impl() {
        if (d_temp_buffer) {
            cudaFree(d_temp_buffer);
        }
    }
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
            // Allocate GPU memory for input and output
            void* d_compressed;
            void* d_decompressed;
            
            cudaError_t err = cudaMalloc(&d_compressed, payload_size);
            if (err != cudaSuccess) {
                // Fall back to CPU
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
            
            // Get decompression temp size (v4.x API: max_chunk_size, batch_size, temp_size)
            size_t temp_size;
            
            // nvCOMP 5.0 API
            nvcompBatchedZstdDecompressOpts_t opts = {};
            nvcompStatus_t status = nvcompBatchedZstdDecompressGetTempSizeAsync(
                payload_size,  // max_chunk_size
                1,             // batch_size
                opts,          // options
                &temp_size,    // temp_bytes
                0              // stream
            );
            
            if (status != nvcompSuccess) {
                cudaFree(d_compressed);
                cudaFree(d_decompressed);
                goto cpu_fallback;
            }
            
            // Allocate or reuse temp buffer
            if (temp_size > impl_->temp_buffer_size) {
                if (impl_->d_temp_buffer) {
                    cudaFree(impl_->d_temp_buffer);
                }
                cudaMalloc(&impl_->d_temp_buffer, temp_size);
                impl_->temp_buffer_size = temp_size;
            }
            
            // Prepare batch parameters
            const void* d_compressed_ptrs[1] = {d_compressed};
            size_t compressed_sizes[1] = {payload_size};
            void* d_decompressed_ptrs[1] = {d_decompressed};
            size_t decompressed_sizes[1] = {header.uncompressed_size};
            
            // Decompress on GPU (nvCOMP 5.0 API)
            status = nvcompBatchedZstdDecompressAsync(
                d_compressed_ptrs,
                compressed_sizes,
                decompressed_sizes,
                nullptr,  // actual_decompressed_sizes
                1,  // batch_size
                impl_->d_temp_buffer,
                temp_size,
                d_decompressed_ptrs,
                opts,  // options struct
                nullptr,  // statuses_out
                0  // stream
            );
            
            if (status != nvcompSuccess) {
                cudaFree(d_compressed);
                cudaFree(d_decompressed);
                goto cpu_fallback;
            }
            
            // Wait for completion
            cudaDeviceSynchronize();
            
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
        // Allocate GPU memory for input and output
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
        
        // Get decompression temp size
        // nvCOMP 5.0: First param is MAX COMPRESSED SIZE (not uncompressed!)
        nvcompBatchedZstdDecompressOpts_t opts = {};
        size_t temp_size;
        nvcompStatus_t status = nvcompBatchedZstdDecompressGetTempSizeAsync(
            payload_size,              // max_compressed_chunk_size
            1,                         // batch_size
            opts,                      // options
            &temp_size,
            0                          // stream
        );
        
        fprintf(stderr, "DEBUG: nvCOMP temp_size=%zu (for compressed_size=%zu)\n", temp_size, payload_size);
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "ERROR: nvcompBatchedZstdDecompressGetTempSize failed: %d\n", status);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        fprintf(stderr, "DEBUG: nvCOMP temp_size=%zu (for uncompressed_size=%u)\n", temp_size, header.uncompressed_size);
        
        // Allocate or reuse temp buffer
        if (temp_size > impl_->temp_buffer_size) {
            if (impl_->d_temp_buffer) {
                cudaFree(impl_->d_temp_buffer);
            }
            err = cudaMalloc(&impl_->d_temp_buffer, temp_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "ERROR: cudaMalloc temp buffer failed: %s\n", cudaGetErrorString(err));
                cudaFree(d_compressed);
                cudaFree(d_decompressed);
                return nullptr;
            }
            impl_->temp_buffer_size = temp_size;
        }
        
        // Prepare batch parameters - arrays must be on GPU!
        const void** d_compressed_ptrs_gpu = nullptr;
        void** d_decompressed_ptrs_gpu = nullptr;
        size_t* compressed_sizes_gpu = nullptr;
        size_t* decompressed_sizes_gpu = nullptr;
        
        // Allocate GPU memory for pointer arrays
        err = cudaMalloc(&d_compressed_ptrs_gpu, sizeof(void*));
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMalloc d_compressed_ptrs_gpu failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        err = cudaMalloc(&d_decompressed_ptrs_gpu, sizeof(void*));
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMalloc d_decompressed_ptrs_gpu failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_compressed_ptrs_gpu);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        err = cudaMalloc(&compressed_sizes_gpu, sizeof(size_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMalloc compressed_sizes_gpu failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_decompressed_ptrs_gpu);
            cudaFree(d_compressed_ptrs_gpu);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        err = cudaMalloc(&decompressed_sizes_gpu, sizeof(size_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMalloc decompressed_sizes_gpu failed: %s\n", cudaGetErrorString(err));
            cudaFree(compressed_sizes_gpu);
            cudaFree(d_decompressed_ptrs_gpu);
            cudaFree(d_compressed_ptrs_gpu);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        // Copy pointer arrays to GPU
        err = cudaMemcpy(d_compressed_ptrs_gpu, &d_compressed, sizeof(void*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMemcpy d_compressed_ptrs failed: %s\n", cudaGetErrorString(err));
            cudaFree(decompressed_sizes_gpu);
            cudaFree(compressed_sizes_gpu);
            cudaFree(d_decompressed_ptrs_gpu);
            cudaFree(d_compressed_ptrs_gpu);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        err = cudaMemcpy(d_decompressed_ptrs_gpu, &d_decompressed, sizeof(void*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMemcpy d_decompressed_ptrs failed: %s\n", cudaGetErrorString(err));
            cudaFree(decompressed_sizes_gpu);
            cudaFree(compressed_sizes_gpu);
            cudaFree(d_decompressed_ptrs_gpu);
            cudaFree(d_compressed_ptrs_gpu);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        err = cudaMemcpy(compressed_sizes_gpu, &payload_size, sizeof(size_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMemcpy compressed_sizes failed: %s\n", cudaGetErrorString(err));
            cudaFree(decompressed_sizes_gpu);
            cudaFree(compressed_sizes_gpu);
            cudaFree(d_decompressed_ptrs_gpu);
            cudaFree(d_compressed_ptrs_gpu);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        size_t uncompressed_size_val = header.uncompressed_size;
        err = cudaMemcpy(decompressed_sizes_gpu, &uncompressed_size_val, sizeof(size_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMemcpy decompressed_sizes failed: %s\n", cudaGetErrorString(err));
            cudaFree(decompressed_sizes_gpu);
            cudaFree(compressed_sizes_gpu);
            cudaFree(d_decompressed_ptrs_gpu);
            cudaFree(d_compressed_ptrs_gpu);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        fprintf(stderr, "DEBUG: Calling nvcompBatchedZstdDecompressAsync...\n");
        
        // Decompress on GPU (nvCOMP 5.0 API)
        status = nvcompBatchedZstdDecompressAsync(
            d_compressed_ptrs_gpu,
            compressed_sizes_gpu,
            decompressed_sizes_gpu,
            nullptr,
            1,
            impl_->d_temp_buffer,
            temp_size,
            d_decompressed_ptrs_gpu,
            opts,  // options struct
            nullptr,  // statuses_out
            0
        );
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "ERROR: nvcompBatchedZstdDecompressAsync failed: %d\n", status);
            cudaFree(decompressed_sizes_gpu);
            cudaFree(compressed_sizes_gpu);
            cudaFree(d_decompressed_ptrs_gpu);
            cudaFree(d_compressed_ptrs_gpu);
            cudaFree(d_compressed);
            cudaFree(d_decompressed);
            return nullptr;
        }
        
        // Wait for completion
        fprintf(stderr, "DEBUG: Waiting for GPU...\n");
        cudaDeviceSynchronize();
        
        fprintf(stderr, "DEBUG: Success! Returning GPU pointer 0x%p\n", d_decompressed);
        
        // Free temp GPU arrays and compressed buffer (no longer needed)
        cudaFree(decompressed_sizes_gpu);
        cudaFree(compressed_sizes_gpu);
        cudaFree(d_decompressed_ptrs_gpu);
        cudaFree(d_compressed_ptrs_gpu);
        cudaFree(d_compressed);
        
        // Return GPU pointer (caller must free with cudaFree)
        return d_decompressed;
        
    } catch (...) {
        fprintf(stderr, "ERROR: Exception caught\n");
        return nullptr;
    }
#else
    fprintf(stderr, "ERROR: NVCOMP not available at compile time\n");
    // No GPU support
    return nullptr;
#endif
}

} // namespace codec

