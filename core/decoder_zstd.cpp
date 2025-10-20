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
            
            // Get decompression metadata
            nvcompBatchedZstdOpts_t decompress_opts = nvcompBatchedZstdDefaultOpts;
            size_t temp_size;
            size_t metadata_size;
            
            nvcompStatus_t status = nvcompBatchedZstdDecompressGetTempSize(
                1,  // num_chunks
                payload_size,
                &temp_size,
                &metadata_size
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
            
            // Decompress on GPU (batched API for v4.x)
            status = nvcompBatchedZstdDecompressAsync(
                d_compressed_ptrs,
                compressed_sizes,
                decompressed_sizes,
                nullptr,  // actual_decompressed_sizes (optional)
                1,  // batch_size
                impl_->d_temp_buffer,
                temp_size,
                d_decompressed_ptrs,
                nullptr,  // statuses_out (optional)
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

} // namespace codec

