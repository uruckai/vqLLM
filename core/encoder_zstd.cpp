/**
 * @file encoder_zstd.cpp
 * @brief Zstd encoder implementation using nvCOMP
 */

#include "encoder_zstd.h"
#include <zstd.h>
#include <stdexcept>
#include <cstring>

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
    // Use nvCOMP GPU compression for format compatibility
    try {
        // Upload data to GPU
        void* d_uncompressed = nullptr;
        cudaError_t err = cudaMalloc(&d_uncompressed, uncompressed_size);
        if (err == cudaSuccess) {
            err = cudaMemcpy(d_uncompressed, data, uncompressed_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                // Get compressed size (nvCOMP 5.0 API)
                // NOTE: nvcompBatchedZstdCompressOpts_t appears to be empty in nvCOMP 5.0
                nvcompBatchedZstdCompressOpts_t opts = {};
                
                size_t temp_size = 0;
                size_t max_comp_size = 0;
                
                const void* d_uncompressed_ptrs[1] = {d_uncompressed};
                size_t uncompressed_sizes[1] = {uncompressed_size};
                
                nvcompStatus_t status = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
                    uncompressed_size,
                    opts,
                    &max_comp_size
                );
                
                if (status == nvcompSuccess) {
                    // nvcompBatchedZstdCompressGetTempSizeSync signature (nvCOMP 5.0):
                    // (batch_size, max_uncompressed_chunk_bytes, opts, temp_bytes)
                    status = nvcompBatchedZstdCompressGetTempSizeSync(
                        1,  // batch_size
                        uncompressed_size,  // max_uncompressed_chunk_bytes
                        opts,
                        &temp_size
                    );
                }
                
                if (status == nvcompSuccess) {
                    // Allocate temp and output buffers
                    void* d_temp = nullptr;
                    void* d_compressed = nullptr;
                    
                    cudaMalloc(&d_temp, temp_size);
                    cudaMalloc(&d_compressed, max_comp_size);
                    
                    void* d_compressed_ptrs[1] = {d_compressed};
                    size_t compressed_sizes[1] = {0};
                    
                    // nvcompBatchedZstdCompressAsync signature (nvCOMP 5.0):
                    // (device_uncompressed_chunk_ptrs, device_uncompressed_chunk_bytes,
                    //  max_compressed_chunk_bytes, batch_size,
                    //  device_temp_ptr, temp_bytes,
                    //  device_compressed_chunk_ptrs, device_compressed_chunk_bytes,
                    //  opts, device_statuses, stream)
                    status = nvcompBatchedZstdCompressAsync(
                        d_uncompressed_ptrs,
                        uncompressed_sizes,
                        max_comp_size,
                        1,  // batch_size
                        d_temp,
                        temp_size,
                        d_compressed_ptrs,
                        compressed_sizes,
                        opts,
                        nullptr,  // device_statuses
                        0         // stream
                    );
                    
                    cudaDeviceSynchronize();
                    
                    if (status == nvcompSuccess && compressed_sizes[0] > 0) {
                        // Copy compressed data back
                        compressed_payload.resize(compressed_sizes[0]);
                        cudaMemcpy(compressed_payload.data(), d_compressed, 
                                 compressed_sizes[0], cudaMemcpyDeviceToHost);
                        compressed_size = compressed_sizes[0];
                    }
                    
                    cudaFree(d_temp);
                    cudaFree(d_compressed);
                }
            }
            cudaFree(d_uncompressed);
        }
    } catch (...) {
        // Fall back to CPU compression
        compressed_size = 0;
    }
#endif
    
    // Fallback to CPU Zstd if nvCOMP failed
    if (compressed_size == 0) {
        size_t max_compressed_size = ZSTD_compressBound(uncompressed_size);
        compressed_payload.resize(max_compressed_size);
        
        compressed_size = ZSTD_compress(
            compressed_payload.data(),
            max_compressed_size,
            data,
            uncompressed_size,
            compression_level_
        );
        
        if (ZSTD_isError(compressed_size)) {
            throw std::runtime_error(std::string("Zstd compression failed: ") + 
                                     ZSTD_getErrorName(compressed_size));
        }
        
        compressed_payload.resize(compressed_size);
    }
    
    // Build output with header
    output.clear();
    output.reserve(sizeof(LayerHeaderZstd) + compressed_size);
    
    // Write placeholder header
    size_t header_offset = output.size();
    output.resize(output.size() + sizeof(LayerHeaderZstd));
    
    // Append compressed data
    output.insert(output.end(), compressed_payload.begin(), compressed_payload.end());
    
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

