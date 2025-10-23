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
    #include <nvcomp/nvcomp.h>
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
        fprintf(stderr, "[ENCODER] Starting nvCOMP GPU compression for %u bytes\n", uncompressed_size);
        
        // Upload data to GPU
        void* d_uncompressed = nullptr;
        cudaError_t err = cudaMalloc(&d_uncompressed, uncompressed_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] cudaMalloc failed: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA malloc failed");
        }
        
        err = cudaMemcpy(d_uncompressed, data, uncompressed_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA memcpy failed");
        }
        
        fprintf(stderr, "[ENCODER] Data uploaded to GPU\n");
        
        // Get cap on compressed size (nvCOMP 5.0 API)
        nvcompBatchedZstdCompressOpts_t opts = nvcompBatchedZstdCompressDefaultOpts;

        nvcompAlignmentRequirements_t align_req{};
        nvcompStatus_t status_align = nvcompBatchedZstdCompressGetRequiredAlignments(opts, &align_req);
        if (status_align == nvcompSuccess) {
            fprintf(stderr, "[ENCODER] Alignment requirements: input=%zu, temp=%zu, output=%zu\n",
                   align_req.input, align_req.temp, align_req.output);
        } else {
            fprintf(stderr, "[ENCODER] WARNING: Failed to query alignments: %d\n", status_align);
        }

        size_t temp_size = 0;
        size_t max_comp_size = 0;

        // Get max compressed size first
        nvcompStatus_t status = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
            uncompressed_size,
            opts,
            &max_comp_size
        );

        if (status != nvcompSuccess) {
            fprintf(stderr, "[ENCODER] GetMaxOutputChunkSize failed: %d\n", status);
            cudaFree(d_uncompressed);
            throw std::runtime_error("GetMaxOutputChunkSize failed");
        }

        fprintf(stderr, "[ENCODER] Max compressed size: %zu\n", max_comp_size);
        
        // Create a CUDA stream
        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_uncompressed);
            throw std::runtime_error("Stream creation failed");
        }
        fprintf(stderr, "[ENCODER] Created CUDA stream: %p\n", stream);

        // Try the ASYNC version which doesn't need device pointers!
        // From the header: nvcompBatchedZstdCompressGetTempSizeAsync takes:
        // (num_chunks, max_uncompressed_chunk_bytes, opts, temp_bytes, max_total_uncompressed_bytes)
        fprintf(stderr, "[ENCODER] ========================================\n");
        fprintf(stderr, "[ENCODER] Trying GetTempSizeAsync (simpler API)\n");
        fprintf(stderr, "[ENCODER] ========================================\n");
        fprintf(stderr, "[ENCODER]   num_chunks: 1\n");
        fprintf(stderr, "[ENCODER]   max_uncompressed_chunk_bytes: %u\n", uncompressed_size);
        fprintf(stderr, "[ENCODER]   compress_opts: {reserved[0]=%d}\n", (int)opts.reserved[0]);
        fprintf(stderr, "[ENCODER]   max_total_uncompressed_bytes: %u\n", uncompressed_size);
        fprintf(stderr, "[ENCODER] ========================================\n");
        
        // Use the Async version which doesn't need actual data pointers
        status = nvcompBatchedZstdCompressGetTempSizeAsync(
            1,                  // num_chunks
            uncompressed_size,  // max_uncompressed_chunk_bytes
            opts,               // compress_opts
            &temp_size,         // temp_bytes (output)
            uncompressed_size   // max_total_uncompressed_bytes
        );

        fprintf(stderr, "[ENCODER] ========================================\n");
        fprintf(stderr, "[ENCODER] GetTempSizeAsync RESULT:\n");
        fprintf(stderr, "[ENCODER]   status = %d", status);
        switch(status) {
            case 0: fprintf(stderr, " (nvcompSuccess)"); break;
            case 1: fprintf(stderr, " (nvcompErrorInvalidValue)"); break;
            case 2: fprintf(stderr, " (nvcompErrorNotSupported)"); break;
            case 3: fprintf(stderr, " (nvcompErrorCannotDecompress)"); break;
            case 4: fprintf(stderr, " (nvcompErrorBadChecksum)"); break;
            case 10: fprintf(stderr, " (nvcompErrorInvalidValue - CHECK PARAMS!)"); break;
            default: fprintf(stderr, " (UNKNOWN ERROR)"); break;
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "[ENCODER]   temp_size = %zu bytes", temp_size);
        if (temp_size > 0) {
            fprintf(stderr, " (%.2f MB)", temp_size / 1024.0 / 1024.0);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "[ENCODER] ========================================\n");
        
        // Try to get more error info from CUDA
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] CUDA error after GetTempSize: %s\n", cudaGetErrorString(cuda_err));
        }
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "[ENCODER] GetTempSizeAsync failed: %d\n", status);
            cudaStreamDestroy(stream);
            cudaFree(d_uncompressed);
            throw std::runtime_error("GetTempSizeAsync failed");
        }

        fprintf(stderr, "[ENCODER] Temp size: %zu bytes\n", temp_size);

        // Allocate temp and output buffers; ensure alignment requirements are respected
        size_t aligned_temp_size = temp_size;
        if (align_req.temp > 1) {
            aligned_temp_size = ((temp_size + align_req.temp - 1) / align_req.temp) * align_req.temp;
        }

        void* d_temp_raw = nullptr;
        void* d_compressed_raw = nullptr;
        void* d_temp = nullptr;
        void* d_compressed = nullptr;

        err = cudaMalloc(&d_temp_raw, aligned_temp_size + align_req.temp);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] cudaMalloc d_temp failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA malloc failed");
        }

        // Align the pointer manually if needed
        uintptr_t temp_addr = reinterpret_cast<uintptr_t>(d_temp_raw);
        temp_addr = (temp_addr + align_req.temp - 1) & ~(static_cast<uintptr_t>(align_req.temp) - 1);
        d_temp = reinterpret_cast<void*>(temp_addr);

        size_t aligned_max_comp_size = max_comp_size;
        if (align_req.output > 1) {
            aligned_max_comp_size = ((max_comp_size + align_req.output - 1) / align_req.output) * align_req.output;
        }

        err = cudaMalloc(&d_compressed_raw, aligned_max_comp_size + align_req.output);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] cudaMalloc d_compressed failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_temp_raw);
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA malloc failed");
        }

        uintptr_t comp_addr = reinterpret_cast<uintptr_t>(d_compressed_raw);
        comp_addr = (comp_addr + align_req.output - 1) & ~(static_cast<uintptr_t>(align_req.output) - 1);
        d_compressed = reinterpret_cast<void*>(comp_addr);

        fprintf(stderr, "[ENCODER] Allocated temp (%zu aligned) and compressed (%zu aligned) buffers\n",
                aligned_temp_size, aligned_max_comp_size);

        // Prepare pointer/size arrays in device memory for actual compression call
        void** d_uncompressed_ptrs = nullptr;
        size_t* d_uncompressed_sizes = nullptr;
        void** d_compressed_ptrs = nullptr;
        size_t* d_compressed_sizes = nullptr;
        
        // Allocate device arrays for pointers and sizes
        err = cudaMalloc(&d_uncompressed_ptrs, sizeof(void*));
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] cudaMalloc d_uncompressed_ptrs failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_temp_raw);
            cudaFree(d_compressed_raw);
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA malloc failed");
        }

        err = cudaMalloc(&d_uncompressed_sizes, sizeof(size_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] cudaMalloc d_uncompressed_sizes failed: %s\n", cudaGetErrorString(err));
            cudaFree((void*)d_uncompressed_ptrs);
            cudaFree(d_temp_raw);
            cudaFree(d_compressed_raw);
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA malloc failed");
        }

        err = cudaMalloc(&d_compressed_ptrs, sizeof(void*));
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] cudaMalloc d_compressed_ptrs failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_uncompressed_sizes);
            cudaFree((void*)d_uncompressed_ptrs);
            cudaFree(d_temp_raw);
            cudaFree(d_compressed_raw);
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA malloc failed");
        }

        err = cudaMalloc(&d_compressed_sizes, sizeof(size_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] cudaMalloc d_compressed_sizes failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_compressed_ptrs);
            cudaFree(d_uncompressed_sizes);
            cudaFree((void*)d_uncompressed_ptrs);
            cudaFree(d_temp_raw);
            cudaFree(d_compressed_raw);
            cudaFree(d_uncompressed);
            throw std::runtime_error("CUDA malloc failed");
        }

        const void* h_uncompressed_ptrs_device[1] = {d_uncompressed};
        size_t h_uncompressed_sizes_device[1] = {uncompressed_size};
        void* h_compressed_ptrs[1] = {d_compressed};
        size_t h_compressed_sizes[1] = {0};

        err = cudaMemcpy(d_uncompressed_ptrs, h_uncompressed_ptrs_device, sizeof(void*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] ⚠ cudaMemcpy d_uncompressed_ptrs failed: %s\n", cudaGetErrorString(err));
        }
        err = cudaMemcpy(d_uncompressed_sizes, h_uncompressed_sizes_device, sizeof(size_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] ⚠ cudaMemcpy d_uncompressed_sizes failed: %s\n", cudaGetErrorString(err));
        }
        err = cudaMemcpy(d_compressed_ptrs, h_compressed_ptrs, sizeof(void*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] ⚠ cudaMemcpy d_compressed_ptrs failed: %s\n", cudaGetErrorString(err));
        }
        err = cudaMemcpy(d_compressed_sizes, h_compressed_sizes, sizeof(size_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] ⚠ cudaMemcpy d_compressed_sizes failed: %s\n", cudaGetErrorString(err));
        }

        fprintf(stderr, "[ENCODER] ========================================\n");
        fprintf(stderr, "[ENCODER] Calling nvcompBatchedZstdCompressAsync\n");
        fprintf(stderr, "[ENCODER] ========================================\n");
        fprintf(stderr, "[ENCODER]   device_uncompressed_chunk_ptrs: %p -> {%p}\n", d_uncompressed_ptrs, d_uncompressed);
        fprintf(stderr, "[ENCODER]   device_uncompressed_chunk_bytes: %p -> {%u}\n", d_uncompressed_sizes, uncompressed_size);
        fprintf(stderr, "[ENCODER]   max_uncompressed_chunk_bytes: %u\n", uncompressed_size);
        fprintf(stderr, "[ENCODER]   num_chunks: 1\n");
        fprintf(stderr, "[ENCODER]   device_temp_ptr: %p (size: %zu)\n", d_temp, temp_size);
        fprintf(stderr, "[ENCODER]   temp_bytes: %zu\n", temp_size);
        fprintf(stderr, "[ENCODER]   device_compressed_chunk_ptrs: %p -> {%p}\n", d_compressed_ptrs, d_compressed);
        fprintf(stderr, "[ENCODER]   device_compressed_chunk_bytes: %p\n", d_compressed_sizes);
        fprintf(stderr, "[ENCODER]   compress_opts: {reserved[0]=%d}\n", (int)opts.reserved[0]);
        fprintf(stderr, "[ENCODER]   device_statuses: nullptr (no status tracking)\n");
        fprintf(stderr, "[ENCODER]   stream: %p\n", stream);
        fprintf(stderr, "[ENCODER] ========================================\n");
        
        // nvcompBatchedZstdCompressAsync signature (nvCOMP 5.0):
        // (device_uncompressed_chunk_ptrs, device_uncompressed_chunk_bytes,
        //  max_uncompressed_chunk_bytes, num_chunks, device_temp_ptr, temp_bytes,
        //  device_compressed_chunk_ptrs, device_compressed_chunk_bytes,
        //  compress_opts, device_statuses, stream)
        status = nvcompBatchedZstdCompressAsync(
            d_uncompressed_ptrs,
            d_uncompressed_sizes,
            uncompressed_size,  // max_uncompressed_chunk_bytes
            1,  // num_chunks
            d_temp,
            temp_size,
            d_compressed_ptrs,
            d_compressed_sizes,
            opts,
            nullptr,  // device_statuses
            stream    // Use the actual stream, not 0
        );
        
        fprintf(stderr, "[ENCODER] ========================================\n");
        fprintf(stderr, "[ENCODER] CompressAsync RESULT:\n");
        fprintf(stderr, "[ENCODER]   status = %d", status);
        switch(status) {
            case 0: fprintf(stderr, " (nvcompSuccess ✓)"); break;
            case 1: fprintf(stderr, " (nvcompErrorInvalidValue)"); break;
            case 2: fprintf(stderr, " (nvcompErrorNotSupported)"); break;
            case 10: fprintf(stderr, " (nvcompErrorInvalidValue - CHECK PARAMS!)"); break;
            default: fprintf(stderr, " (ERROR CODE %d)", status); break;
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "[ENCODER] ========================================\n");
        
        cudaError_t compress_cuda_err = cudaGetLastError();
        if (compress_cuda_err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] ⚠ CUDA error after CompressAsync: %s\n", cudaGetErrorString(compress_cuda_err));
        }
        
        if (status != nvcompSuccess) {
            fprintf(stderr, "[ENCODER] ✗ nvcompBatchedZstdCompressAsync FAILED\n");
            cudaFree(d_temp_raw);
            cudaFree(d_compressed_raw);
            cudaFree((void*)d_uncompressed_ptrs);
            cudaFree(d_uncompressed_sizes);
            cudaFree(d_compressed_ptrs);
            cudaFree(d_compressed_sizes);
            cudaFree(d_uncompressed);
            throw std::runtime_error("Compression failed");
        }
                    
        fprintf(stderr, "[ENCODER] Synchronizing device...\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] ⚠ cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        }
        fprintf(stderr, "[ENCODER] ✓ Compression complete, retrieving size...\n");
        
        // Copy compressed size back
        size_t actual_compressed_size = 0;
        err = cudaMemcpy(&actual_compressed_size, d_compressed_sizes, sizeof(size_t), cudaMemcpyDeviceToHost);
        
        if (err != cudaSuccess) {
            fprintf(stderr, "[ENCODER] Failed to copy compressed size: %s\n", cudaGetErrorString(err));
            cudaFree(d_temp_raw);
            cudaFree(d_compressed_raw);
            cudaFree((void*)d_uncompressed_ptrs);
            cudaFree(d_uncompressed_sizes);
            cudaFree(d_compressed_ptrs);
            cudaFree(d_compressed_sizes);
            cudaFree(d_uncompressed);
            throw std::runtime_error("Failed to retrieve size");
        }
        
        fprintf(stderr, "[ENCODER] Actual compressed size: %zu bytes\n", actual_compressed_size);
        
        if (actual_compressed_size > 0 && actual_compressed_size <= max_comp_size) {
            // Copy compressed data back
            compressed_payload.resize(actual_compressed_size);
            err = cudaMemcpy(compressed_payload.data(), d_compressed, 
                           actual_compressed_size, cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                compressed_size = actual_compressed_size;
                fprintf(stderr, "[ENCODER] ✓ nvCOMP GPU compression SUCCESS: %u -> %zu bytes (%.2fx)\n", 
                       uncompressed_size, compressed_size, (float)uncompressed_size / compressed_size);
            } else {
                fprintf(stderr, "[ENCODER] Failed to copy compressed data: %s\n", cudaGetErrorString(err));
                compressed_size = 0;
            }
        } else {
            fprintf(stderr, "[ENCODER] Invalid compressed size: %zu (max: %zu)\n", 
                   actual_compressed_size, max_comp_size);
            compressed_size = 0;
        }
        
        cudaFree(d_temp_raw);
        cudaFree(d_compressed_raw);
        cudaFree((void*)d_uncompressed_ptrs);
        cudaFree(d_uncompressed_sizes);
        cudaFree(d_compressed_ptrs);
        cudaFree(d_compressed_sizes);
        cudaFree(d_uncompressed);
        cudaStreamDestroy(stream);
    } catch (...) {
        // Fall back to CPU compression
        compressed_size = 0;
    }
#endif
    
    // Fallback to CPU Zstd if nvCOMP failed
    if (compressed_size == 0) {
        fprintf(stderr, "GPU compression failed or unavailable, using CPU Zstd fallback\n");
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

