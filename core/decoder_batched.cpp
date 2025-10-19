/**
 * @file decoder_batched.cpp
 * @brief Batched GPU decoder host implementation
 */

#include "decoder_batched.h"
#include "rans.h"  // For RANSSymbol
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

namespace codec {

// Forward declare CUDA kernel
extern "C" void launchDecodeTilesBatched(
    const uint8_t* d_tile_data,
    const TileIndexEntry* d_tile_index,
    const RANSSymbol* d_rans_table,
    int8_t* d_output,
    uint32_t tile_size,
    uint32_t output_cols,
    uint32_t num_tiles,
    cudaStream_t stream
);

BatchedGPUDecoder::BatchedGPUDecoder() : d_workspace_(nullptr), workspace_size_(0) {
    // Allocate workspace
    workspace_size_ = 256 * 1024 * 1024;  // 256 MB
    cudaMalloc(&d_workspace_, workspace_size_);
}

BatchedGPUDecoder::~BatchedGPUDecoder() {
    if (d_workspace_) {
        cudaFree(d_workspace_);
    }
}

float BatchedGPUDecoder::decodeLayer(const std::vector<uint8_t>& compressed, int8_t* output) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Parse header
    const LayerHeader* header = reinterpret_cast<const LayerHeader*>(compressed.data());
    
    if (header->magic != BATCHED_MAGIC) {
        std::cerr << "Invalid magic number" << std::endl;
        return -1.0f;
    }
    
    uint32_t num_tiles = header->num_tiles;
    uint32_t output_size = header->rows * header->cols;
    
    // Allocate GPU memory for entire layer (ONE TRANSFER!)
    uint8_t* d_compressed;
    TileIndexEntry* d_tile_index;
    RANSSymbol* d_rans_table;
    int8_t* d_output;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_compressed, compressed.size());
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for compressed data: " 
                  << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }
    
    err = cudaMalloc(&d_tile_index, num_tiles * sizeof(TileIndexEntry));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for tile index: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_compressed);
        return -1.0f;
    }
    
    err = cudaMalloc(&d_rans_table, RANS_TABLE_SIZE);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for RANS table: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_compressed);
        cudaFree(d_tile_index);
        return -1.0f;
    }
    
    err = cudaMalloc(&d_output, output_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for output: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_compressed);
        cudaFree(d_tile_index);
        cudaFree(d_rans_table);
        return -1.0f;
    }
    
    // Start timing
    cudaEventRecord(start);
    
    // Copy everything to GPU in ONE batch
    err = cudaMemcpy(d_compressed, compressed.data(), compressed.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy compressed data to GPU: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_compressed);
        cudaFree(d_tile_index);
        cudaFree(d_rans_table);
        cudaFree(d_output);
        return -1.0f;
    }
    
    // Copy RANS table
    const uint8_t* rans_table_ptr = compressed.data() + sizeof(LayerHeader);
    err = cudaMemcpy(d_rans_table, rans_table_ptr, RANS_TABLE_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy RANS table to GPU" << std::endl;
        cudaFree(d_compressed);
        cudaFree(d_tile_index);
        cudaFree(d_rans_table);
        cudaFree(d_output);
        return -1.0f;
    }
    
    // Copy tile index
    const uint8_t* tile_index_ptr = compressed.data() + header->tile_index_offset;
    err = cudaMemcpy(d_tile_index, tile_index_ptr, 
                     num_tiles * sizeof(TileIndexEntry), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy tile index to GPU" << std::endl;
        cudaFree(d_compressed);
        cudaFree(d_tile_index);
        cudaFree(d_rans_table);
        cudaFree(d_output);
        return -1.0f;
    }
    
    // Get pointer to tile data on GPU
    const uint8_t* d_tile_data = d_compressed + header->tile_data_offset;
    
    // Launch kernel: ONE BLOCK PER TILE (parallel!)
    launchDecodeTilesBatched(
        d_tile_data,
        d_tile_index,
        d_rans_table,
        d_output,
        header->tile_size,
        header->cols,
        num_tiles,
        0  // default stream
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_compressed);
        cudaFree(d_tile_index);
        cudaFree(d_rans_table);
        cudaFree(d_output);
        return -1.0f;
    }
    
    // Copy result back (ONE TRANSFER!)
    err = cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy result from GPU: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_compressed);
        cudaFree(d_tile_index);
        cudaFree(d_rans_table);
        cudaFree(d_output);
        return -1.0f;
    }
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Cleanup
    cudaFree(d_compressed);
    cudaFree(d_tile_index);
    cudaFree(d_rans_table);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

bool BatchedGPUDecoder::isAvailable() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

} // namespace codec

