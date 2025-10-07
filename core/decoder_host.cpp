/**
 * @file decoder_host.cpp
 * @brief GPU decoder host implementation
 */

#include "decoder_host.h"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

namespace codec {

// Forward declare CUDA kernel
extern "C" void launchDecodeKernel(
    const uint8_t* d_compressed,
    const Header* d_header,
    const TileMetadata* d_metadata,
    int8_t* d_output,
    cudaStream_t stream
);

GPUDecoder::GPUDecoder() : d_workspace_(nullptr), workspace_size_(0) {
    // Allocate workspace (will resize as needed)
    workspace_size_ = 256 * 1024 * 1024;  // 256 MB initial
    cudaMalloc(&d_workspace_, workspace_size_);
}

GPUDecoder::~GPUDecoder() {
    if (d_workspace_) {
        cudaFree(d_workspace_);
    }
}

float GPUDecoder::decode(const std::vector<uint8_t>& compressed, int8_t* output) {
    // Parse header
    const Header* header = reinterpret_cast<const Header*>(compressed.data());
    
    if (header->magic != MAGIC) {
        std::cerr << "Invalid magic number" << std::endl;
        return -1.0f;
    }
    
    // Calculate sizes
    uint32_t num_tiles = header->num_tiles_row * header->num_tiles_col;
    size_t metadata_size = num_tiles * sizeof(TileMetadata);
    size_t output_size = header->output_rows * header->output_cols;
    
    // Allocate GPU memory
    uint8_t* d_compressed;
    Header* d_header;
    TileMetadata* d_metadata;
    int8_t* d_output;
    
    cudaMalloc(&d_compressed, compressed.size());
    cudaMalloc(&d_header, sizeof(Header));
    cudaMalloc(&d_metadata, metadata_size);
    cudaMalloc(&d_output, output_size);
    
    // Copy data to GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    cudaMemcpy(d_compressed, compressed.data(), compressed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_header, header, sizeof(Header), cudaMemcpyHostToDevice);
    cudaMemcpy(d_metadata, compressed.data() + sizeof(Header), metadata_size, cudaMemcpyHostToDevice);
    
    // Launch decode kernel
    launchDecodeKernel(d_compressed, d_header, d_metadata, d_output, 0);
    
    // Copy result back
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Cleanup
    cudaFree(d_compressed);
    cudaFree(d_header);
    cudaFree(d_metadata);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

bool GPUDecoder::isAvailable() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

} // namespace codec

