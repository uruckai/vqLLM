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

    cudaError_t err;
    err = cudaMalloc(&d_compressed, compressed.size());
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate compressed memory: " << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }

    err = cudaMalloc(&d_header, sizeof(Header));
    if (err != cudaSuccess) {
        cudaFree(d_compressed);
        std::cerr << "Failed to allocate header memory: " << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }

    err = cudaMalloc(&d_metadata, metadata_size);
    if (err != cudaSuccess) {
        cudaFree(d_compressed);
        cudaFree(d_header);
        std::cerr << "Failed to allocate metadata memory: " << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }

    err = cudaMalloc(&d_output, output_size);
    if (err != cudaSuccess) {
        cudaFree(d_compressed);
        cudaFree(d_header);
        cudaFree(d_metadata);
        std::cerr << "Failed to allocate output memory: " << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }

    // Copy data to GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    err = cudaMemcpy(d_compressed, compressed.data(), compressed.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy compressed data: " << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }

    err = cudaMemcpy(d_header, header, sizeof(Header), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy header: " << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }

    err = cudaMemcpy(d_metadata, compressed.data() + sizeof(Header), metadata_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy metadata: " << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }

    // Launch decode kernel
    launchDecodeKernel(d_compressed, d_header, d_metadata, d_output, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }

    // Copy result back
    err = cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy result back: " << cudaGetErrorString(err) << std::endl;
        return -1.0f;
    }

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
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    
    return true;
}

} // namespace codec

