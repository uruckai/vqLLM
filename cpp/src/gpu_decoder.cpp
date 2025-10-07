/**
 * @file gpu_decoder.cpp
 * @brief GPU decoder implementation
 */

#include "wcodec/gpu_decoder.h"
#include "wcodec/decoder.h"
#include <chrono>
#include <stdexcept>

#ifdef WCODEC_CUDA_ENABLED
#include <cuda_runtime.h>

// Forward declare CUDA launch functions
namespace wcodec {
namespace cuda {
    void launch_rans_decode(
        const uint8_t* d_compressed,
        const uint32_t* d_freq_tables,
        const size_t* d_tile_offsets,
        const size_t* d_tile_sizes,
        int8_t* d_residuals,
        int num_tiles,
        int tile_rows,
        int tile_cols,
        cudaStream_t stream
    );
    
    void launch_predictor_reconstruct(
        const int8_t* d_residuals,
        const uint8_t* d_predictor_modes,
        int8_t* d_output,
        int num_tiles,
        int tile_rows,
        int tile_cols,
        int output_rows,
        int output_cols,
        cudaStream_t stream
    );
} // namespace cuda
} // namespace wcodec
#endif

namespace wcodec {

// Check if CUDA is available
bool isCUDAAvailable() {
#ifdef WCODEC_CUDA_ENABLED
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

// Get CUDA device info
std::vector<CUDADeviceInfo> getCUDADevices() {
    std::vector<CUDADeviceInfo> devices;
    
#ifdef WCODEC_CUDA_ENABLED
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess) {
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                CUDADeviceInfo info;
                info.name = prop.name;
                info.total_memory = prop.totalGlobalMem;
                info.compute_capability_major = prop.major;
                info.compute_capability_minor = prop.minor;
                info.multi_processor_count = prop.multiProcessorCount;
                devices.push_back(info);
            }
        }
    }
#endif
    
    return devices;
}

// Implementation class
class GPUDecoder::Impl {
public:
    GPUDecoderConfig config;
    TileConfig tile_config;
    bool gpu_available;
    
#ifdef WCODEC_CUDA_ENABLED
    int device_id;
    std::vector<cudaStream_t> streams;
    uint8_t* d_compressed = nullptr;
    int8_t* d_residuals = nullptr;
    int8_t* d_output = nullptr;
    size_t allocated_size = 0;
#endif
    
    // Fallback CPU decoder
    std::unique_ptr<Decoder> cpu_decoder;
    
    Impl(const GPUDecoderConfig& cfg) : config(cfg) {
        tile_config.tile_rows = 16;
        tile_config.tile_cols = 16;
        
        gpu_available = isCUDAAvailable();
        
        if (gpu_available) {
#ifdef WCODEC_CUDA_ENABLED
            device_id = config.device_id;
            cudaSetDevice(device_id);
            
            // Create streams
            streams.resize(config.num_streams);
            for (auto& stream : streams) {
                cudaStreamCreate(&stream);
            }
#endif
        } else if (config.fallback_to_cpu) {
            // Initialize CPU decoder
            cpu_decoder = std::make_unique<Decoder>(tile_config);
        }
    }
    
    ~Impl() {
#ifdef WCODEC_CUDA_ENABLED
        if (gpu_available) {
            if (d_compressed) cudaFree(d_compressed);
            if (d_residuals) cudaFree(d_residuals);
            if (d_output) cudaFree(d_output);
            
            for (auto& stream : streams) {
                cudaStreamDestroy(stream);
            }
        }
#endif
    }
    
    void allocateGPUMemory(size_t required_size) {
#ifdef WCODEC_CUDA_ENABLED
        if (required_size > allocated_size) {
            if (d_compressed) cudaFree(d_compressed);
            if (d_residuals) cudaFree(d_residuals);
            if (d_output) cudaFree(d_output);
            
            cudaMalloc(&d_compressed, required_size);
            cudaMalloc(&d_residuals, required_size);
            cudaMalloc(&d_output, required_size);
            
            allocated_size = required_size;
        }
#endif
    }
};

// Constructor
GPUDecoder::GPUDecoder(const GPUDecoderConfig& config)
    : pImpl(std::make_unique<Impl>(config)) {
}

// Destructor
GPUDecoder::~GPUDecoder() = default;

// Check GPU availability
bool GPUDecoder::isGPUAvailable() const {
    return pImpl->gpu_available;
}

// Set tile size
void GPUDecoder::setTileSize(size_t tile_rows, size_t tile_cols) {
    pImpl->tile_config.tile_rows = tile_rows;
    pImpl->tile_config.tile_cols = tile_cols;
    
    if (pImpl->cpu_decoder) {
        pImpl->cpu_decoder->setTileSize(tile_rows, tile_cols);
    }
}

// Decode to GPU memory
GPUDecodeStats GPUDecoder::decodeLayerToGPU(
    const uint8_t* compressed,
    size_t compressed_size,
    size_t rows,
    size_t cols,
    int8_t* d_output
) {
    GPUDecodeStats stats;
    
    if (!pImpl->gpu_available) {
        throw std::runtime_error("GPU not available");
    }
    
#ifdef WCODEC_CUDA_ENABLED
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate GPU memory
    size_t output_size = rows * cols;
    pImpl->allocateGPUMemory(std::max(compressed_size, output_size));
    
    // Transfer compressed data to GPU
    auto transfer_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(pImpl->d_compressed, compressed, compressed_size, cudaMemcpyHostToDevice);
    auto transfer_end = std::chrono::high_resolution_clock::now();
    stats.transfer_time_ms = std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();
    
    // TODO: Parse compressed format to extract:
    // - Frequency tables
    // - Tile offsets
    // - Tile sizes
    // - Predictor modes
    
    // For now, use CPU decoder as fallback
    if (pImpl->cpu_decoder) {
        std::vector<int8_t> output_host(output_size);
        DecodeStats cpu_stats = pImpl->cpu_decoder->decodeLayer(
            compressed, compressed_size, rows, cols, output_host.data()
        );
        
        // Copy result to GPU
        cudaMemcpy(d_output, output_host.data(), output_size, cudaMemcpyHostToDevice);
        
        auto end = std::chrono::high_resolution_clock::now();
        stats.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats.decode_time_ms = cpu_stats.decode_time_ms;
        stats.bytes_decoded = output_size;
        stats.throughput_mbps = (output_size / (1024.0 * 1024.0)) / (stats.total_time_ms / 1000.0);
        
        return stats;
    }
    
    // TODO: Full GPU decode path
    // int num_tiles = ...;
    // cuda::launch_rans_decode(...);
    // cuda::launch_predictor_reconstruct(...);
    
    throw std::runtime_error("GPU decode not fully implemented yet");
#else
    throw std::runtime_error("CUDA not enabled at compile time");
#endif
}

// Decode to host memory
GPUDecodeStats GPUDecoder::decodeLayer(
    const uint8_t* compressed,
    size_t compressed_size,
    size_t rows,
    size_t cols,
    int8_t* output
) {
    // Use CPU decoder for now (GPU kernels need more integration work)
    // This provides a working baseline while we complete GPU pipeline
    if (pImpl->cpu_decoder) {
        auto start = std::chrono::high_resolution_clock::now();
        
        DecodeStats cpu_stats = pImpl->cpu_decoder->decodeLayer(
            compressed, compressed_size, rows, cols, output
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        
        GPUDecodeStats stats;
        stats.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats.decode_time_ms = cpu_stats.decode_time_ms;
        stats.bytes_decoded = rows * cols;
        stats.throughput_mbps = (stats.bytes_decoded / (1024.0 * 1024.0)) / (stats.total_time_ms / 1000.0);
        
        return stats;
    }
    
#ifdef WCODEC_CUDA_ENABLED
    if (pImpl->gpu_available) {
        // Decode to GPU, then copy back
        size_t output_size = rows * cols;
        int8_t* d_output_temp;
        cudaMalloc(&d_output_temp, output_size);
        
        GPUDecodeStats stats = decodeLayerToGPU(compressed, compressed_size, rows, cols, d_output_temp);
        
        cudaMemcpy(output, d_output_temp, output_size, cudaMemcpyDeviceToHost);
        cudaFree(d_output_temp);
        
        return stats;
    }
#endif
    
    throw std::runtime_error("No decoder available");
}

// Synchronize
void GPUDecoder::synchronize() {
#ifdef WCODEC_CUDA_ENABLED
    if (pImpl->gpu_available) {
        cudaDeviceSynchronize();
    }
#endif
}

} // namespace wcodec

