#!/bin/bash
# Test different nvCOMP API variations to isolate the issue

echo "=== nvCOMP 5.0 API Variation Tests ==="
echo ""

cat > /tmp/test_all_variations.cu << 'EOF'
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvcomp/zstd.h>

int main() {
    printf("=== Test 1: GetTempSizeAsync with all combinations ===\n");
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    nvcompBatchedZstdCompressOpts_t opts = nvcompBatchedZstdCompressDefaultOpts;
    
    // Test various chunk sizes
    size_t test_sizes[] = {1024, 4096, 16384, 65536, 131072};
    
    for (int i = 0; i < 5; i++) {
        size_t chunk_size = test_sizes[i];
        size_t temp_size = 0;
        
        printf("\nTesting chunk_size=%zu:\n", chunk_size);
        printf("  Calling with: num_chunks=1, max_chunk=%zu, max_total=%zu\n", 
               chunk_size, chunk_size);
        
        nvcompStatus_t status = nvcompBatchedZstdCompressGetTempSizeAsync(
            1,          // num_chunks
            chunk_size, // max_uncompressed_chunk_bytes
            opts,       // compress_opts
            &temp_size, // temp_bytes (output)
            chunk_size  // max_total_uncompressed_bytes
        );
        
        printf("  Result: status=%d, temp_size=%zu\n", status, temp_size);
        
        if (status == 0) {
            printf("  ✓ SUCCESS!\n");
            break;
        }
    }
    
    printf("\n=== Test 2: Try with different num_chunks ===\n");
    for (int chunks = 1; chunks <= 4; chunks++) {
        size_t temp_size = 0;
        nvcompStatus_t status = nvcompBatchedZstdCompressGetTempSizeAsync(
            chunks,  // num_chunks
            16384,   // max_uncompressed_chunk_bytes
            opts,
            &temp_size,
            16384 * chunks  // max_total
        );
        printf("num_chunks=%d: status=%d, temp_size=%zu\n", chunks, status, temp_size);
    }
    
    printf("\n=== Test 3: Check if opts needs initialization ===\n");
    {
        size_t temp_size = 0;
        
        // Try with uninitialized opts
        nvcompBatchedZstdCompressOpts_t bad_opts;
        memset(&bad_opts, 0xFF, sizeof(bad_opts));  // Fill with garbage
        nvcompStatus_t status = nvcompBatchedZstdCompressGetTempSizeAsync(
            1, 65536, bad_opts, &temp_size, 65536
        );
        printf("Garbage opts: status=%d\n", status);
        
        // Try with explicitly zeroed opts
        nvcompBatchedZstdCompressOpts_t zero_opts;
        memset(&zero_opts, 0, sizeof(zero_opts));
        status = nvcompBatchedZstdCompressGetTempSizeAsync(
            1, 65536, zero_opts, &temp_size, 65536
        );
        printf("Zeroed opts: status=%d\n", status);
        
        // Try with default opts
        status = nvcompBatchedZstdCompressGetTempSizeAsync(
            1, 65536, nvcompBatchedZstdCompressDefaultOpts, &temp_size, 65536
        );
        printf("Default opts: status=%d\n", status);
    }
    
    printf("\n=== Test 4: Check chunk size limits ===\n");
    {
        size_t temp_size = 0;
        // Try maximum allowed (16MB per docs)
        nvcompStatus_t status = nvcompBatchedZstdCompressGetTempSizeAsync(
            1, 16*1024*1024, opts, &temp_size, 16*1024*1024
        );
        printf("16MB chunk: status=%d\n", status);
        
        // Try minimum
        status = nvcompBatchedZstdCompressGetTempSizeAsync(
            1, 1, opts, &temp_size, 1
        );
        printf("1 byte chunk: status=%d\n", status);
        
        // Try recommended (65536 per docs)
        status = nvcompBatchedZstdCompressGetTempSizeAsync(
            1, 65536, opts, &temp_size, 65536
        );
        printf("65536 chunk (recommended): status=%d\n", status);
    }
    
    printf("\n=== Summary ===\n");
    printf("If ALL tests failed with error 10, nvCOMP 5.0 Zstd batched API is broken.\n");
    printf("See NVCOMP_STATUS.md for next steps.\n");
    
    cudaStreamDestroy(stream);
    return 0;
}
EOF

echo "Compiling comprehensive test..."
nvcc /tmp/test_all_variations.cu -o /tmp/test_all_variations -lnvcomp -I/usr/include/nvcomp_12

if [ $? -eq 0 ]; then
    echo "Running comprehensive tests..."
    echo ""
    /tmp/test_all_variations
else
    echo "❌ Compilation failed"
fi

