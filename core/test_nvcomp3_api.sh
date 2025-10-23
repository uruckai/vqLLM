#!/bin/bash
echo "=== Testing nvCOMP 3.0.6 Zstd API ==="

cat > /tmp/test_nvcomp3.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>
#include <nvcomp/zstd.h>

int main() {
    printf("Testing nvCOMP 3.0.6 Zstd batched API...\n\n");
    
    // Test 1: Get temp size for compression
    printf("=== Test 1: Compression GetTempSize ===\n");
    size_t comp_temp_bytes = 0;
    nvcompBatchedZstdOpts_t opts = nvcompBatchedZstdDefaultOpts;
    
    nvcompStatus_t status = nvcompBatchedZstdCompressGetTempSize(
        1,      // batch_size
        65536,  // max_uncompressed_chunk_bytes
        opts,
        &comp_temp_bytes
    );
    
    printf("Compression GetTempSize: status=%d, temp_bytes=%zu\n", status, comp_temp_bytes);
    if (status == 0) {
        printf("  ✓ SUCCESS!\n");
    } else {
        printf("  ✗ FAILED with error %d\n", status);
    }
    
    // Test 2: Get max output size
    printf("\n=== Test 2: GetMaxOutputChunkSize ===\n");
    size_t max_output = 0;
    status = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        65536,  // max_uncompressed_chunk_bytes
        opts,
        &max_output
    );
    printf("Max output size: status=%d, size=%zu\n", status, max_output);
    if (status == 0) {
        printf("  ✓ SUCCESS!\n");
    }
    
    // Test 3: Get temp size for decompression
    printf("\n=== Test 3: Decompression GetTempSize ===\n");
    size_t decomp_temp_bytes = 0;
    status = nvcompBatchedZstdDecompressGetTempSize(
        1,      // num_chunks
        65536,  // max_uncompressed_chunk_size
        &decomp_temp_bytes
    );
    printf("Decompression GetTempSize: status=%d, temp_bytes=%zu\n", status, decomp_temp_bytes);
    if (status == 0) {
        printf("  ✓ SUCCESS!\n");
    }
    
    // Test 4: Try different chunk sizes
    printf("\n=== Test 4: Different chunk sizes ===\n");
    size_t test_sizes[] = {4096, 16384, 65536, 131072};
    for (int i = 0; i < 4; i++) {
        size_t temp = 0;
        status = nvcompBatchedZstdCompressGetTempSize(
            1, test_sizes[i], opts, &temp
        );
        printf("  chunk_size=%zu: status=%d, temp=%zu\n", test_sizes[i], status, temp);
    }
    
    printf("\n");
    if (comp_temp_bytes > 0 && decomp_temp_bytes > 0) {
        printf("========================================\n");
        printf("✓ nvCOMP 3.0.6 Zstd API is WORKING!\n");
        printf("========================================\n");
        return 0;
    } else {
        printf("========================================\n");
        printf("✗ nvCOMP 3.0.6 Zstd API failed\n");
        printf("========================================\n");
        return 1;
    }
}
EOF

echo "Compiling test..."
nvcc /tmp/test_nvcomp3.cu -o /tmp/test_nvcomp3 -lnvcomp -I/usr/local/include -L/usr/local/lib

if [ $? -eq 0 ]; then
    echo "Running test..."
    echo ""
    /tmp/test_nvcomp3
else
    echo "❌ Compilation failed"
fi

