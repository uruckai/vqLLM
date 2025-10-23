#!/bin/bash
echo "=== CUDA Version Check ==="
echo ""
echo "CUDA Driver version:"
nvidia-smi | grep "CUDA Version"
echo ""
echo "CUDA Runtime version:"
nvcc --version | grep "release"
echo ""
echo "nvCOMP library version:"
strings /usr/lib/x86_64-linux-gnu/libnvcomp.so.5 | grep -i "nvcomp.*5\." | head -5
echo ""
echo "=== Checking nvCOMP headers ==="
find /usr/include -name "zstd.h" -path "*/nvcomp*" 2>/dev/null | head -3
echo ""
echo "=== Let's try a minimal nvCOMP test ==="
cat > /tmp/test_nvcomp_minimal.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>
#include <nvcomp/zstd.h>

int main() {
    printf("Testing nvCOMP 5.0 Zstd API...\n");
    
    // Try the simplest possible call
    size_t temp_size = 0;
    nvcompBatchedZstdCompressOpts_t opts;
    opts = nvcompBatchedZstdCompressDefaultOpts;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    printf("Calling GetTempSizeSync with:\n");
    printf("  batch_size=1, max_chunk=1024, max_total=1024\n");
    
    nvcompStatus_t status = nvcompBatchedZstdCompressGetTempSizeSync(
        nullptr,  // device_uncompressed_ptrs
        nullptr,  // device_uncompressed_bytes  
        1,        // batch_size
        1024,     // max_uncompressed_chunk_bytes
        opts,     // options
        &temp_size,
        1024,     // max_total_uncompressed_bytes
        stream
    );
    
    printf("Result: status=%d, temp_size=%zu\n", status, temp_size);
    
    if (status == nvcompSuccess) {
        printf("✓ nvCOMP 5.0 API works!\n");
    } else {
        printf("✗ nvCOMP 5.0 API failed with error %d\n", status);
    }
    
    cudaStreamDestroy(stream);
    return status != nvcompSuccess;
}
EOF

echo "Compiling minimal test..."
nvcc /tmp/test_nvcomp_minimal.cu -o /tmp/test_nvcomp_minimal -lnvcomp -I/usr/include/nvcomp_12
echo ""
echo "Running minimal test..."
/tmp/test_nvcomp_minimal

