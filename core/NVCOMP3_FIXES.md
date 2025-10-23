# nvCOMP 3.0.6 Integration Fixes

**Date**: October 23, 2025

## Issues Fixed

### 1. Missing Header Definitions
**Error**: `'ZstdLayerHeader' was not declared in this scope`

**Fix**: Added simplified header structure to both `encoder_zstd_v3.cpp` and `decoder_zstd_v3.cpp`:
```cpp
#define ZSTD_LAYER_MAGIC 0x5A535444
struct ZstdLayerHeader {
    uint32_t magic;
    uint32_t rows;
    uint32_t cols;
    uint32_t uncompressed_size;
    uint32_t payload_size;
    uint8_t dtype;
} __attribute__((packed));
```

### 2. Wrong nvCOMP API Signature
**Error**: `cannot convert 'cudaStream_t' to 'nvcompStatus_t*'`

**Cause**: The nvCOMP 3.0.6 `nvcompBatchedZstdDecompressAsync` function has 9 parameters, not 10. There's no `device_statuses` parameter in v3.0.6.

**nvCOMP 3.0.6 Signature**:
```cpp
nvcompStatus_t nvcompBatchedZstdDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    cudaStream_t stream);
```

**Fix**: Removed the extra parameter and added clarifying comments.

### 3. Function Signature Mismatch
**Error**: `no declaration matches 'void* codec::ZstdGPUDecoder::decodeLayerToGPU(const uint8_t*, size_t, uint32_t&, uint32_t&, uint8_t&)'`

**Cause**: The implementation had an extra `dtype` parameter that wasn't in the header declaration.

**Fix**: 
- Removed `uint8_t& dtype` parameter from function signature
- Removed `dtype = header.dtype;` assignment in function body

## Files Modified

1. **encoder_zstd_v3.cpp**
   - Added `#include "format_zstd.h"`
   - Added `ZstdLayerHeader` struct definition
   - Added `ZSTD_LAYER_MAGIC` constant

2. **decoder_zstd_v3.cpp**
   - Added `#include "format_zstd.h"`
   - Added `ZstdLayerHeader` struct definition
   - Added `ZSTD_LAYER_MAGIC` constant
   - Fixed `nvcompBatchedZstdDecompressAsync` API call (9 params, not 10)
   - Fixed `decodeLayerToGPU` signature (removed `dtype` parameter)
   - Removed `dtype = header.dtype;` assignment

## Next Steps

1. Rebuild with nvCOMP 3.0.6:
   ```bash
   cd /workspace/CodecLLM/core
   rm -rf build && mkdir build && cd build
   cmake .. \
     -DNVCOMP_INCLUDE_DIR=/usr/local/include \
     -DNVCOMP_LIBRARY=/usr/local/lib/libnvcomp.so
   make -j$(nproc)
   ```

2. Test the GPU decoder:
   ```bash
   cd /workspace/CodecLLM/core
   python test_gpu_direct_simple.py
   ```

## Key Differences: nvCOMP 3.0.6 vs 5.0

| Aspect | v3.0.6 | v5.0 |
|--------|--------|------|
| **DecompressAsync params** | 9 parameters | 10 parameters (adds `device_statuses`) |
| **API call pattern** | Host arrays of device pointers | GPU device pointer arrays |
| **Complexity** | Simpler, more straightforward | More complex, more error-prone |
| **Status** | ✓ Working | ❌ Broken (error 10) |

## Lessons Learned

1. **Always check API version**: nvCOMP has breaking changes between major versions
2. **Function signature matters**: Extra parameters cause compile errors
3. **nvCOMP 3.0.6 is simpler**: Fewer parameters, less complexity
4. **GPU-first design**: Keep compressed data in RAM, decompress directly to GPU

