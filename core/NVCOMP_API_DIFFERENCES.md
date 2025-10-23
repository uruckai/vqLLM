# nvCOMP API Differences: 3.0.6 vs 5.0

## Summary
nvCOMP 3.0.6 and 5.0 have **completely different APIs**. Our code was written for nvCOMP 5.0's batched API, which doesn't exist in 3.0.6.

## What's Available in nvCOMP 3.0.6

### ✓ Manager API (High-level, Recommended)
```cpp
#include <nvcomp/nvcompManagerFactory.hpp>

// Compression
auto manager = nvcomp::create_manager(nvcomp::ZstdManager, stream);
manager->compress(uncompressed_data, compressed_data, config);

// Decompression  
auto decomp_manager = nvcomp::create_manager(compressed_data, stream);
decomp_manager->decompress(decompressed_data, compressed_data, config);
```

### ✗ Batched API (Low-level) - NOT in 3.0.6
```cpp
// These functions DO NOT EXIST in nvCOMP 3.0.6:
nvcompBatchedZstdCompressOpts_t
nvcompBatchedZstdCompressDefaultOpts
nvcompBatchedZstdCompressGetTempSizeAsync
nvcompBatchedZstdCompressAsync
nvcompBatchedZstdDecompressOpts_t
nvcompBatchedZstdDecompressAsync
```

## What Changed in nvCOMP 5.0

### New in 5.0: Batched API
- Added low-level batched API for fine-grained control
- Requires manual memory management
- **BUT: The Zstd batched API is broken (error 10 on all calls)**

### Manager API Still Available
- Exists in both 3.0.6 and 5.0
- Higher-level, easier to use
- Handles temp buffer allocation automatically

## Our Situation

### Current Code Status
- ✗ Written for nvCOMP 5.0 batched API
- ✗ Batched API broken in nvCOMP 5.0 (error 10)
- ✗ Batched API doesn't exist in nvCOMP 3.0.6
- ✓ Manager API available in 3.0.6

## Recommended Solution: Use Manager API

### Advantages
1. **Works in nvCOMP 3.0.6** (proven stable)
2. **Simpler code** - no manual temp buffer management
3. **Format compatibility** - can compress on CPU (libzstd) and decompress on GPU
4. **Maintained by NVIDIA** - higher-level API is better supported

### Implementation Plan
1. Keep CPU compression using libzstd (works perfectly)
2. Implement GPU decompression using Manager API
3. This gives us the best of both worlds:
   - CPU: compress once offline (acceptable)
   - GPU: decompress during inference (critical path)

## Next Steps

Create new files:
- `decoder_zstd_manager.cpp/.h` - Manager API implementation
- Keep existing files for reference

The Manager API will be much simpler and actually work!

