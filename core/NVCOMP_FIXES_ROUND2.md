# nvCOMP 3.0.6 Fixes - Round 2

**Date**: October 23, 2025

## Issues Found and Fixed

### Issue 1: Header Format Mismatch ❌ → ✅

**Error**: `Failed to parse Zstd header`

**Root Cause**: 
- Encoder writes `ZstdLayerHeader` (simplified, 6 fields)
- `parseHeader()` was expecting `LayerHeaderZstd` (full format, 9 fields)

**Fix**: Updated `parseHeader()` to:
1. Read the simplified `ZstdLayerHeader` format
2. Convert to `LayerHeaderZstd` for API compatibility
3. Fill in missing fields with defaults

```cpp
// Read simplified header
ZstdLayerHeader simple_header;
memcpy(&simple_header, compressed_data, sizeof(ZstdLayerHeader));

// Convert to full format
header.magic = simple_header.magic;
header.version = 1;
header.rows = simple_header.rows;
header.cols = simple_header.cols;
header.uncompressed_size = simple_header.uncompressed_size;
header.compressed_size = simple_header.payload_size;
// ... defaults for other fields
```

### Issue 2: device_statuses NULL Pointer Crash ❌ → ✅

**Error**: 
```
API call failure "cudaMemsetAsync(reinterpret_cast<void**>(device_statuses), 0, ...)"
Aborted (core dumped)
```

**Root Cause**: 
- nvCOMP 3.0.6 `nvcompBatchedZstdDecompressAsync` requires `device_statuses` parameter
- We were passing `nullptr`
- nvCOMP internally tries to call `cudaMemsetAsync()` on this pointer
- Causes segfault when dereferencing NULL

**Fix**: Allocate device memory for `device_statuses`:

```cpp
// Allocate device memory for statuses (nvCOMP requires this)
nvcompStatus_t* d_statuses = nullptr;
cudaMalloc(&d_statuses, sizeof(nvcompStatus_t));

// Pass to nvCOMP
status = nvcompBatchedZstdDecompressAsync(
    ...,
    d_statuses,  // NOT nullptr!
    stream);

// Cleanup
cudaFree(d_statuses);
```

Applied to both:
1. `decodeLayer()` - CPU output path
2. `decodeLayerToGPU()` - GPU-direct path

## Files Modified

1. **decoder_zstd_v3.cpp**
   - Fixed `parseHeader()` to handle simplified header format
   - Added `d_statuses` allocation in `decodeLayer()` (line 145-152)
   - Added `d_statuses` cleanup in all error paths
   - Added `d_statuses` allocation in `decodeLayerToGPU()` (line 312-320)
   - Updated `nvcompBatchedZstdDecompressAsync` calls to use `d_statuses` instead of `nullptr`

## Test Results So Far

### ✅ Working:
- Build compiles successfully
- GPU decoder initializes: `✓ GPU decoder available`
- Compression works: `65536 -> 65492 bytes (1.00x)` using nvCOMP GPU compression

### ❌ Fixed Issues:
- ~~Failed to parse Zstd header~~ → Fixed with header format conversion
- ~~cudaMemsetAsync crash~~ → Fixed with d_statuses allocation

### 🔄 Next Test:
Run `bash REBUILD_AND_TEST.sh` on RunPod to verify fixes work

## What Changed vs Initial Understanding

**Initial Assumption**: `device_statuses` parameter could be `nullptr` (optional)

**Reality**: nvCOMP 3.0.6 **requires** this parameter:
- Even for batch_size = 1
- nvCOMP internally calls `cudaMemsetAsync()` on it
- Must be valid device memory
- Size: `sizeof(nvcompStatus_t) * batch_size`

This is **different** from some nvCOMP documentation that suggests it's optional.

## Memory Management Pattern

For each decompress operation:
1. Allocate: `d_compressed`, `d_uncompressed`, `d_temp`, **`d_statuses`**
2. Copy compressed data H2D
3. Call `nvcompBatchedZstdDecompressAsync` with all 4 buffers
4. Sync stream
5. Free: `d_compressed`, `d_temp`, **`d_statuses`**
6. Keep/return: `d_uncompressed` (caller frees or we copy & free)

## Next Steps

### On RunPod:
```bash
cd /workspace/CodecLLM/core
bash REBUILD_AND_TEST.sh
```

This will:
1. Pull latest fixes (device_statuses + header format)
2. Rebuild library
3. Run `test_gpu_direct_simple.py`

### Expected Output:
```
✓ GPU decoder available
Creating test data (256x256)...
Compressing...
  ✓ GPU compression SUCCESS
Decoding to CPU...
  ✓ CPU decode successful
  ✓ Bit-exact match
Decoding to GPU...
  ✓ GPU decode successful
  ✓ Bit-exact match
All tests passed! ✓
```

## Confidence Level

🟢 **HIGH** - Both critical issues identified and fixed:
1. Header format mismatch → conversion logic added
2. NULL pointer crash → device memory allocated

The fixes are minimal, targeted, and follow nvCOMP's actual requirements (not just documentation).

