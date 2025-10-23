# nvCOMP 5.0 Debugging Status

## Summary
nvCOMP 5.0's Zstd batched API consistently returns `nvcompErrorInvalidValue` (error 10) for **all** function calls, even with the simplest possible parameters.

## What We've Tried

### 1. Using GetTempSizeSync with Device Pointer Arrays
```c
void** d_uncompressed_ptrs;      // Device memory
size_t* d_uncompressed_sizes;    // Device memory
cudaMalloc(&d_uncompressed_ptrs, sizeof(void*));
cudaMalloc(&d_uncompressed_sizes, sizeof(size_t));
cudaMemcpy(d_uncompressed_ptrs, &d_uncompressed, ...);
cudaMemcpy(d_uncompressed_sizes, &uncompressed_size, ...);

nvcompBatchedZstdCompressGetTempSizeSync(
    d_uncompressed_ptrs,     // ✓ Device pointer array
    d_uncompressed_sizes,    // ✓ Device size array
    1,                       // num_chunks
    65536,                   // max_uncompressed_chunk_bytes
    opts,                    // {reserved[64] = {0}}
    &temp_size,             // output
    65536,                   // max_total_uncompressed_bytes
    stream                   // Valid CUDA stream
);
```
**Result:** Error 10 (nvcompErrorInvalidValue)

### 2. Using GetTempSizeAsync (Simpler API)
```c
nvcompBatchedZstdCompressGetTempSizeAsync(
    1,                       // num_chunks
    65536,                   // max_uncompressed_chunk_bytes
    opts,                    // {reserved[64] = {0}}
    &temp_size,             // output
    65536                    // max_total_uncompressed_bytes
);
```
**Result:** Error 10 (nvcompErrorInvalidValue)

### 3. Minimal Test Program
Created a standalone CUDA program that calls the exact same API - **also fails with error 10**.

## Observations

1. **All parameters are correct** according to nvCOMP 5.0 headers
2. **CUDA operations succeed** (memory allocation, copies, stream creation)
3. **Alignment requirements met** (queried via GetRequiredAlignments)
4. **Manager API also fails** with "CUDA driver version insufficient" (but driver is 12.8, runtime is 12.8 - they match!)
5. **Decompression also fails** with error 10

## Possible Root Causes

### Theory 1: nvCOMP 5.0 is Broken/Incomplete
- The RMM (RAPIDS Memory Manager) errors in version strings suggest build issues
- nvCOMP 5.0 might have been released with incomplete/broken Zstd support
- The "CUDA driver insufficient" error is spurious (versions match)

### Theory 2: Missing Initialization
- nvCOMP 5.0 might require some global initialization call we're not making
- But the headers show no such function

### Theory 3: Build Configuration Issue
- nvCOMP 5.0 was built against different CUDA toolkit version
- Library incompatibility despite matching version numbers

### Theory 4: Documentation/API Mismatch
- The headers might not match the actual library implementation
- nvCOMP 5.0 API might be in flux

## Recommendation

**Downgrade to nvCOMP 3.0.6** which:
- Has a simpler, more stable API
- Was successfully used in many projects
- Doesn't have the Manager API overhead
- Should work with our code after minor API adjustments

## Next Steps

1. Remove nvCOMP 5.0
2. Install nvCOMP 3.0.6 from NVIDIA archives
3. Update `encoder_zstd.cpp` and `decoder_zstd.cpp` to use nvCOMP 3.0 API
4. Test compression/decompression

## nvCOMP 3.0 API Reference

### Compression
```c
// Get temp size (simpler in 3.0)
nvcompBatchedZstdCompressGetTempSize(
    num_chunks,
    max_chunk_size,
    &temp_bytes
);

// Compress
nvcompBatchedZstdCompressAsync(
    device_in_ptrs,
    device_in_bytes,
    chunk_size,
    num_chunks,
    device_temp,
    temp_bytes,
    device_out_ptrs,
    device_out_bytes,
    stream
);
```

### Decompression
```c
// Get temp size
nvcompBatchedZstdDecompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_size,
    &temp_bytes
);

// Decompress
nvcompBatchedZstdDecompressAsync(
    device_in_ptrs,
    device_in_bytes,
    device_out_bytes,
    device_out_bytes_written,
    num_chunks,
    device_temp,
    temp_bytes,
    device_out_ptrs,
    stream
);
```

## Conclusion

nvCOMP 5.0 appears to have fundamental issues with its Zstd implementation. We've exhausted all reasonable debugging approaches. Time to revert to a known-working version.
