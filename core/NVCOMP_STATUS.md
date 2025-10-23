# nvCOMP 5.0 Investigation Summary

## Problem
Both compression and decompression using nvCOMP 5.0's Zstd batched API consistently fail with error code 10 (`nvcompErrorInvalidValue`).

## What We Tried

### 1. GetTempSizeSync with Device Pointer Arrays ❌
- Allocated device memory for pointer and size arrays
- Copied pointers to device memory
- Passed device arrays to `nvcompBatchedZstdCompressGetTempSizeSync`
- **Result**: Error 10

### 2. GetTempSizeSync with nullptr ❌
- Attempted passing NULL for pointer arrays
- **Result**: Error 10

### 3. GetTempSizeSync with Host Pointer Arrays ❌
- Passed host-side pointer arrays
- **Result**: Error 10

### 4. GetTempSizeAsync (Simplest API) ❌
- Used the async version which only needs max sizes
- No device pointers required
- Parameters: `(num_chunks=1, max_size=65536, opts={0}, &temp_size, max_total=65536)`
- **Result**: Error 10

### 5. Minimal Standalone Test ❌
- Created a minimal C++ program outside our codebase
- Called nvCOMP API with same parameters
- **Result**: Error 10

## Evidence of nvCOMP 5.0 Issues

```
nvCOMP library version strings contain RMM errors:
- RMM failure at: .../format.hpp:60: Error during formatting.
- RMM failure at: .../pool_memory_resource.hpp:276: Maximum pool size exceeded
```

## Additional Failures

### Manager API
```
ERROR: Manager API failed: Encountered Cuda Error: 35: 
'CUDA driver version is insufficient for CUDA runtime version'.
```

This occurs despite:
- CUDA Driver: 12.8 (570.153.02)
- CUDA Runtime: 12.8.93
- Versions match perfectly!

### Decompression
`nvcompBatchedZstdDecompressAsync` also fails with error 10 using the same pattern.

## Conclusion

**nvCOMP 5.0.0.6 Zstd batched API appears to be broken or has undocumented requirements.**

Evidence:
1. ✓ All parameters are correct per documentation
2. ✓ Device memory allocations succeed
3. ✓ CUDA operations work
4. ✓ CPU Zstd compression/decompression works
5. ✓ Minimal test fails the same way
6. ✗ nvCOMP 5.0 consistently returns error 10
7. ✗ RMM errors in library version strings

## Recommendation

### Option 1: Downgrade to nvCOMP 3.0.6 ⭐ RECOMMENDED
- nvCOMP 3.0.6 has a proven track record
- The batched API was stable in 3.x
- We can test if the older version works

### Option 2: Use CPU Compression + GPU Decompression Only
- Keep using libzstd for CPU compression (works perfectly)
- Use nvCOMP Manager API for GPU decompression
- Avoid the batched API entirely

### Option 3: Contact NVIDIA
- File a bug report with nvCOMP team
- Provide minimal reproduction case
- Wait for fix

## Next Steps

**Try nvCOMP 3.0.6 first:**
```bash
# Remove nvCOMP 5
sudo apt-get remove -y nvcomp
sudo rm -rf /var/nvcomp-local-repo-ubuntu2404-5.0.0.6

# We need to find working nvCOMP 3.0.6 download
# or build from source
```
