# nvCOMP v4.x API Notes

## Issue with Batched Zstd API

We're getting `nvcompErrorInvalidValue` (10) when calling `nvcompBatchedZstdDecompressAsync`.

### Problem
The batched API is complex and designed for processing many chunks at once. For our use case (single layer at a time), it's overkill and error-prone.

### Options

1. **Use non-batched API** (if it exists in v4.x)
   - Simpler for single-layer decode
   - Less metadata to manage

2. **Fix batched API usage**
   - Ensure all pointers are on GPU ✓ (we did this)
   - Check temp buffer size (454MB for 65KB seems wrong!)
   - Verify stream parameter

3. **Fall back to CPU Zstd decode**
   - Already works (we see "CPU decode successful")
   - Would mean GPU-direct decode isn't possible with nvCOMP

### Current Status
- CPU decode: ✓ Works
- GPU batched decode: ✗ Error 10 (InvalidValue)

The huge temp_size (454MB) suggests we might be calling the API wrong.

## Alternative: Use CUB or custom CUDA kernel

If nvCOMP batched API doesn't work for single chunks, we could:
1. Write a simple CUDA kernel to call standard zstd on GPU
2. Use CPU decode but with GPU-side buffers (compromise)
3. Switch back to rANS (we know it works!)

