# nvCOMP 5.0 Status Report

## Current Situation

After extensive debugging, we've identified that **nvCOMP 5.0's Zstd batched API is not working** with our current implementation:

### Compression Issues
- `nvcompBatchedZstdCompressGetTempSizeSync` returns error 10 (nvcompErrorInvalidValue)
- Parameters we're passing appear to be incorrect
- The 7th parameter in the function signature is unknown/undocumented
- CPU Zstd fallback works (produces ~65KB compressed data)

### Decompression Issues  
- Manager API fails: "CUDA driver version is insufficient for CUDA runtime version" (error 35)
- Batched API: `nvcompBatchedZstdDecompressGetTempSizeAsync` returns 454MB temp_size for 64KB data (clearly wrong)
- `nvcompBatchedZstdDecompressAsync` fails with error 10 (nvcompErrorInvalidValue)

### Format Compatibility Problem
**Critical Issue**: Using CPU libzstd for compression and nvCOMP for decompression creates a format mismatch. nvCOMP may expect a different Zstd frame format than standard libzstd produces.

## Root Causes

1. **API Documentation**: nvCOMP 5.0 documentation is incomplete or we're misunderstanding the batched API
2. **Zstd Support**: Uncertain if nvCOMP 5.0 fully supports Zstd in batched mode
3. **Driver Mismatch**: CUDA driver/runtime version mismatch preventing Manager API use

## Proposed Solutions

### Option 1: CPU-Only Zstd (RECOMMENDED)
**Use libzstd for both compression and decompression**

✅ Pros:
- Guaranteed format compatibility
- Well-documented, stable API
- Works immediately
- Simple fallback

❌ Cons:
- No GPU acceleration for decompression
- Higher CPU load during inference

### Option 2: Use LZ4 with nvCOMP
**Switch to LZ4 compression which is better supported in nvCOMP**

✅ Pros:
- LZ4 is well-supported in nvCOMP 5.0
- GPU acceleration for decompression
- Fast compression/decompression

❌ Cons:
- Lower compression ratio than Zstd (~1.5-2x vs 3-4x)
- Requires rewriting encoder/decoder
- More testing needed

### Option 3: Stick with rANS
**Continue using the working rANS implementation**

✅ Pros:
- Already working and tested
- Good compression ratios
- CPU decoder is fast enough for now

❌ Cons:
- No GPU acceleration
- More complex format

### Option 4: Wait for nvCOMP Fix/Docs
**Continue debugging nvCOMP 5.0 Zstd API**

✅ Pros:
- Best of both worlds if it works
- GPU acceleration + good compression

❌ Cons:
- Uncertain if nvCOMP 5.0 supports Zstd batched API
- Time-consuming to debug undocumented API
- May not be solvable without NVIDIA support

## Recommendation

**Go with Option 1 (CPU-Only Zstd) for now**:
1. Disable nvCOMP GPU compression/decompression in `encoder_zstd.cpp` and `decoder_zstd.cpp`
2. Use pure libzstd for both operations
3. Keep the code structure so we can swap in GPU support later if nvCOMP issues are resolved
4. Focus on getting the low-memory inference working end-to-end

**Why this makes sense**:
- The primary goal is **reducing VRAM usage**, not maximizing speed
- Decompression happens once per layer per forward pass
- With proper batching, CPU Zstd decompression overhead should be acceptable
- Format compatibility is guaranteed
- We can revisit GPU decompression after core functionality works

## Next Steps

1. Modify `encoder_zstd.cpp` and `decoder_zstd.cpp` to disable nvCOMP paths
2. Test full inference pipeline with CPU Zstd
3. Measure actual performance impact
4. If decompression is a bottleneck, consider Option 2 (LZ4) or Option 3 (rANS)

**User Decision Required**: Which option should we pursue?

