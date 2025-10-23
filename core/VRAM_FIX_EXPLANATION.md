# VRAM Usage Fix - On-The-Fly Decompression

**Date**: October 23, 2025

## The Problem

### Original Results:
- **Baseline VRAM**: 2.06 GB
- **Compressed VRAM**: 3.86 GB (**+1.80 GB worse!** ❌)

### Root Cause:
The test was **caching all decompressed weights on GPU** instead of decompressing on-the-fly!

```python
# OLD CODE (BAD):
class CompressedLinear:
    def __init__(self):
        self._cached_weight_gpu = None  # ← Caches on GPU!
    
    def decompress_to_gpu(self):
        # Decompress once, keep forever
        self._cached_weight_gpu = weight.to('cuda')
    
    def forward(self, x):
        self.decompress_to_gpu()  # First call only
        return F.linear(x, self._cached_weight_gpu)  # Use cached
```

This caused:
1. Pre-decompression phase: Decompressed 154 layers to GPU (1.80 GB)
2. Kept them all there permanently
3. Result: **Original model (2.06 GB) + Decompressed weights (1.80 GB) = 3.86 GB**

### Why This Defeats The Purpose:
- Compressed weights in RAM: **65.8 MB**
- Decompressed weights on GPU: **1.80 GB**
- **We're storing BOTH compressed AND decompressed versions!**

## The Solution

### New Approach: True On-The-Fly Decompression

```python
# NEW CODE (GOOD):
class CompressedLinear:
    def __init__(self):
        # NO caching variables!
        pass
    
    def forward(self, x):
        # Decompress fresh every time
        weight_int8 = self.decoder.decode_layer(self.compressed)
        weight_float = weight_int8 * self.scale
        weight_tensor = torch.from_numpy(weight_float).reshape(self.shape)
        weight_gpu = weight_tensor.to(x.device)
        
        # Use it
        output = F.linear(x, weight_gpu, self.bias)
        
        # FREE IMMEDIATELY!
        del weight_gpu
        del weight_tensor
        del weight_float
        del weight_int8
        
        return output
```

### Key Changes:
1. **Removed caching**: No `_cached_weight_gpu`, no `_cached_weight_pinned`
2. **Removed pre-decompression**: Deleted the entire `[5.5/6] Decompressing layers to GPU...` section
3. **Decompress in forward()**: Every call decompresses fresh
4. **Immediate cleanup**: `del` all temporary variables immediately after use

## Expected New Results

### VRAM Usage:
- **Baseline**: 2.06 GB (full model)
- **Compressed**: Should be **< 1 GB!**
  - Original uncompressed layers: ~1.85 GB (135/155 layers)
  - **Only 1 compressed layer decompressed at a time**: ~11 MB peak
  - Compressed weights stay in RAM: 65.8 MB

### Memory Pattern (Per Layer):
```
1. Decompress:  0 MB → 11 MB (weight on GPU)
2. Compute:     11 MB (forward pass)
3. Free:        11 MB → 0 MB (immediately deleted)
4. Next layer:  Repeat
```

### Trade-offs:
**Pros:**
- ✅ Massive VRAM savings (potentially 50-70% reduction)
- ✅ Enables larger models on same hardware
- ✅ True low-memory inference

**Cons:**
- ⚠️ Slower inference (decompress overhead per layer per token)
- ⚠️ More CPU/GPU communication
- ⚠️ No benefit from weight reuse within a forward pass

## Performance Expectations

### Inference Speed:
- **Baseline**: 0.57s
- **Compressed (old cached)**: 0.19s (faster due to INT8 weights!)
- **Compressed (new on-the-fly)**: Likely **2-5x slower** than baseline
  - Each layer decompressed ~10 times (10 tokens generated)
  - 20 layers × 10 decompressions = 200 decompressions total
  - At ~0.5ms per decompress = +100ms overhead

### Why Slower?
1. **Decompress overhead**: nvCOMP GPU decode takes time (even if fast)
2. **No reuse**: Same layer decompressed multiple times per inference
3. **Memory transfers**: Constant H2D copies

### Why Still Useful?
1. **VRAM is the bottleneck**: Can't run model at all without compression
2. **Enables larger models**: 8B model on 16GB GPU, etc.
3. **Better than swapping**: In-memory compression faster than disk swapping

## Optimization Opportunities

### Future Improvements:
1. **Per-batch caching**: Cache for one forward pass, free after
2. **Smarter scheduling**: Pre-decompress next layer while computing current
3. **Better quantization**: Per-channel INT8 for quality
4. **GPU-direct decode**: Skip CPU entirely (already using nvCOMP!)
5. **Fused kernels**: Decompress + matmul in one kernel

## Test Command

On RunPod:
```bash
cd /workspace/CodecLLM && git pull && cd core && python test_zstd_inference.py
```

## Success Criteria

### Must Have:
- [ ] VRAM < 1.5 GB (vs 2.06 GB baseline)
- [ ] Inference completes without OOM
- [ ] Memory doesn't grow over time (no leaks)

### Nice to Have:
- [ ] Inference < 3x slower than baseline
- [ ] Output text reasonable (quantization OK)
- [ ] Can compress more layers (50+, 100+)

## Conclusion

The original test was **not testing low-memory inference** - it was testing "compress, decompress all, then use normally". 

The new test implements **true on-the-fly decompression** where:
- Compressed weights stay in RAM
- Only 1 layer decompressed on GPU at a time
- Immediate cleanup after each use

**This is what low-memory inference should be!** 🎯

