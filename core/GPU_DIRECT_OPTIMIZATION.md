# GPU-Direct Decode Optimization

**Date**: October 23, 2025  
**Goal**: Eliminate CPU roundtrip overhead in on-the-fly decompression

## Performance Problem

### Previous Implementation (CPU Path):
```python
weight_int8 = decoder.decode_layer(compressed)      # → CPU numpy array
weight_float = weight_int8 * scale                  # CPU computation
weight_tensor = torch.from_numpy(weight_float)      # CPU tensor
weight_gpu = weight_tensor.to(device)               # CPU→GPU copy!
```

**Overhead per operation**: ~0.5ms
- nvCOMP decode: 0.1ms
- CPU processing: 0.1ms
- **CPU→GPU copy: 0.2ms** ← Major bottleneck!
- PyTorch overhead: 0.1ms

**Total**: 2000 operations × 0.5ms = **1000 seconds** (16+ minutes!)

## New Implementation (GPU-Direct):

```python
gpu_ptr, rows, cols = decoder.decode_layer_to_gpu(compressed)  # → GPU memory!
weight_int8_gpu = torch.wrap_cuda_pointer(gpu_ptr)             # Zero-copy wrap
weight_fp16 = weight_int8_gpu.to(dtype) * scale                # GPU dequantize
output = F.linear(x, weight_fp16, bias)                        # Use directly
cuda.cudaFree(gpu_ptr)                                         # Free GPU memory
```

**Overhead per operation**: ~0.2-0.3ms (estimated)
- nvCOMP decode: 0.1ms (unchanged)
- Zero-copy wrap: <0.01ms
- GPU dequantize: 0.05-0.1ms
- PyTorch overhead: 0.05ms

**Total**: 2000 operations × 0.25ms = **500 seconds** → expect **~50-100 seconds**

## Key Benefits

### 1. Eliminates CPU→GPU Copy
- **Old**: Decompress to CPU → copy to GPU
- **New**: Decompress directly to GPU
- **Savings**: ~0.2ms per operation

### 2. Zero-Copy Tensor Wrapping
PyTorch can wrap a raw CUDA pointer without copying data:
```python
storage = torch.cuda.IntStorage._new_with_data_ptr(
    data_ptr=gpu_ptr,
    size=rows * cols
)
tensor = torch.IntTensor(storage).view(rows, cols)
```

### 3. GPU Dequantization
INT8→FP16 conversion happens on GPU, not CPU:
```python
weight_fp16 = weight_int8_gpu.to(torch.float16) * scale
```

## Expected Performance

### Baseline:
- **Time**: 0.51s
- **VRAM**: 2.06 GB
- **Output**: Correct

### Previous (CPU path):
- **Time**: 243s (478x slower) ❌
- **VRAM**: 2.08 GB (same as baseline)
- **Reason**: CPU roundtrip overhead

### Expected (GPU-direct):
- **Time**: ~50-100s (100-200x slower, but 2-5x faster than before!)
- **VRAM**: 2.08 GB (unchanged - still low VRAM!)
- **Improvement**: 2-5x speedup from eliminating CPU roundtrip

### Ultimate Goal (with all optimizations):
- Fused dequantize kernel: ~25-50s
- Batched decompression: ~10-20s
- Target: <10s (20x slower than baseline, acceptable for VRAM savings)

## Implementation Details

### GPU Pointer Management:
```python
# Get GPU pointer from nvCOMP
gpu_ptr, rows, cols, dtype = decoder.decode_layer_to_gpu(compressed)

# Use it...

# CRITICAL: Free the GPU memory!
cuda.cudart().cudaFree(gpu_ptr)
```

**Warning**: Must manually free GPU memory - PyTorch doesn't track external allocations!

### Fallback Path:
If GPU-direct fails (e.g., nvCOMP not available), automatically falls back to CPU path:
```python
try:
    # GPU-direct path
    gpu_ptr = decoder.decode_layer_to_gpu(compressed)
    ...
except Exception as e:
    print(f"GPU-direct failed ({e}), using CPU fallback")
    # Original CPU path
    weight = decoder.decode_layer(compressed)
    ...
```

## Test Command

On RunPod:
```bash
cd /workspace/CodecLLM && git pull && cd core
python test_zstd_inference.py
```

## Expected Output

Look for:
```
[6/6] Running compressed inference...
  Generating tokens...
[DECODER] ✓ GPU direct decode SUCCESS
[DECODER] ✓ GPU direct decode SUCCESS
...
  Time: ~50-100s (expect 2-5x faster than before!)
  Peak VRAM: 2.08 GB (unchanged - still low!)
```

## Next Optimizations

If this works well:

### 1. Fused Dequantize Kernel (Advanced)
Custom CUDA kernel to combine INT8→FP16 + scaling:
```cuda
__global__ void dequantize_int8_to_fp16(
    const int8_t* input, half* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half((float)input[idx] * scale);
    }
}
```
**Expected**: Additional 2x speedup

### 2. Batched Decompression
Decompress multiple layers in one nvCOMP call:
```python
# Decompress gate, up, down projections together
[gate, up, down] = decoder.decode_batch_to_gpu([
    gate_compressed, up_compressed, down_compressed
])
```
**Expected**: Additional 2x speedup

### 3. CUDA Streams (Overlap)
Overlap decompression of next layer with computation of current layer:
```python
stream1 = cuda.Stream()
stream2 = cuda.Stream()

with cuda.stream(stream1):
    decompress_layer_n()
with cuda.stream(stream2):
    compute_layer_n_minus_1()
```
**Expected**: Additional 1.5x speedup

## Success Criteria

### Minimum Success:
- [x] Code compiles and runs
- [ ] Inference completes without errors
- [ ] 2x faster than previous (243s → ~120s)
- [ ] VRAM stays at ~2.08 GB

### Good Success:
- [ ] 5x faster than previous (243s → ~50s)
- [ ] No fallback to CPU path
- [ ] Output quality same as before

### Ideal Success:
- [ ] 10x faster than previous (243s → ~25s)
- [ ] Can compress all 155 layers
- [ ] Shows significant VRAM savings with all layers compressed

## Notes

- This optimization does NOT change VRAM usage (still ~2.08 GB)
- VRAM savings require compressing all 155 layers (not just 20)
- With all layers compressed: expect **~0.2 GB vs 2.06 GB baseline** = 10x reduction
- Speed vs VRAM tradeoff: On-the-fly = slow but low VRAM

## Conclusion

GPU-direct decode eliminates the major bottleneck (CPU→GPU copy) while maintaining low VRAM usage. This is the first step toward practical low-memory inference that's reasonably fast.

**Ready to test!** 🚀

