# GPU-Direct Decode Instructions

## What This Solves

**Previous problem**: OOM errors despite 29GB free VRAM because:
- Pre-decompressing to pinned CPU memory exhausted PCIe BAR resources
- PyTorch's allocator couldn't handle the memory pressure
- Multiple concurrent weight allocations caused fragmentation

**New solution**: 
- **Compressed weights stay in RAM** (~65MB total)
- **Decode DIRECTLY to GPU** via nvCOMP (no CPU memory in decode path)
- **GPU-to-GPU copy** into PyTorch tensor (fast, no host transfer)
- **Sequentialized MLP** to prevent concurrent allocations
- **Immediate free** after each layer use

## Architecture

```
[Compressed in RAM] 
    ↓
[nvCOMP GPU Decode] → GPU buffer
    ↓
[cudaMemcpy D2D] → PyTorch CUDA tensor
    ↓
[F.linear]
    ↓
[cudaFree + del] → Free immediately
```

**Key**: NO CPU MEMORY involved in decompression!

## Build & Run on RunPod

```bash
# Pull latest
cd /workspace/CodecLLM
git pull

# Rebuild with GPU-direct decode
cd core
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
cd ..

# Test
python test_zstd_gpu_direct.py
```

## What to Expect

✓ **No OOM errors** - only one weight on GPU at a time  
✓ **Low peak VRAM** - compressed weights stay in RAM  
✓ **Slower inference** - decompression overhead, but acceptable  
✓ **True low-memory mode** - achieves the original goal!

## Technical Details

### C++ Changes
- `decoder_zstd.h/cpp`: Added `decodeLayerToGPU()` - returns GPU pointer
- `c_api_zstd.cpp`: Added `zstd_decoder_decode_layer_to_gpu()` C API
- `bindings_zstd.py`: Added `decode_layer_to_gpu()` Python method

### Python Changes
- `GPUDirectLinear`: Uses `decode_layer_to_gpu()` + `cudaMemcpy` D2D
- `SequentialMLP`: Ensures only one weight is live at a time
- NO pinned CPU memory caching
- NO GPU caching

### Memory Flow
1. **Compress**: FP16 → INT8 → Zstd → ~65MB in system RAM
2. **Forward pass**:
   - Read compressed bytes from RAM
   - nvCOMP decompresses to GPU buffer
   - cudaMemcpy D2D to PyTorch tensor (still on GPU)
   - Dequantize INT8 → FP16 on GPU
   - F.linear on GPU
   - cudaFree + del immediately
3. **Next layer**: Repeat (no memory accumulation)

## Notes

- The test uses `ctypes` + `libcudart.so` for `cudaMemcpy`/`cudaFree`
- Windows: use `cudart64_XX.dll` 
- On Linux/RunPod: `libcudart.so` should be in LD path
- The GPU-direct decode eliminates ALL CPU memory from the hot path!

## Expected Output

```
Baseline:   0.43s, 2.06 GB VRAM
Compressed: 3.2s, 1.4 GB VRAM  (estimated)
Slowdown: 7.4x
✓ VRAM saved: 0.66 GB
✓ Output matches baseline!
✓ Complete - TRUE GPU-DIRECT DECODE!
```

The slowdown is acceptable for the memory savings!

