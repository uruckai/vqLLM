# Week 4: GPU Decode Path

## Goal
Accelerate decoding 100x+ by moving rANS decode and reconstruction to GPU.

Target: Decode 7B model (28GB) in < 60 seconds on RTX 5090.

---

## Deliverables

### 1. CUDA Kernels (`cuda/`)
- **`rans_decode.cu`**: Per-tile parallel rANS decoder
  - Each tile = 1 threadblock (256 threads)
  - Warp-level decode (32 symbols in parallel)
  - Shared memory for frequency tables
  
- **`predictor_reconstruct.cu`**: GPU-side reconstruction
  - Parallel tile reconstruction
  - Mode: left/top/avg/planar
  - Direct write to GPU buffers
  
- **`transform.cu`**: Inverse DCT/ADST on GPU
  - Vectorized INT8 operations
  - Tensor core utilization (if beneficial)

### 2. Memory Management
- **Zero-copy decode**: rANS → GPU memory directly
- **Pinned host buffers**: For compressed data streaming
- **Multi-stream**: Overlap decode + reconstruction

### 3. Integration (`cpp/src/gpu_decoder.cpp`)
- Unified CPU/GPU decoder API
- Automatic device selection
- Fallback to CPU if CUDA unavailable

### 4. PyTorch Integration (`python/wcodec/torch_loader.py`)
- `load_wcodec_checkpoint()`: Direct to `state_dict`
- `WCodecModel.from_pretrained()`: HuggingFace-style API

---

## Architecture

```
Compressed .wcodec file
    ↓
[Host] Read compressed tiles → Pinned buffers
    ↓
[GPU Stream 1] rANS decode tile batch 1
[GPU Stream 2] rANS decode tile batch 2
    ↓
[GPU] Parallel reconstruction (predictor + inverse transform)
    ↓
[GPU Memory] INT8 tensors ready for inference
```

---

## Performance Targets

| Metric | Target | Baseline (CPU) |
|--------|--------|----------------|
| Decode 16MB layer | < 20ms | ~10-20 sec |
| Decode 7B model | < 60 sec | ~30-60 min |
| Memory overhead | < 2GB | N/A |
| Throughput | > 500 MB/s | ~0.05 MB/s |

---

## Implementation Plan

### Day 1-2: CUDA Kernels
- rANS parallel decoder
- Predictor reconstruction
- Unit tests (small tiles)

### Day 3-4: Integration
- GPU decoder wrapper
- Memory management
- Multi-stream pipeline

### Day 5: PyTorch Loader
- `load_wcodec_checkpoint()`
- Compatibility with `safetensors` API

### Day 6-7: Optimization & Benchmarking
- Kernel tuning
- Profile with Nsight Compute
- Hit performance targets

---

## Testing Strategy

1. **Correctness**: GPU decode matches CPU decode (bit-exact)
2. **Performance**: Profile decode latency vs layer size
3. **Integration**: Load Llama-2-7B checkpoint from `.wcodec`
4. **Stress**: Multi-GPU, large models (70B)

---

## Files Created

```
cuda/
  rans_decode.cu           # Parallel rANS decoder
  predictor_reconstruct.cu # GPU reconstruction
  transform.cu             # Inverse transforms
  kernels.cuh              # Shared kernel utilities

cpp/include/wcodec/
  gpu_decoder.h            # GPU decoder API

cpp/src/
  gpu_decoder.cpp          # GPU decoder implementation

python/wcodec/
  torch_loader.py          # PyTorch integration
  
tests/
  test_gpu_decoder.py      # GPU correctness tests
  benchmark_decode.py      # Performance benchmarks

scripts/
  build_cuda.sh            # CUDA build script
```

---

## Success Criteria

✅ GPU decode 100x faster than CPU  
✅ Bit-exact reconstruction vs CPU  
✅ Load 7B model in < 60 seconds  
✅ Direct PyTorch integration working  
✅ Multi-stream pipeline implemented  

---

**Note:** This is where the 5090 GPU finally gets used! All previous weeks were CPU-only algorithm development.

