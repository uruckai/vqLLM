# Week 6: GPU Integration Completion & Optimization

## Goal
Complete GPU decoder integration and achieve 100x+ speedup over CPU decode.

Target: Full 7B model decode in < 60 seconds on RTX 5090.

---

## Critical Path

### Phase 1: Wire GPU Decoder (Day 1-2)
1. **Parse container format in GPU decoder**
   - Extract per-tile frequency tables
   - Extract predictor modes
   - Extract tile offsets/sizes

2. **Transfer metadata to GPU**
   - Allocate GPU buffers for frequency tables
   - Copy tile metadata
   - Prepare compressed bitstream

3. **Launch CUDA kernels**
   - Call `launch_rans_decode()`
   - Call `launch_predictor_reconstruct()`
   - Synchronize and validate

### Phase 2: Validation (Day 3)
1. **Bit-exact testing**
   - GPU vs CPU decode on same data
   - Verify every byte matches

2. **Edge case testing**
   - Different tile sizes
   - Different predictor modes
   - Large layers

### Phase 3: Optimization (Day 4-5)
1. **Profile with Nsight Compute**
   - Identify bottlenecks
   - Measure kernel occupancy

2. **Optimize kernels**
   - Warp-level rANS decode
   - Coalesced memory access
   - Shared memory tuning

3. **Multi-stream overlap**
   - Overlap H2D transfer + decode
   - Pipeline tiles through streams

### Phase 4: End-to-End (Day 6)
1. **Full checkpoint decode**
   - Test on real checkpoint
   - Measure end-to-end latency

2. **PyTorch integration**
   - Load directly to GPU tensors
   - Zero-copy where possible

---

## Implementation Tasks

### Task 1: Update GPU Decoder Implementation
File: `cpp/src/gpu_decoder.cpp`

Need to replace the TODO section with actual implementation:
```cpp
// Current (TODO placeholder):
// TODO: Parse container format...
// Falls back to CPU

// New implementation:
// 1. Parse LayerInfo from container
// 2. Extract tile metadata
// 3. Allocate GPU memory
// 4. Transfer metadata
// 5. Launch kernels
// 6. Copy results back
```

### Task 2: Create GPU Metadata Extractor
File: `cpp/src/gpu_metadata_extractor.cpp`

Helper to convert LayerInfo → GPU-ready structures:
```cpp
struct GPUTileMetadata {
    uint32_t* d_freq_tables;    // Device pointer
    uint8_t* d_predictor_modes;
    size_t* d_tile_offsets;
    size_t* d_tile_sizes;
    // ...
};

GPUTileMetadata extractGPUMetadata(const LayerInfo& info);
```

### Task 3: Update Python Bindings
Enable GPU decode in Python:
```python
# decoder_api.py
decoder = GPUDecoder(use_gpu=True)  # Actually use GPU now
decoded = decoder.decode_layer(...)
```

### Task 4: Benchmarking Suite
File: `tests/benchmark_gpu.py`

Comprehensive GPU performance tests:
- Throughput (MB/s)
- Latency per layer size
- Comparison vs CPU
- Memory usage

---

## Success Criteria

| Metric | Target | Current (CPU) |
|--------|--------|---------------|
| Decode 256×256 | < 1ms | ~50ms |
| Decode 4096×4096 | < 50ms | ~20 sec |
| Full 7B model | < 60 sec | ~30-60 min |
| Throughput | > 500 MB/s | ~0.05 MB/s |
| Bit-exact | 100% | 100% |

---

## Files to Create/Update

```
cpp/src/
  gpu_decoder.cpp (UPDATE)       # Wire up GPU path
  gpu_metadata_extractor.cpp     # Helper for metadata
  
cpp/include/wcodec/
  gpu_metadata_extractor.h       # Header
  
tests/
  benchmark_gpu.py               # GPU benchmarks
  test_gpu_validation.py         # GPU vs CPU validation
  
scripts/
  profile_gpu.sh                 # Nsight Compute profiling
```

