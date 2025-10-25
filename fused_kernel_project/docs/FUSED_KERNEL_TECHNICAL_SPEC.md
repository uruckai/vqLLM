# Fused Kernel Technical Specification

## Core Architecture Decisions

### 1. Tile-Based Parallelization (Maintain Current Structure)

**Decision:** Keep existing 256×256 tile architecture

```python
# Current (working):
TILE_SIZE = 256
tiles_per_layer = 176  # 2048×5632 / 256×256
gpu_blocks = 176       # One CUDA block per tile

# Fused kernel (same structure):
__global__ void fusedKernel(const uint8_t* compressed, const half* input, half* output) {
    int tile_idx = blockIdx.x;  // 0-175 for this layer
    // Decompress tile_idx, compute GEMM, store results
}
```

**Why this works:**
- ✅ **Proven GPU utilization:** 176 blocks = 100% SM saturation
- ✅ **Memory efficient:** Each tile = 65KB (fits in L2 cache)
- ✅ **Parallel independent:** Tiles don't depend on each other
- ✅ **Shared frequency table:** 512 bytes loaded per block

### 2. Compression Format (Use Existing rANS)

**Decision:** Use existing rANS compressed format

```cpp
// Header (global, once per layer)
Header header = {
    magic: 0x52414E53,
    tile_size: 256,
    num_tiles_row: 8,
    num_tiles_col: 22,
    // ...
};

// Per-tile metadata (array of 176 entries)
TileMetadata metadata[176] = {
    predictor_mode: 3,
    data_offset: 1024,    // Byte offset in compressed data
    data_size: 15432,     // Compressed bytes for this tile
    // ...
};

// Frequency table (global, 512 bytes)
uint16_t freq_table[256] = {10, 15, 8, ...};
```

**Integration:**
```cuda
__global__ void fusedKernel(...) {
    // Load shared frequency table (512 bytes)
    __shared__ RANSSymbol symbols[256];
    if (threadIdx.x == 0) {
        loadFreqTable(symbols);  // 512 bytes
    }
    __syncthreads();

    // Each block processes one tile
    int tile_idx = blockIdx.x;
    decompressTile(tile_idx, symbols);  // Uses shared table
}
```

### 3. Memory Layout Optimization

**Current layout (optimized):**
```
Compressed layer: 11MB (1.4x compression)
├── Header: 64 bytes
├── Frequency table: 512 bytes
├── Metadata array: 176 × 32 = 5,632 bytes
├── Tile 0 compressed: 15,432 bytes
├── Tile 1 compressed: 15,221 bytes
├── ...
└── Tile 175 compressed: 14,876 bytes
```

**Fused kernel access pattern:**
```cuda
// Coalesced reads for metadata array
TileMetadata meta = metadata[tile_idx];  // 32 bytes per tile

// Streamed reads for compressed data
const uint8_t* tile_data = compressed + meta.data_offset;
```

### 4. Quantization Strategy

**Option A: INT8 with quantization-aware training (Recommended)**
```cuda
// Training: Model learns INT8 representation
weights_int8 = quantize_aware_train(model)

// Inference: Decompress + dequantize + compute
__device__ half dequantize(int8_t x, float scale) {
    return (half)x * scale;
}

// Fused kernel: INT8 → FP16 → GEMM
half weight = dequantize(decompressed_int8, scale);
output = weight * input;  // FP16 computation
```

**Option B: FP8 for Blackwell GPUs (Future)**
```cuda
// FP8 storage (2x compression vs FP16)
weights_fp8 = convert_to_fp8(weights_fp16);

// Fused kernel: FP8 → FP8/FP16 → Tensor Core GEMM
half weight = convert_fp8_to_half(decompressed_fp8);
output = tensor_core_gemm(weight, input);  // Optimized path
```

---

## 5. Integration Points

### 5.1 Replace Current Decompression

**Current (two-step):**
```python
# Step 1: GPU decompression
gpu_decode_kernel<<<176, 256>>>(compressed, output_int8);

# Step 2: CPU/GPU dequantization + GEMM
dequantize_and_gemm(output_int8, input, output_fp16);
```

**Fused (single-step):**
```python
# Single fused kernel
fused_kernel<<<176, 256>>>(compressed, input, output_fp16);
```

**Integration:**
- ✅ **Same CUDA launch:** `<<<176, 256>>>` (optimal block count)
- ✅ **Same memory layout:** Compressed data format unchanged
- ✅ **Same parallelization:** One block per tile
- ❌ **Different output:** Direct FP16 instead of INT8

### 5.2 PyTorch Integration

**Hook-based loading (maintain current framework):**
```python
def fused_weight_loader(compressed_data, layer):
    """Replace linear layer with fused decompression"""

    class FusedLinear(torch.nn.Module):
        def __init__(self, compressed_data):
            super().__init__()
            self.compressed = compressed_data
            self.scale = compressed_data['scale']

        def forward(self, x):
            # Launch fused kernel
            output = fused_cuda_kernel(self.compressed, x, self.scale)
            return output

    # Replace the layer
    return FusedLinear(compressed_data)
```

### 5.3 Memory Management

**Staging buffer strategy:**
```python
# Pre-allocate staging buffers
staging_buffers = {
    'int8': torch.empty((max_tiles, 256, 256), dtype=torch.int8, device='cpu'),
    'fp16': torch.empty((max_tiles, 256, 256), dtype=torch.float16, device='cuda')
}

# Fused kernel decompresses to staging, computes, stores result
fused_kernel(compressed, input, staging_fp16_output)
```

---

## 6. Performance Targets

### 6.1 Single Layer Performance

**Target metrics:**
```python
# Compression
compression_ratio = 1.4  # INT8 (minimum)
compression_time = 50ms  # CPU encoding (optimized)

# Decompression
decompression_time = 1ms   # GPU (target)
speedup_vs_baseline = 1.4  # Memory bandwidth savings
gpu_utilization = 95%     # During decompression
```

### 6.2 Multi-Layer Performance

**Target metrics:**
```python
# Concurrent decompression
num_concurrent_layers = 5   # CUDA streams
total_decompression_time = 1.5ms  # Overlapped
throughput_improvement = 1.7  # Additional from overlapping

# Memory efficiency
vram_savings = 30%  # Compressed storage
memory_bandwidth_reduction = 50%  # INT8 vs FP16 reads
```

### 6.3 Accuracy Targets

**With quantization-aware training:**
```python
# Model accuracy preservation
baseline_accuracy = 0.85  # Original model
fused_accuracy = 0.83     # Target (2% degradation acceptable)

# Numerical stability
bit_exact_decompression = True  # rANS verified
dequantization_error = 1e-4     # Scale factor precision
```

---

## 7. Implementation Priority

### Phase 1: Core Fused Kernel (Week 1-2)
```
1. Basic fused kernel (decompress + dequantize + GEMM)
2. Integration with existing tile architecture
3. Single layer testing and benchmarking
4. Accuracy validation with known data
```

### Phase 2: Optimization (Week 3)
```
1. CUDA streams for multi-layer overlapping
2. Memory staging buffer optimization
3. Performance profiling and tuning
4. Integration with PyTorch hook system
```

### Phase 3: Production (Week 4)
```
1. Full model integration
2. Dynamic loading compatibility
3. Error handling and fallback
4. End-to-end testing
```

---

## 8. Risk Mitigation

### High Risk (Address First)
```
1. Quantization-aware training compatibility
   - Test with simple model first
   - Validate accuracy preservation

2. Dynamic loading integration
   - KV cache compatibility
   - PyTorch hook timing
   - Memory management
```

### Medium Risk (Monitor)
```
1. GPU memory pressure
   - Staging buffer sizing
   - VRAM allocation patterns

2. Performance regression
   - Larger tiles reduce parallelization
   - Kernel launch overhead
```

### Low Risk (Proven Working)
```
1. rANS decompression (bit-exact verified)
2. GPU parallelization (176 blocks optimal)
3. Tile-based architecture (scalable)
4. Compressed format (header + metadata + tiles)
```

---

## 9. Expected Outcomes

### Performance
```
- Decompression speedup: 1.4×-2.0× vs current optimized
- Memory bandwidth: 50% reduction
- GPU utilization: 90%+ during decompression
- VRAM savings: 30% (compressed storage)
```

### Accuracy
```
- Quantization-aware: 95-100% baseline accuracy
- Numerical stability: Bit-exact decompression
- Autoregressive consistency: No error amplification
```

### Integration
```
- Dynamic loading: Full compatibility
- KV cache: Preserved functionality
- Fallback: Graceful degradation to baseline
```

This technical specification provides the foundation for implementing fused kernels while maintaining the proven architecture and performance optimizations discovered during the experimental phase.

