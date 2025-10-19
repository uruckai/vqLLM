# Running Low-Memory Inference Test

## Quick Start (RunPod)

```bash
# 1. Pull latest changes
cd /workspace/CodecLLM
git pull

# 2. Rebuild (if needed)
cd core
./build.sh

# 3. Test basic compression flow
python3 test_lowmem_simple.py

# 4. If basic test passes, run full inference test
python3 test_inference_lowmem.py
```

## What to Expect

### Test 1: `test_lowmem_simple.py` (Quick validation)
**Purpose**: Test the basic building blocks without loading a full LLM

**Tests**:
1. **CompressedTensor**: Compress/decompress a 2048×2048 tensor
   - Should achieve ~1.3x compression
   - Reconstruction error < 1% due to INT8 quantization
   
2. **CompressedLinear**: Wrap an `nn.Linear` layer with on-demand decompression
   - Forward pass should work correctly
   - Output should match original (within quantization error)

**Expected output**:
```
TEST 1: CompressedTensor Compression/Decompression
  Compression ratio: 1.33x
  ✅ Reconstruction quality: GOOD

TEST 2: CompressedLinear Layer
  Compression ratio: 1.33x
  ✅ Forward pass quality: GOOD
  
✅ ALL TESTS PASSED!
```

**Time**: ~5-10 seconds

---

### Test 2: `test_inference_lowmem.py` (Full inference)
**Purpose**: Test actual inference with TinyLlama model

**What it does**:
1. Loads TinyLlama-1.1B (uncompressed) → baseline inference
2. Compresses all Linear layers → measure compression
3. Runs inference with on-demand decompression → measure VRAM savings

**Expected output**:
```
[1/5] Loading codec...
✓ Codec library loaded
✓ GPU decoder available

[2/5] Loading PyTorch and Transformers...
✓ Libraries loaded

[3/5] Loading model...
✓ Model loaded: TinyLlama/TinyLlama-1.1B-Chat-v1.0

[4/5] Running baseline inference (uncompressed)...
  Generated: 'The capital of France is Paris...'
  Peak VRAM: 2.5 GB

[5/5] Compressing model and running low-memory inference...
  ✓ Compressed 220 Linear layers
  Original size:    2.20 GB
  Compressed size:  1.65 GB
  Compression ratio: 1.33x
  Space saved:      24.9%
  
  Generated: 'The capital of France is Paris...'
  Peak VRAM: 1.2 GB

VRAM USAGE COMPARISON
Baseline (uncompressed):  2.5 GB
Compressed (on-demand):   1.2 GB
VRAM reduction:           2.1x
VRAM saved:               52%

✅ OUTPUTS MATCH!
```

**Time**: ~2-5 minutes (includes model download first time)

---

## Troubleshooting

### If `test_lowmem_simple.py` fails:

**Error**: `Codec library not found`
```bash
cd /workspace/CodecLLM/core
./build.sh
```

**Error**: `PyTorch not installed`
```bash
pip install torch transformers
```

**Error**: Compression fails or segfault
- This means there's a bug - please share the full error output!

### If `test_inference_lowmem.py` fails:

**Error**: `HuggingFace token required`
```bash
export HF_TOKEN=your_token_here
python3 test_inference_lowmem.py
```

**Error**: Out of memory during model load
- Normal - the test tries to load the full model first for baseline
- If you only care about low-memory mode, we can skip baseline

**Error**: Inference produces different output
- Some difference is expected due to INT8 quantization
- As long as it's generating coherent text, it's working!

---

## Key Metrics to Watch

### Compression Ratio
- **Target**: 1.3-1.4x on INT8 weights
- **Bad**: < 1.2x (codec not working well)
- **Good**: 1.3-1.4x (as expected)
- **Great**: > 1.4x (better than average!)

### VRAM Reduction
- **Target**: 2-3x reduction vs baseline
- **Why?**: We compress weights (1.33x) + only load current layer (reduces peak)
- **Actual savings depend on**:
  - Model architecture (how many layers)
  - Batch size (activations still uncompressed)
  - Sequence length (KV cache still uncompressed)

### Inference Speed
- **Expected**: 2-5x slower than baseline
- **Why?**: Decompression overhead (CPU decode + GPU upload per layer)
- **Trade-off**: Slower inference for much lower VRAM

---

## Success Criteria

✅ **Minimum**:
- `test_lowmem_simple.py` passes both tests
- Compression ratio > 1.2x
- No crashes or segfaults

✅ **Good**:
- `test_inference_lowmem.py` completes without errors
- Compression ratio ~1.33x
- VRAM reduction > 1.5x

✅ **Excellent**:
- Both tests pass
- Compression ratio ~1.33x  
- VRAM reduction > 2x
- Generated text is coherent

---

## What This Demonstrates

**The Point**: You can run larger models on smaller GPUs!

**Example Use Case**:
- **Normal**: Llama-3.1-8B requires ~16GB VRAM (FP16)
- **With INT8**: Requires ~8GB VRAM
- **With INT8 + Our Codec (low-mem mode)**: Requires ~3-4GB VRAM!

**This means**:
- Run 8B models on consumer GPUs (RTX 3060, 4060)
- Serve multiple models on single GPU (multi-tenancy)
- Reduce cloud GPU costs (smaller instance = cheaper)

**Trade-off**:
- Slower inference (2-5x)
- But enables use cases that were impossible before!

---

## Next Steps After Success

1. **Try larger models**: Llama-3.1-8B, Llama-2-13B
2. **Measure actual speedup**: Compare to baseline inference time
3. **Optimize decompression**: 
   - Prefetch next layer while computing current
   - Use GPU decoder (currently using CPU)
   - Batch decompress multiple layers
4. **Production integration**:
   - HuggingFace Transformers plugin
   - vLLM integration
   - TensorRT-LLM integration

