# Weight Codec (WCodec) — Research Project

**Single-person research: codec-inspired compression for LLM weights (AV1/VP9 techniques)**

## Goal

Achieve **30–60% smaller checkpoint files** with bit-exact reconstruction and zero runtime overhead.

**Techniques:**
- Predictive coding (intra-prediction)
- Transform coding (integer DCT/ADST)
- Bitplane representation
- Context-adaptive rANS entropy coding

---

## Current Status

**Phase:** Week 5 Complete (85%) ✅  
**Next:** GPU integration finalization & optimization

### Roadmap

- [x] Week 1: Specification, baseline tools ✅
- [x] Week 2: C++ encoder/decoder (predictive + rANS) ✅
- [x] Week 3: Transform coding (DCT/ADST) + bitplanes ✅
- [x] Week 4: GPU decode infrastructure (CUDA kernels) ✅
- [x] Week 5: Container format + high-level APIs ✅
- [ ] Week 5.5: GPU integration finalization (in progress)
- [ ] Week 6: Optimization & benchmarking
- [ ] Week 7: Production-ready polish & documentation

---

## Quick Reference

### Build the Project

```bash
# CPU-only build
mkdir -p build && cd build
cmake .. && make -j8

# With CUDA support (auto-detects GPU)
bash scripts/build_cuda.sh
```

### Run Tests

```bash
# End-to-end integration tests
python3 tests/test_end_to_end.py

# Compression roundtrip tests (CPU)
python3 tests/test_compression_roundtrip.py

# GPU decoder tests
python3 tests/test_gpu_decoder.py

# Performance benchmark
python3 tests/benchmark_decode.py
```

### Encode/Decode Checkpoints

```bash
# Encode safetensors to .wcodec (Python API)
python3 -c "
from wcodec.encoder_api import encode_checkpoint
stats = encode_checkpoint('model.safetensors', 'model.wcodec')
print(f'Compression: {stats[\"compression_ratio\"]:.2f}x')
"

# Or use CLI tool
python3 scripts/encode_checkpoint.py model.safetensors model.wcodec --tile-size 16
```

### Run Baseline Measurements

```bash
# Create synthetic checkpoint and measure
python scripts/baseline_harness.py --model llama3-8b --quant int8 --size medium --output baselines/
```

### Install Package

```bash
cd python && pip install -e .
```

---

## Key Documents

- **[docs/codec_spec_v0.md](docs/codec_spec_v0.md)** — Complete technical specification
- **[docs/integration_guide.md](docs/integration_guide.md)** — PyTorch/TensorRT integration patterns  
- **[docs/week1_summary.md](docs/week1_summary.md)** — Week 1 progress and next steps
- **[CodecLLMDiscussion.txt](CodecLLMDiscussion.txt)** — Full research context

---

## Architecture

```
Quantized Weights → Tile (16×16) → Predict → Transform → Bitplane → rANS → .wcodec
```

**Target:** RTX 5090 with FP8/INT4 support

---

## Performance Targets

| Metric | Target |
|--------|--------|
| File size | ≥30–60% smaller than INT8/INT4 |
| Decode latency | ≤ model warm-up time |
| Accuracy delta | ≤0.1 pp |
| Bit-exactness | 100% |

