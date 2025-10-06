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

**Phase:** Week 3 Complete ✅  
**Next:** Build & test, then Week 4 GPU decode

### Roadmap

- [x] Week 1: Specification, baseline tools ✅
- [x] Week 2: C++ encoder/decoder (predictive + rANS) ✅
- [x] Week 3: Transform coding (DCT/ADST) + bitplanes + container format ✅
- [ ] Week 3.5: Integration & testing (target 30-50% compression)
- [ ] Week 4: GPU decode path (CUDA)
- [ ] Week 5: PyTorch integration
- [ ] Week 6: Benchmarking & optimization

---

## Quick Reference

### Run Baseline Measurements

```bash
# Create synthetic checkpoint and measure
python scripts/baseline_harness.py --model llama3-8b --quant int8 --size medium --output baselines/
```

### Verify Bit-Exact Reconstruction

```bash
# (Once encoder/decoder implemented)
python scripts/verify_checkpoint.py --original model.safetensors --decoded model_decoded.safetensors
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

