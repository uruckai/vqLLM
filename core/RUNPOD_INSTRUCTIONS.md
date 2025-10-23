# RunPod Testing Instructions

## 📋 Current Situation

You're picking up from where quantization artifacts were last seen. The playbook shows:
- **Last test**: 1 compressed layer, output was "The capital of France is, 1..." (artifacts)
- **Goal**: Scale to 5-20 layers and validate quantization quality
- **Status**: Per-channel quantization implemented, but needs validation

---

## 🚀 Quick Start (What to Run Now)

### Step 1: Pull Latest Code & Check Environment
```bash
cd /workspace/CodecLLM
git pull
cd core
python3 check_runpod_status.py
```

**Expected output**: All 6 checks should pass ✓

If any checks fail:
- **Library not found**: Run `bash build.sh` first
- **GPU decoder not available**: Check nvCOMP installation (`bash INSTALL_NVCOMP3.sh`)
- **Round-trip test failed**: GPU decode issue, check CUDA version

---

### Step 2: Run Progressive Compression Test
```bash
cd /workspace/CodecLLM/core
python3 test_progressive_compression.py
```

**What it does**:
1. Runs baseline inference (uncompressed)
2. Tests with 1 compressed layer
3. Tests with 5 compressed layers
4. Tests with 10 compressed layers
5. Tests with 20 compressed layers
6. Compares output quality and performance

**Expected runtime**: ~5-10 minutes total

**What to look for**:
- ✓ **PERFECT**: Output text matches baseline exactly
- ⚠ **MINOR DIFF**: First 20 chars match (acceptable)
- ✗ **MAJOR DIFF**: Garbage output (quantization broken)

---

### Step 3: Share Results

Copy the **FINAL SUMMARY** section and share:
```
Layers   Time        VRAM       Ratio   Quality         Output
------------------------------------------------------------------------
1        X.XXs       X.XX GB    X.XXx   [quality]       'text...'
5        X.XXs       X.XX GB    X.XXx   [quality]       'text...'
10       X.XXs       X.XX GB    X.XXx   [quality]       'text...'
20       X.XXs       X.XX GB    X.XXx   [quality]       'text...'
```

---

## 🔍 Interpretation Guide

### If All Tests Show ✓ PERFECT or ⚠ MINOR DIFF
**Great! Quantization is working!** Next steps:
1. Scale to 50+ layers: Edit test_progressive_compression.py, change `[1, 5, 10, 20]` to `[20, 50, 100, 155]`
2. Focus on performance optimization (batch decompress, fused kernels)
3. Update playbook with successful results

### If Tests Show ✗ MAJOR DIFF
**Quantization issue detected.** Debug steps:
1. Check scale values: Should be in range [1e-6, 1.0], not zeros
2. Check INT8 data: Should be in range [-127, 127], not all zeros
3. Run detailed debug: `python3 test_quantization_debug.py`

Common causes:
- **Scales are zeros**: NumPy aliasing bug (should be fixed already)
- **INT8 values wrong**: Compression/decompression issue
- **Dequantization wrong**: Broadcasting issue in PyTorch

---

## 📊 Performance Expectations

### Current Status (from playbook):
- **1 layer**: 11.5x slower than baseline
- **20 layers**: Expected ~50-100x slower (per-token decompression overhead)

### Target Goals:
- **Accuracy**: Output should be readable and similar to baseline
- **VRAM**: Should stay stable (no growth), ideally lower than baseline
- **Speed**: Acceptable if < 100x slower (optimization comes later)

---

## 🐛 Troubleshooting

### "GPU decoder not available"
```bash
cd /workspace/CodecLLM/core
bash INSTALL_NVCOMP3.sh
bash build.sh
```

### "Library not found"
```bash
cd /workspace/CodecLLM/core
bash build.sh
```

### "Round-trip test failed"
Check CUDA and nvCOMP:
```bash
nvidia-smi
find /usr /opt -name "libnvcomp*.so" 2>/dev/null
```

### "Garbage output (���)"
This was the original bug! Check if it's back:
1. Verify per-channel quantization is enabled (test_zstd_inference.py line 122)
2. Check scales are copied properly (line 156: `scales.squeeze().copy()`)
3. Run: `python3 test_quantization_debug.py` for detailed analysis

---

## 📝 After Testing

Update the playbook with your results:
1. Edit `PROJECT_PLAYBOOK.md` section "📊 Current Status & Test Results"
2. Add your progressive test results
3. Update "Next Steps & Roadmap" based on findings

---

## 🎯 Decision Tree

```
Run check_runpod_status.py
  ├─ All pass → Run test_progressive_compression.py
  │              ├─ ✓ PERFECT → Scale to more layers (50+)
  │              ├─ ⚠ MINOR   → Acceptable, scale up
  │              └─ ✗ MAJOR   → Debug quantization (test_quantization_debug.py)
  │
  └─ Checks fail → Fix environment
                   ├─ Library missing → bash build.sh
                   ├─ GPU decoder missing → bash INSTALL_NVCOMP3.sh
                   └─ Other issues → Check CUDA (nvidia-smi)
```

---

## 📞 Quick Commands Reference

```bash
# Full setup (if starting fresh)
cd /workspace/CodecLLM/core
bash FULL_SETUP_AND_TEST.sh

# Just run tests
cd /workspace/CodecLLM/core
python3 check_runpod_status.py        # Diagnostic
python3 test_progressive_compression.py  # Main test

# Debug quantization
python3 test_quantization_debug.py

# Manual test with specific layer count
python3 test_zstd_inference.py        # 1 layer (default)
python3 test_zstd_inference_10layers.py  # 5 layers
```

---

**Ready to test!** Start with `check_runpod_status.py` to make sure everything is working, then run the progressive test. 🚀

