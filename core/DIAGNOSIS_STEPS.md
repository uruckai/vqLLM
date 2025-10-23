# Diagnosis Steps for Garbage Output

**Issue**: LLM output is still corrupted despite per-channel quantization
**Output**: `'The capital of France is����������'`
**Compression ratio**: Dropped from 3.50x to 2.26x (suspicious!)

---

## Possible Causes

### 1. **Compression/Decompression Corruption** (Most Likely)
The UTF-8 decoding errors (`����`) suggest **binary data corruption**, not just quantization error.

**Test**:
```bash
cd /workspace/CodecLLM && git pull && cd core
python test_roundtrip.py
```

**Expected**: Should show "BIT-EXACT" match
**If it fails**: Compression pipeline is broken

### 2. **Per-Channel Scale Broadcasting Issue**
The scale might not be broadcasting correctly during dequantization.

**What I changed**: 
- Used `self.scale.view(-1, 1)` instead of `self.scale.unsqueeze(1)`
- Added explicit check for scale dimensionality

**Test**: Run the LLM test again

### 3. **Compression Ratio Drop**
3.50x → 2.26x is a **35% reduction in compression efficiency**

**Possible reasons**:
- Per-channel quantization uses more unique INT8 values (expected)
- But 35% drop is too much - suggests something else changed

---

## Debugging Steps

### Step 1: Verify Compression Pipeline
```bash
python test_roundtrip.py
```

**Look for**:
- "✓ BIT-EXACT MATCH!" in Step 4
- "✓ GPU decode matches CPU decode!" in Step 7
- Quantization error < 1%

**If this passes**: Compression is fine, issue is in inference code  
**If this fails**: Compression is corrupting data

### Step 2: Test Quantization Scheme
```bash
python test_quantization_debug.py
```

**Look for**:
- Per-channel should have ~10x lower error than per-tensor
- Per-channel should use more unique values

### Step 3: Re-run LLM Test
```bash
python test_zstd_inference.py
```

**Expected after fixes**:
- Output should match baseline
- Compression ratio might be slightly lower (2.5-3x) due to per-channel
- Time should be similar (~60-135s)

---

## What I've Changed

### 1. Improved Scale Broadcasting (test_zstd_inference.py)

**Before**:
```python
weight_fp = weight_int8_gpu.to(self.dtype) * self.scale.unsqueeze(1)
```

**After**:
```python
weight_fp_unscaled = weight_int8_gpu.to(self.dtype)

if self.scale.dim() == 1:
    # Per-channel: expand scale to (rows, 1)
    scale_expanded = self.scale.view(-1, 1)
else:
    # Per-tensor: scalar
    scale_expanded = self.scale

weight_fp = weight_fp_unscaled * scale_expanded
```

**Why**: More explicit handling of scale dimensions, ensures correct broadcasting

### 2. Added Diagnostic Tests

- `test_roundtrip.py`: End-to-end compression test
- `test_quantization_debug.py`: Quantization scheme comparison

---

## Expected Test Results

### test_roundtrip.py:
```
STEP 4: Verify bit-exact reconstruction
  ✓ BIT-EXACT MATCH!

STEP 7: GPU Decompress
  ✓ GPU decode successful
  ✓ GPU decode matches CPU decode!

SUMMARY:
✓ Compression/decompression: BIT-EXACT
✓ Quantization error: 0.0234% (expected <1%)
✓ Compression ratio: 2.50x
```

### test_quantization_debug.py:
```
PER-TENSOR QUANTIZATION:
  Overall error: 0.045231 (avg)

PER-CHANNEL QUANTIZATION:
  Overall error: 0.003892 (avg)

DIAGNOSIS:
✓ Per-channel is MUCH better than per-tensor (expected)
✓ Per-channel uses more unique values (better utilization)
```

### test_zstd_inference.py:
```
Baseline:
  Output: 'The capital of France is Paris.\n\n2. B. The capital'

Compressed (Zstd):
  Output: 'The capital of France is Paris.\n\n2. B. The capital'
                                     ✓ Same output!
```

---

## If Tests Still Fail

### Scenario A: Round-trip test fails (BIT-EXACT mismatch)
**Problem**: Compression is corrupting data
**Solution**: 
- Check nvCOMP version
- Verify encoder/decoder are using same format
- Check for buffer overflow in C++ code

### Scenario B: Round-trip passes, LLM test fails
**Problem**: Issue in inference code (dequantization, tensor operations)
**Solutions**:
1. Print intermediate tensors to check for NaN/Inf
2. Verify scale tensor is on correct device
3. Check if weight_fp has correct shape after dequantization
4. Compare compressed layer output vs uncompressed layer output

### Scenario C: Output has some correct tokens, then garbage
**Problem**: Quantization accumulated error or mixed compressed/uncompressed layers
**Solutions**:
1. Verify all 20 layers are actually being replaced
2. Check if any layer has extreme scale values
3. Try compressing only first 5 layers to isolate issue

---

## Next Steps

1. **Run `test_roundtrip.py`** - This is the most important test!
2. If round-trip passes, run `test_zstd_inference.py` again
3. Report results:
   - Does round-trip show BIT-EXACT?
   - Does GPU decode match CPU decode?
   - What's the LLM output now?
   - What's the compression ratio?

This will tell us exactly where the problem is!

---

## Quick Test Script

Copy-paste this on RunPod:
```bash
cd /workspace/CodecLLM && git pull && cd core

echo "========================================="
echo "TEST 1: Round-trip verification"
echo "========================================="
python test_roundtrip.py
echo ""

echo "========================================="
echo "TEST 2: Quantization comparison"
echo "========================================="
python test_quantization_debug.py
echo ""

echo "========================================="
echo "TEST 3: LLM inference"
echo "========================================="
python test_zstd_inference.py
```

This will run all three tests in sequence and show us exactly what's broken.

