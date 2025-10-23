# Quiet Test Guide - Remove Verbose Logs

The tests generate extremely verbose logs from C++ encoder/decoder. Here's how to get clean, actionable output.

---

## 🎯 Quick Solutions

### Option 1: Use Quiet Test (Recommended)
```bash
cd /workspace/CodecLLM/core
python3 test_all_layers_quiet.py 2>&1 | grep -vE "ENCODER|DECODER"
```

**What it does:**
- Removes ALL encoder/decoder C++ messages
- Shows only Python output (progress, results, errors)
- Much easier to read

**Output will show:**
- Model loading
- Progress every 20 layers
- Final results comparison
- Actionable recommendations

---

### Option 2: Keep First Occurrence Only
```bash
python3 test_all_layers_quiet.py 2>&1 | awk '
  /ENCODER.*Starting/ {if (!seen_start++) print; next}
  /ENCODER.*SUCCESS/ {if (++success % 50 == 1) print; next}
  /DECODER.*SUCCESS/ {if (++decode % 50 == 1) print; next}
  {print}
'
```

**What it does:**
- Shows first encoder message
- Shows every 50th success message for progress
- Keeps all other output

---

### Option 3: Save Full Log, View Filtered
```bash
# Save everything
python3 test_all_layers_quiet.py 2>&1 | tee full_log.txt

# View filtered
grep -vE "ENCODER|DECODER" full_log.txt
```

**What it does:**
- Saves complete log for debugging if needed
- View clean version on screen
- Best of both worlds

---

## 📊 Expected Clean Output

With filtering, you should see:

```
================================================================================
ALL LAYERS COMPRESSED TEST (QUIET MODE)
================================================================================

Device: cuda

[1/3] Loading model...
  Found 155 Linear layers
  Baseline output: 'The capital of France is Paris...'
  Time: 0.60s, VRAM: 2.06 GB

[2/3] Compressing ALL 155 layers...
  Progress: 1/155 layers...
  Progress: 20/155 layers...
  Progress: 40/155 layers...
  ...
  Progress: 155/155 layers...
  ✓ Compression complete!
    Original: 1973.0 MB → Compressed: 880.8 MB
    Ratio: 2.24x

  Creating compressed model...
    Layer 0 (model.embed_tokens): scale range [0.000123, 0.045678]
    Layer 1 (model.layers.0.q_proj): scale range [0.000456, 0.067890]
    Layer 2 (model.layers.0.k_proj): scale range [0.000234, 0.034567]
  ✓ Replaced 155 layers

[3/3] Running compressed inference...
  Generating 10 tokens...

================================================================================
RESULTS
================================================================================

Baseline:
  Output: 'The capital of France is Paris.\n\n2. B. The capital'
  Time:   0.60s
  VRAM:   2.06 GB

All 155 layers compressed:
  Output: 'The capital of France is??????????'
  Time:   68.21s (113.7x slower)
  VRAM:   2.36 GB
  Ratio:  2.24x

✗ MAJOR DIFF - Outputs diverge
  Expected: 'The capital of France is Paris.'
  Got:      'The capital of France is??????????'
  Diverges at position 25: expected 'P' got '?'

================================================================================
ACTIONABLE INFORMATION
================================================================================

✗ Major quality issues detected:
  1. Run: python3 test_quantization_roundtrip.py
  2. Check scale precision (float16 vs float32)
  3. Verify dequantization math
  4. Try per-tensor quantization instead of per-channel
```

---

## 🔧 For Future: Suppress C++ Debug (Permanent Fix)

To permanently fix this, modify the C++ source:

### In `encoder_zstd_v3.cpp`:
```cpp
// Add at top:
bool QUIET_MODE = std::getenv("CODEC_QUIET") != nullptr;

// Replace debug prints:
if (!QUIET_MODE) {
    printf("[ENCODER] Starting nvCOMP...\n");
}
```

### In `decoder_zstd_v3.cpp`:
```cpp
// Same pattern:
bool QUIET_MODE = std::getenv("CODEC_QUIET") != nullptr;

if (!QUIET_MODE) {
    printf("[DECODER] GPU direct decode SUCCESS\n");
}
```

### Then rebuild and use:
```bash
cd /workspace/CodecLLM/core
CODEC_QUIET=1 python3 test_all_layers_quiet.py
```

---

## 📝 Comparison

| Method | Pros | Cons |
|--------|------|------|
| `grep -vE` | Instant, no code changes | Removes ALL C++ messages |
| `awk` filtering | Shows some progress | More complex command |
| Save + filter | Can review full log later | Extra file created |
| C++ env var | Clean permanent solution | Requires rebuild |

**Recommendation**: Use `grep -vE` for now, implement C++ env var for production.

---

## 🎯 Commands Summary

```bash
# Quietest (recommended):
python3 test_all_layers_quiet.py 2>&1 | grep -vE "ENCODER|DECODER"

# With some progress:
python3 test_all_layers_quiet.py 2>&1 | grep -v "^\[ENCODER\] Temp:"

# Save full log:
python3 test_all_layers_quiet.py 2>&1 | tee full.log | grep -vE "ENCODER|DECODER"

# Round-trip test (less verbose):
python3 test_quantization_roundtrip.py  # Already minimal output
```

---

## ✅ Verification

After running quiet test, you should see:
- ✓ Clear progress indicators
- ✓ Final results comparison
- ✓ Actionable next steps
- ✗ NO repetitive encoder/decoder spam

**Log size reduction**: ~50MB → ~5KB (99% reduction!)

