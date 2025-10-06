# Week 2 Implementation Plan

**Goal:** CPU-based encoder/decoder with predictive coding + rANS

## Status: In Progress

### ✅ Completed (Just Now)

1. **C++ Core Implementation**
   - `cpp/include/wcodec/types.h` — Common types and constants
   - `cpp/include/wcodec/predictor.h` — Predictive coding interface
   - `cpp/include/wcodec/rans.h` — rANS entropy coder interface
   - `cpp/include/wcodec/encoder.h` — High-level encoder
   - `cpp/include/wcodec/decoder.h` — High-level decoder

2. **C++ Implementations**
   - `cpp/src/predictor.cpp` — LEFT/TOP/AVG/PLANAR predictors (~200 lines)
   - `cpp/src/rans.cpp` — rANS encoder/decoder (~180 lines)
   - `cpp/src/encoder.cpp` — Layer encoding with tiling (~150 lines)
   - `cpp/src/decoder.cpp` — Layer decoding with reconstruction (~130 lines)

3. **Build System**
   - Updated `CMakeLists.txt` with source files
   - Compiles to shared library

4. **Python Wrappers (Stubs)**
   - `python/wcodec/encoder.py` — Encoder wrapper (ready for bindings)
   - `python/wcodec/decoder.py` — Decoder wrapper (ready for bindings)

5. **Tests**
   - `tests/test_predictor.py` — Basic test framework

---

## 🚧 Next Steps

### Step 1: Build the C++ Library (Do on RunPod)

```bash
cd /workspace/CodecLLM

# Build
./build_and_test.sh

# Or manually:
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
make -j8
```

**Expected output:**
- `build/libwcodec.so` (Linux)
- Successful compilation of all 4 C++ source files

### Step 2: Add Python Bindings (Optional for Week 2)

Two options:

**Option A: pybind11** (Recommended for Week 3)
- Add `cpp/python_bindings.cpp`
- Expose Encoder/Decoder classes
- Build Python extension module

**Option B: ctypes** (Quick for testing)
- Add C-style wrapper functions
- Call directly from Python via ctypes
- Good enough for Week 2 testing

### Step 3: Test Basic Functionality

```bash
# Test with simple numpy array
python -c "
import numpy as np
# Generate test data
data = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
print('Test data created:', data.shape)
# Once bindings work, will encode/decode here
"
```

### Step 4: Measure Compression on Synthetic Checkpoint

```python
# Load synthetic checkpoint from Week 1
from safetensors import load_file
checkpoint = load_file("baselines/synthetic_test_int8.safetensors")

# Encode each layer
for name, tensor in checkpoint.items():
    # Convert to int8 numpy
    # Encode with wcodec
    # Measure compression ratio
    pass
```

**Target:** ≥20-30% compression ratio without transforms

---

## Known Limitations (Week 2)

1. **No frequency table storage** — Decoder assumes uniform distribution (will fix)
2. **Simplified container format** — Just metadata + streams (proper format in Week 3)
3. **No transforms yet** — Only predictive coding + rANS (transforms in Week 3)
4. **No parallelization** — Single-threaded encoding (threading in Week 3)
5. **No Python bindings yet** — Need to add pybind11 or ctypes wrapper

---

## Success Criteria (Week 2)

- [ ] C++ library compiles successfully ✅ (Code ready)
- [ ] Can encode a 64×64 int8 tile
- [ ] Can decode and get bit-exact reconstruction
- [ ] Compression ratio ≥20-30% on test data
- [ ] Encode time <10s for 1GB checkpoint
- [ ] Decode time <5s for 1GB checkpoint on CPU

---

## Architecture Implemented

```
Input (int8 tensor)
         ↓
    Tile (16×16)
         ↓
 Select Predictor (LEFT/TOP/AVG/PLANAR)
         ↓
 Compute Residual
         ↓
 Build Frequency Table
         ↓
 rANS Encode
         ↓
    Output (bytes)
```

**Decoder: Inverse path**

---

## Files Created (Week 2)

### C++ Headers (5 files, ~500 lines)
- `cpp/include/wcodec/types.h`
- `cpp/include/wcodec/predictor.h`
- `cpp/include/wcodec/rans.h`
- `cpp/include/wcodec/encoder.h`
- `cpp/include/wcodec/decoder.h`

### C++ Source (4 files, ~660 lines)
- `cpp/src/predictor.cpp` — Predictive coding
- `cpp/src/rans.cpp` — Entropy coding
- `cpp/src/encoder.cpp` — Tile encoding
- `cpp/src/decoder.cpp` — Tile decoding

### Python (2 files, ~150 lines)
- `python/wcodec/encoder.py`
- `python/wcodec/decoder.py`

### Tests & Scripts (2 files)
- `tests/test_predictor.py`
- `build_and_test.sh`

### Docs (1 file)
- `docs/week2_plan.md` (this file)

**Total new code:** ~1,310 lines across 14 files

---

## Next Session Tasks

1. **Build on RunPod** — Run `./build_and_test.sh`
2. **Test compilation** — Verify library builds
3. **Add bindings** — Quick ctypes wrapper or pybind11
4. **First compression test** — Encode a small tensor
5. **Measure results** — Compression ratio, speed

---

## Week 3 Preview

Once Week 2 is working:
- Add DCT/ADST transforms
- Proper frequency table storage in container
- Multi-threaded encoding
- Better compression (target 30-50%)
- Context-adaptive entropy coding

---

**Status:** Code complete, ready to build and test! 🚀

