# Core Codec - Quick Start

## On RunPod (RTX 5090)

### 1. Transfer Files

```bash
# From local machine
cd C:\Users\cfisc\OneDrive\Documents\CodecLLM
scp -r core root@<POD_IP>:/workspace/
```

Or use Git (if already set up):
```bash
cd /workspace
git clone git@github.com:cwfischer89-png/CodecLLM.git
cd CodecLLM/core
```

### 2. Build

```bash
cd /workspace/core  # or /workspace/CodecLLM/core
chmod +x build.sh
./build.sh
```

Expected output:
```
✓ CUDA found: 12.8
Configuring...
Building...
✓ Build successful!
-rwxr-xr-x 1 root root 1.2M libcodec_core.so
```

### 3. Test

```bash
python3 test_core.py
```

Expected output:
```
=== Test 1: Small Data (64x64) ===
Original: (64, 64), 4096 bytes
Compressed: 2048 bytes (2.00x)
Decode time: 0.123 ms
Bit-exact: True
✓ Test 1 PASSED

... (more tests)

Results: 4 passed, 0 failed
🎉 All tests passed!
```

## If Build Fails

### Missing dependencies
```bash
pip3 install numpy
```

### CUDA not found
```bash
which nvcc
# Should show: /usr/local/cuda/bin/nvcc

# If not, check CUDA installation
ls /usr/local/cuda*/
```

### Library not found (when running Python)
```bash
# Check library exists
ls -lh build/libcodec_core.so

# Update bindings.py path if needed
# (should auto-detect)
```

## Quick Validation

### Minimal test (no GPU needed)
```python
from bindings import Encoder
import numpy as np

data = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
encoder = Encoder()
compressed, ratio = encoder.encode(data)
print(f"Compression: {ratio:.2f}x")  # Should be ~1.5-2.0x
```

### GPU test
```python
from bindings import GPUDecoder

if GPUDecoder.is_available():
    print("✓ GPU available")
else:
    print("✗ GPU not available")
```

## What to Check

### After build
- [x] `build/libcodec_core.so` exists
- [x] File size ~1-2 MB
- [x] No compilation errors

### After test
- [x] All 4 tests pass
- [x] Bit-exact: True on all tests
- [x] Compression ratio: 1.5-3.0x
- [x] Decode time: < 1 ms for small data

## Next Steps

### If all tests pass ✓
1. Test on larger data (4096x4096)
2. Test on real LLM checkpoint
3. Profile GPU performance
4. Optimize bottlenecks

### If tests fail ✗
1. Check which test failed
2. Read error message
3. Common issues:
   - rANS decode error → frequency table bug
   - Not bit-exact → reconstruction bug
   - GPU error → memory/kernel launch issue

## Performance Targets

| Size | Compression | Decode Time | Throughput |
|------|-------------|-------------|------------|
| 64×64 | 2.0x | 0.1 ms | - |
| 256×256 | 2.5x | 0.5 ms | 100 MB/s |
| 4096×4096 | 2.5x | 1.0 ms | 16 GB/s |

If much slower, check:
- GPU actually being used? (check nvidia-smi)
- CUDA architecture correct? (sm_90 for 5090)
- Memory transfers? (should be minimal)

## File Structure

```
core/
├── format.h           # Format definitions
├── encoder.h/cpp      # Encoder (CPU)
├── decoder_host.h/cpp # Decoder (host)
├── decoder_gpu.cu     # Decoder (CUDA)
├── c_api.cpp          # C API
├── bindings.py        # Python bindings
├── test_core.py       # Tests
└── build/
    └── libcodec_core.so  # Built library
```

## Debugging

### Enable verbose output
```python
# In test_core.py, add after imports:
import sys
np.set_printoptions(threshold=sys.maxsize)
```

### Check GPU usage
```bash
watch -n 1 nvidia-smi
# Run test in another terminal
# Should see GPU memory/utilization spike
```

### Check CUDA errors
```bash
# In decoder_gpu.cu, errors printed to stderr
# Look for "CUDA kernel launch error: ..." messages
```

## Contact

If you get stuck, check:
1. `IMPLEMENTATION.md` - Technical details
2. `CORE_REBOOT.md` - Why we built this
3. Error messages - usually informative

## Summary

```bash
# Full workflow
cd /workspace/core
./build.sh
python3 test_core.py

# Should see:
# ✓ Build successful!
# 🎉 All tests passed!
```

That's it! If this works, we have a working codec.

