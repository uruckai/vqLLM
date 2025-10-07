# Core Codec - Minimal Working Implementation

## Goal
Reduce LLM memory usage using video codec techniques:
- **Predictive coding** (intra-prediction like VP9/AV1)
- **Entropy coding** (rANS)
- **GPU acceleration** for fast decode

## What This Does
1. **Encoder** (CPU): Compress INT8 weight matrices 2-3x
2. **Decoder** (GPU): Decompress 100x+ faster than CPU
3. **Bit-exact**: Lossless compression, perfect reconstruction

## Architecture

```
INT8 Weights (4096x4096)
         ↓
    [Encoder CPU]
    - Tile 16x16 blocks
    - Intra-predict (LEFT/TOP/AVG/PLANAR)
    - rANS entropy code
         ↓
Compressed (~50% size)
         ↓
    [Decoder GPU]
    - Parse tiles (parallel)
    - rANS decode (CUDA)
    - Reconstruct (CUDA)
         ↓
INT8 Weights (reconstructed, bit-exact)
```

## Files
- `encoder.cpp/h` - CPU encoder (predictor + rANS)
- `decoder_gpu.cu` - GPU decoder kernel
- `decoder_host.cpp/h` - GPU decoder host code
- `format.h` - Simple binary format
- `bindings.py` - Python interface
- `test_core.py` - End-to-end test
- `CMakeLists.txt` - Build system

## Build & Test

```bash
mkdir build && cd build
cmake .. && make -j8
cd ..
python3 test_core.py
```

## Expected Results
- Compression: 2.0-2.5x on typical LLM weights
- GPU decode: 100-500x faster than CPU
- Bit-exact: 100% match with original

