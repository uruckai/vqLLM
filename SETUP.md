# CodecLLM Setup Guide

Complete setup instructions for running CodecLLM on RunPod or any Ubuntu/CUDA system.

---

## 🔧 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.5
  - Tested on: RTX 5090, RTX 4090, A100, H100
  - Minimum: RTX 3090 or better
- **VRAM**: 8GB minimum (16GB+ recommended)
- **RAM**: 16GB system RAM
- **Storage**: 10GB free space

### Software
- **OS**: Ubuntu 20.04+ (or compatible Linux)
- **CUDA**: 11.8 or 12.x
- **Python**: 3.8+
- **CMake**: 3.18+
- **GCC/G++**: 9.0+

---

## 📦 Dependencies

### System Packages
```bash
apt-get install -y git cmake build-essential wget
```

### Python Libraries
- `torch` (with CUDA support)
- `transformers` (≥4.30.0)
- `accelerate`
- `numpy`
- `hf-transfer` (optional, for faster model downloads)

### External Libraries

#### nvCOMP 3.0.6 (REQUIRED)
NVIDIA's GPU compression library for Zstd acceleration.

**Download**: https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz

**Installation**:
```bash
wget https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz
tar -xzf nvcomp_3.0.6_x86_64_12.x.tgz
cp -r nvcomp_3.0.6_x86_64_12.x/include/* /usr/local/include/
cp -r nvcomp_3.0.6_x86_64_12.x/lib/* /usr/local/lib/
ldconfig
```

---

## 🚀 Quick Setup (Automated)

### On Fresh RunPod Instance:

```bash
cd /workspace
git clone https://github.com/uruckai/vqLLM.git CodecLLM
cd CodecLLM
chmod +x setup.sh
./setup.sh
```

That's it! The script will:
1. Install all system dependencies
2. Download and install nvCOMP
3. Install Python packages
4. Build the codec library
5. Verify the installation

---

## 🔨 Manual Setup

If you prefer to set up manually or the automated script fails:

### Step 1: Clone Repository
```bash
cd /workspace
git clone https://github.com/uruckai/vqLLM.git CodecLLM
cd CodecLLM
```

### Step 2: Install System Dependencies
```bash
apt-get update
apt-get install -y git cmake build-essential wget
```

### Step 3: Install nvCOMP
```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz
tar -xzf nvcomp_3.0.6_x86_64_12.x.tgz
cp -r nvcomp_3.0.6_x86_64_12.x/include/* /usr/local/include/
cp -r nvcomp_3.0.6_x86_64_12.x/lib/* /usr/local/lib/
ldconfig
rm -rf nvcomp_3.0.6_x86_64_12.x*
```

### Step 4: Install Python Dependencies
```bash
pip install torch transformers accelerate numpy hf-transfer
```

### Step 5: Build Codec Library
```bash
cd /workspace/CodecLLM/core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Verify the build:
```bash
ls -la libcodec_core.so
# Should see: libcodec_core.so (around 200-300KB)
```

### Step 6: Verify Installation
```bash
cd /workspace/CodecLLM/core
python3 -c "
import torch
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print('Codec library loaded successfully!')
"
```

---

## ✅ Verification

After setup, run the verification script:

```bash
cd /workspace/CodecLLM/core
python3 check_runpod_status.py
```

You should see:
```
✓ CUDA available
✓ GPU detected
✓ Codec library found
✓ Encoder/Decoder working
✓ Compression round-trip passed
```

---

## 🧪 Running Tests

### Basic Test (No Cache)
```bash
cd /workspace/CodecLLM/core
python3 test_no_kv_cache.py
```

### FP32 KV Cache Test (Recommended)
```bash
cd /workspace/CodecLLM/core
python3 test_fp32_kv_cache.py
```

### Filter Verbose Logs
```bash
python3 test_fp32_kv_cache.py 2>&1 | grep -vE "ENCODER|DECODER"
```

---

## 🐛 Troubleshooting

### Issue: "Cannot find codec library"
**Solution**: The codec library wasn't built. Run:
```bash
cd /workspace/CodecLLM/core/build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
ls -la libcodec_core.so  # Verify it exists
```

### Issue: "GPU decoding failed"
**Solution**: nvCOMP not installed or not in library path. Run:
```bash
ldconfig -p | grep nvcomp
# Should show libnvcomp.so and libnvcomp_device.so
# If not, reinstall nvCOMP (see Step 3)
```

### Issue: "No module named 'hf_transfer'"
**Solution**: Disable fast transfer or install the package:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
# OR
pip install hf-transfer
```

### Issue: Build fails with "private within this context"
**Solution**: This is the old rANS codec (not used). Build from `core/` not `cpp/`:
```bash
cd /workspace/CodecLLM/core/build
cmake .. && make
```

### Issue: Out of memory during generation
**Solution**: Reduce batch size or sequence length:
```python
outputs = model.generate(..., max_new_tokens=10)  # Start small
```

---

## 🔄 Updating

To update to the latest version:

```bash
cd /workspace/CodecLLM
git pull
cd core/build
make clean
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 📊 Expected Performance

### TinyLlama 1.1B on RTX 5090

| Configuration | Memory | Speed | Quality |
|--------------|--------|-------|---------|
| Baseline (uncompressed) | 2.1 GB | 100% | Perfect |
| FP32 KV Cache (attention compressed) | 1.7 GB (-19%) | ~85% | Perfect |
| All layers compressed | 1.2 GB (-43%) | ~70% | Perfect (with FP32 cache) |

---

## 🎯 Next Steps

After successful setup:

1. **Run baseline test**: `python3 test_no_kv_cache.py`
2. **Test FP32 cache**: `python3 test_fp32_kv_cache.py`
3. **Check results**: Look for "✓✓✓ PERFECT MATCH! ✓✓✓"
4. **Experiment**: Try different compression levels, layer counts
5. **Scale up**: Test on larger models (7B, 13B, 70B)

---

## 📝 File Structure

```
CodecLLM/
├── SETUP.md                  # This file
├── setup.sh                  # Automated setup script
├── core/
│   ├── bindings_zstd.py      # Python bindings
│   ├── encoder_zstd.cpp      # Compression implementation
│   ├── decoder_zstd.cpp      # Decompression implementation
│   ├── c_api_zstd.cpp        # C API wrapper
│   ├── CMakeLists.txt        # Build configuration
│   ├── build/
│   │   └── libcodec_core.so  # Built library (after make)
│   ├── test_fp32_kv_cache.py # FP32 cache test
│   ├── test_no_kv_cache.py   # No cache test
│   └── check_runpod_status.py # Verification script
└── cpp/                      # Old rANS codec (not used)
```

---

## 💬 Support

If you encounter issues:

1. Check this document's troubleshooting section
2. Verify all dependencies are installed: `./setup.sh --verify-only`
3. Check the build log for errors
4. Ensure GPU is visible: `nvidia-smi`
5. Test with minimal example: `python3 check_runpod_status.py`

---

## 📜 Version History

- **v1.0** (Oct 2024): Initial Zstd implementation
- **v1.1** (Oct 2024): FP32 KV cache solution
- **Current**: Complete automated setup

---

## ⚖️ License

See LICENSE file in repository root.

