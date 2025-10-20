# nvCOMP Installation Guide for RunPod

## 🚀 Quick Install (Recommended - Python Package)

This is the **easiest method** and works with our C++ code:

```bash
# Check your CUDA version first
nvcc --version

# For CUDA 11.x (most RunPod instances)
pip install nvidia-nvcomp-cu11

# For CUDA 12.x
pip install nvidia-nvcomp-cu12
```

**Then set the environment variable:**
```bash
# Find where pip installed it
python3 -c "import nvidia.nvcomp; import os; print(os.path.dirname(nvidia.nvcomp.__file__))"

# Example output: /usr/local/lib/python3.10/dist-packages/nvidia/nvcomp

# Set NVCOMP_ROOT (adjust path based on output above)
export NVCOMP_ROOT=/usr/local/lib/python3.10/dist-packages/nvidia/nvcomp
export LD_LIBRARY_PATH=$NVCOMP_ROOT/lib:$LD_LIBRARY_PATH
```

---

## 🔧 Manual Install (Alternative)

If the Python package doesn't work, use the latest binary release:

### Step 1: Check Your CUDA Version

```bash
nvcc --version
# Look for: "release 11.8" or "release 12.x"
```

### Step 2: Download nvCOMP 4.1.0 (Latest)

**For CUDA 11.x:**
```bash
cd /workspace
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-4.1.0.6_cuda11-archive.tar.xz
tar -xf nvcomp-linux-x86_64-4.1.0.6_cuda11-archive.tar.xz
mv nvcomp-linux-x86_64-4.1.0.6_cuda11-archive nvcomp_install
```

**For CUDA 12.x:**
```bash
cd /workspace
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-4.1.0.6_cuda12-archive.tar.xz
tar -xf nvcomp-linux-x86_64-4.1.0.6_cuda12-archive.tar.xz
mv nvcomp-linux-x86_64-4.1.0.6_cuda12-archive nvcomp_install
```

### Step 3: Set Environment Variables

```bash
export NVCOMP_ROOT=/workspace/nvcomp_install
export LD_LIBRARY_PATH=$NVCOMP_ROOT/lib:$LD_LIBRARY_PATH

# Make it permanent (add to .bashrc)
echo 'export NVCOMP_ROOT=/workspace/nvcomp_install' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$NVCOMP_ROOT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## 🧪 Verify Installation

```bash
# Check if library exists
ls $NVCOMP_ROOT/lib/libnvcomp.so

# Check if header exists
ls $NVCOMP_ROOT/include/nvcomp.hpp

# If both exist, you're good!
```

---

## 🏗️ Build Codec Library

```bash
cd /workspace/CodecLLM/core
rm -rf build
mkdir build
cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# You should see:
# ✓ nvCOMP found: /path/to/libnvcomp.so

# Build
cmake --build . -j$(nproc)
```

---

## 🐛 Troubleshooting

### "nvCOMP not found" during CMake

**Solution 1:** Check NVCOMP_ROOT is set:
```bash
echo $NVCOMP_ROOT
# Should print: /workspace/nvcomp_install (or similar)
```

**Solution 2:** Manual CMake hints:
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCOMP_INCLUDE_DIR=$NVCOMP_ROOT/include \
  -DNVCOMP_LIBRARY=$NVCOMP_ROOT/lib/libnvcomp.so
```

### "libnvcomp.so: cannot open shared object file" at runtime

**Solution:** Add library path:
```bash
export LD_LIBRARY_PATH=$NVCOMP_ROOT/lib:$LD_LIBRARY_PATH
# Or for Python package:
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nvcomp/lib:$LD_LIBRARY_PATH
```

### Still not working?

**CPU Fallback:** Our code automatically falls back to CPU Zstd decode if nvCOMP isn't available. It's slower (50ms vs 20ms per layer) but still **7x faster than rANS**!

```bash
# Check if CPU fallback is working
cd /workspace/CodecLLM/core
python3 bindings_zstd.py

# You should see:
# GPU decoder available: False
# (but tests still pass)
```

---

## 📊 Performance Comparison

| nvCOMP Status | Decode Speed | vs rANS | Usable? |
|---------------|--------------|---------|---------|
| GPU (nvCOMP) | 20ms/layer | 17x faster | ✅ Best |
| CPU fallback | 50ms/layer | 7x faster | ✅ Good |
| rANS GPU | 348ms/layer | 1x baseline | ⚠️ Slow |

**Bottom line:** Even without nvCOMP, Zstd is still much faster than rANS!

---

## 🔗 Official Resources

- [nvCOMP Downloads](https://developer.nvidia.com/nvcomp-download-archive)
- [nvCOMP Documentation](https://docs.nvidia.com/cuda/nvcomp/)
- [nvCOMP GitHub](https://github.com/NVIDIA/nvcomp)
- [Python Package (PyPI)](https://pypi.org/project/nvidia-nvcomp-cu11/)

---

**Created:** October 20, 2025  
**Last Updated:** October 20, 2025

