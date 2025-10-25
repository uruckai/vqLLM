# Fused Kernel New Project Setup

## Recommended Files to Copy

Based on workspace analysis, here's what to copy for a clean, focused fused kernel implementation project:

### 📋 Essential Files (Copy These)

#### **Core Documentation**
- `docs/FUSED_KERNEL_IMPLEMENTATION_GUIDE.md` - Strategic roadmap and implementation guide
- `docs/FUSED_KERNEL_TECHNICAL_SPEC.md` - Detailed technical specifications and architecture
- `docs/FUSED_KERNEL_ANALYSIS.md` - Analysis of why fused kernels work and performance benefits
- `docs/SPEED_OPTIMIZATION_GUIDE.md` - Performance optimization strategies and benchmarks

#### **Working Implementation Reference**
- `core_rans/` - Complete working rANS implementation (simplified, GPU-enabled)
  - `CMakeLists.txt` - Build configuration
  - `c_api.cpp` - C API interface
  - `decoder_gpu.cu` - GPU decompression kernel
  - `decoder_host.cpp/h` - Host-side decoder
  - `encoder_simple.cpp` - CPU encoder
  - `encoder.cpp/h` - Encoder interface
  - `format.h` - Data format definitions
  - `rans.cpp/h` - rANS algorithm implementation

#### **Build System & Dependencies**
- `CMakeLists.txt` (root) - Main build configuration
- `requirements.txt` - Python dependencies (PyTorch, CUDA, etc.)
- `SETUP.md` - Installation and setup instructions

#### **Project Context**
- `README.md` - Project overview and objectives (keep relevant sections)

### 📋 Optional Reference Files (Copy If Needed)

#### **Python Bindings** (if you need Python integration)
- `python/wcodec/` - Python bindings for compression library
  - `__init__.py`, `bindings.py`, `decoder.py`, `encoder.py`, etc.

#### **Advanced Compression Internals** (for deep customization)
- `cpp/src/` and `cpp/include/` - Full compression library implementation
  - May be useful if you need to extend or modify the compression algorithm

### 🚫 Files to EXCLUDE (Irrelevant for New Project)

#### **Experimental Tests** (Don't copy - create new tests)
- All `test_*.py` files (21+ test files)
- These are experimental validation scripts, not core implementation

#### **Old Implementations** (Don't copy)
- `core/` directory - Old Zstd-based implementation
- `core/archive/` - Backup files
- `cpp/src/` (if not needed) - Complex implementation with containers/transforms

#### **Project Management** (Don't copy)
- `docs/DEVELOPMENT_HISTORY.md` - Historical development log
- `docs/PROJECT_PLAYBOOK.md` - Project management and status
- `docs/integration_guide.md` - Integration documentation (outdated)
- Various shell scripts (`setup.sh`, `restore_working_rans.sh`)
- `CodecLLMDiscussion.txt` - Discussion logs
- `codecllmkey.txt` - API keys

## 🏗️ New Project Structure

### Recommended Directory Layout

```
fused_kernel_project/
├── docs/
│   ├── FUSED_KERNEL_IMPLEMENTATION_GUIDE.md    # Strategic guide
│   ├── FUSED_KERNEL_TECHNICAL_SPEC.md         # Technical specs
│   ├── FUSED_KERNEL_ANALYSIS.md               # Performance analysis
│   └── SPEED_OPTIMIZATION_GUIDE.md             # Optimization guide
├── src/
│   └── rans/                                    # Compression library
│       ├── CMakeLists.txt
│       ├── c_api.cpp
│       ├── decoder_gpu.cu
│       ├── decoder_host.cpp
│       ├── decoder_host.h
│       ├── encoder_simple.cpp
│       ├── encoder.cpp
│       ├── encoder.h
│       ├── format.h
│       ├── rans.cpp
│       └── rans.h
├── CMakeLists.txt                               # Build configuration
├── requirements.txt                             # Dependencies
├── README.md                                    # Project overview
└── SETUP.md                                     # Installation guide
```

## 🔧 Dependencies to Include

From `requirements.txt`, ensure you have:
```
torch>=2.0.0
transformers>=4.20.0
numpy
cmake>=3.18
```

**CUDA Requirements:**
- CUDA Toolkit 11.8+ (for RTX 5090 compatibility)
- cuDNN 8.0+
- Python 3.8+

## 🚀 Quick Start Commands

After copying files:

```bash
# Build the compression library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Verify installation
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 📊 What This Gives You

### ✅ **Complete Foundation**
- Working GPU-parallelized rANS compression
- Comprehensive implementation guides
- Performance optimization strategies
- Technical specifications for fused kernels

### ✅ **Clean Starting Point**
- No experimental test files cluttering the workspace
- No outdated implementations
- Focused on fused kernel objectives

### ✅ **Ready for Development**
- Build system configured for CUDA development
- Python integration ready
- Documentation for implementation decisions

## 🎯 Next Steps After Setup

1. **Review the implementation guides** to understand the technical approach
2. **Study the rANS implementation** in `src/rans/` for compression details
3. **Set up development environment** with CUDA and PyTorch
4. **Start with basic fused kernel POC** (single tile, single layer)
5. **Scale to multi-layer** using existing parallelization insights

This setup gives you everything needed to implement fused kernels while avoiding the complexity of the experimental codebase.

