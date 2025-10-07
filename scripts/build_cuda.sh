#!/bin/bash
# Build with CUDA support

set -e

echo "=============================================="
echo "Building Weight Codec with CUDA Support"
echo "=============================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "✗ CUDA not found (nvcc not in PATH)"
    echo ""
    echo "Building CPU-only version..."
    mkdir -p build
    cd build
    cmake .. -DWCODEC_ENABLE_CUDA=OFF
    make -j$(nproc)
    exit 0
fi

# CUDA available
CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo "✓ CUDA found: version $CUDA_VERSION"
echo ""

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU devices:"
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | \
        awk '{print "  - " $0}'
    echo ""
fi

# Configure with CUDA
echo "Configuring with CUDA support..."
mkdir -p build
cd build

cmake .. \
    -DWCODEC_ENABLE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"  # Ampere, Ada, Hopper, Blackwell

echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=============================================="
echo "✓ Build complete!"
echo "=============================================="
echo ""
echo "Library: build/libwcodec.so"
echo ""
echo "Test with:"
echo "  python3 tests/test_gpu_decoder.py"
echo "  python3 tests/benchmark_decode.py"

