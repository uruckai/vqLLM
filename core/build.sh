#!/bin/bash
# Build script for core codec

set -e

echo "=============================================="
echo "Building Core Codec"
echo "=============================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found - nvcc not in PATH"
    exit 1
fi

echo "✓ CUDA found: $(nvcc --version | grep release | awk '{print $5}')"

# Create build directory
mkdir -p build
cd build

# Configure
echo ""
echo "Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building..."
make -j$(nproc)

# Check output
echo ""
echo "=============================================="
if [ -f "libcodec_core.so" ]; then
    echo "✓ Build successful!"
    ls -lh libcodec_core.so
    echo ""
    echo "Next steps:"
    echo "  cd .."
    echo "  python3 test_core.py"
else
    echo "❌ Build failed - library not found"
    exit 1
fi
echo "=============================================="

