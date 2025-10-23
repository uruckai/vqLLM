#!/bin/bash
# Rebuild codec with nvCOMP 3.0.6

set -e

echo "========================================="
echo "Rebuilding codec with nvCOMP 3.0.6"
echo "========================================="

cd /workspace/CodecLLM/core

# Clean build
echo "Cleaning old build..."
rm -rf build
mkdir build
cd build

# Configure with nvCOMP 3.0.6
echo "Configuring CMake..."
cmake .. \
  -DNVCOMP_INCLUDE_DIR=/usr/local/include \
  -DNVCOMP_LIBRARY=/usr/local/lib/libnvcomp.so

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "========================================="
echo "✓ Build complete!"
echo "========================================="
echo ""
echo "Library location: /workspace/CodecLLM/core/build/libcodec_core.so"
echo ""
echo "Next steps:"
echo "  1. Test basic GPU decode:"
echo "     python test_gpu_direct_simple.py"
echo ""
echo "  2. Test full inference:"
echo "     python test_zstd_inference.py"
echo ""

