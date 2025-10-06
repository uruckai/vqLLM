#!/bin/bash
# Build script for Week 2

set -e  # Exit on error

echo "========================================="
echo "Building Weight Codec (Week 2)"
echo "========================================="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo ""
echo "[1/4] Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DBUILD_TESTS=ON

# Build
echo ""
echo "[2/4] Building C++ library..."
make -j$(nproc) || make -j8

# Check if library was built
if [ -f "libwcodec.so" ] || [ -f "libwcodec.dylib" ] || [ -f "wcodec.dll" ]; then
    echo "✓ Library built successfully"
else
    echo "✗ Library build failed"
    exit 1
fi

cd ..

# Install Python package
echo ""
echo "[3/4] Installing Python package..."
cd python
pip install -e . --quiet

# Test
echo ""
echo "[4/4] Running tests..."
cd ..
python tests/test_predictor.py

echo ""
echo "========================================="
echo "✓ Build complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  - Add pybind11 bindings for Python integration"
echo "  - Test on real checkpoints"
echo "  - Measure compression ratio"

