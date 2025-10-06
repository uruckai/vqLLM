#!/bin/bash
# Rebuild with C API and run integration tests

set -e

echo "========================================="
echo "Rebuilding with C API bindings"
echo "========================================="

cd build

echo ""
echo "[1/3] Rebuilding..."
make clean
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
make -j$(nproc) || make -j8

echo ""
echo "[2/3] Checking library..."
if [ -f "libwcodec.so" ]; then
    echo "✓ libwcodec.so built successfully"
    ls -lh libwcodec.so
else
    echo "✗ Build failed"
    exit 1
fi

cd ..

echo ""
echo "[3/3] Running integration tests..."
python tests/test_roundtrip.py

echo ""
echo "========================================="
echo "✓ Week 2 Complete!"
echo "========================================="
echo ""
echo "Summary:"
echo "  - C++ encoder/decoder: ✓ Working"
echo "  - Python bindings: ✓ Working"
echo "  - Bit-exact reconstruction: ✓ Verified"
echo "  - Compression working: ✓ Measured"
echo ""
echo "Next: Week 3 - Add transforms, better container format"

