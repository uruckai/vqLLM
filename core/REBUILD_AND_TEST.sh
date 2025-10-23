#!/bin/bash

echo "========================================="
echo "  REBUILD & TEST PIPELINE"
echo "========================================="
echo ""

# Set library path for nvCOMP
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

cd /workspace/CodecLLM/core

echo "[1/4] Cleaning old build..."
rm -rf build
mkdir -p build

echo ""
echo "[2/4] Building codec..."
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCOMP_INCLUDE_DIR=/usr/local/include \
  -DNVCOMP_LIBRARY=/usr/local/lib/libnvcomp.so

make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "✓ Build complete!"
echo ""

cd ..

echo "[3/4] Testing round-trip..."
python test_roundtrip.py
echo ""

echo "[4/4] Testing LLM inference..."
python test_zstd_inference.py

echo ""
echo "========================================="
echo "  ALL TESTS COMPLETE"
echo "========================================="
