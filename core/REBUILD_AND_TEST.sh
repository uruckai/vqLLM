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
echo "[2/4] Finding nvCOMP..."
# Find nvCOMP library
NVCOMP_LIB=$(find /usr/lib /usr/local/lib -name "libnvcomp.so" 2>/dev/null | head -n 1)
if [ -z "$NVCOMP_LIB" ]; then
    echo "❌ nvCOMP library not found!"
    echo "Searched: /usr/lib, /usr/local/lib"
    exit 1
fi
echo "Found nvCOMP: $NVCOMP_LIB"

# Find nvCOMP include directory
NVCOMP_INC=$(find /usr/include /usr/local/include -name "nvcomp" -type d 2>/dev/null | head -n 1)
if [ -z "$NVCOMP_INC" ]; then
    NVCOMP_INC=$(dirname $(find /usr/include /usr/local/include -name "nvcomp.hpp" 2>/dev/null | head -n 1))
fi
if [ -z "$NVCOMP_INC" ]; then
    echo "⚠️  nvCOMP include dir not found, using /usr/local/include"
    NVCOMP_INC="/usr/local/include"
else
    echo "Found nvCOMP include: $NVCOMP_INC"
fi

echo ""
echo "[3/4] Building codec..."
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCOMP_INCLUDE_DIR="$NVCOMP_INC" \
  -DNVCOMP_LIBRARY="$NVCOMP_LIB"

make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "✓ Build complete!"
echo ""

cd ..

echo "[4/4] Testing round-trip..."
python test_roundtrip.py
echo ""

echo "[5/5] Testing LLM inference..."
python test_zstd_inference.py

echo ""
echo "========================================="
echo "  ALL TESTS COMPLETE"
echo "========================================="
