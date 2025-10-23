#!/bin/bash

echo "========================================="
echo "  COMPLETE SETUP & BUILD"
echo "========================================="
echo ""

# Set library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

cd /workspace/CodecLLM/core

# Check if nvCOMP is already installed
echo "[1/6] Checking for nvCOMP..."
NVCOMP_LIB=$(find /usr/lib /usr/local/lib -name "libnvcomp.so" 2>/dev/null | head -n 1)

if [ -z "$NVCOMP_LIB" ]; then
    echo "nvCOMP not found. Installing nvCOMP 3.0.6..."
    echo ""
    
    echo "[2/6] Downloading nvCOMP 3.0.6..."
    cd /workspace
    wget -q https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz
    
    if [ $? -ne 0 ]; then
        echo "❌ Download failed! Trying alternative..."
        # Try direct link
        wget -q https://github.com/NVIDIA/nvcomp/releases/download/v3.0.6/nvcomp_3.0.6_x86_64_12.x.tgz
    fi
    
    if [ ! -f nvcomp_3.0.6_x86_64_12.x.tgz ]; then
        echo "❌ Could not download nvCOMP!"
        echo "Please install manually from: https://developer.nvidia.com/nvcomp"
        exit 1
    fi
    
    echo "[3/6] Extracting nvCOMP..."
    tar -xzf nvcomp_3.0.6_x86_64_12.x.tgz
    
    echo "[4/6] Installing nvCOMP to /usr/local..."
    sudo cp -r lib/* /usr/local/lib/
    sudo cp -r include/* /usr/local/include/
    sudo ldconfig
    
    # Verify installation
    NVCOMP_LIB=$(find /usr/lib /usr/local/lib -name "libnvcomp.so" 2>/dev/null | head -n 1)
    if [ -z "$NVCOMP_LIB" ]; then
        echo "❌ nvCOMP installation failed!"
        exit 1
    fi
    
    echo "✓ nvCOMP installed: $NVCOMP_LIB"
    
    cd /workspace/CodecLLM/core
else
    echo "✓ nvCOMP already installed: $NVCOMP_LIB"
    echo ""
fi

echo ""
echo "[5/6] Building codec..."
rm -rf build
mkdir -p build
cd build

# Find nvCOMP paths
NVCOMP_LIB=$(find /usr/lib /usr/local/lib -name "libnvcomp.so" 2>/dev/null | head -n 1)
NVCOMP_INC=$(find /usr/include /usr/local/include -name "nvcomp" -type d 2>/dev/null | head -n 1)
if [ -z "$NVCOMP_INC" ]; then
    NVCOMP_INC=$(dirname $(find /usr/include /usr/local/include -name "nvcomp.hpp" 2>/dev/null | head -n 1))
fi

echo "nvCOMP library: $NVCOMP_LIB"
echo "nvCOMP include: $NVCOMP_INC"
echo ""

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

echo "[6/6] Running tests..."
echo ""
echo "--- Round-trip test ---"
python test_roundtrip.py
echo ""

echo "--- LLM inference test ---"
python test_zstd_inference.py

echo ""
echo "========================================="
echo "  ALL DONE!"
echo "========================================="

