#!/bin/bash

echo "================================================================================"
echo "  COMPLETE SETUP AND TEST PIPELINE"
echo "================================================================================"
echo ""

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

cd /workspace/CodecLLM/core

# Step 0: Install dependencies
echo "========================================="
echo "STEP 0: Installing dependencies"
echo "========================================="
echo ""

echo "Installing libzstd-dev..."
sudo apt-get update -qq
sudo apt-get install -y libzstd-dev

if [ $? -ne 0 ]; then
    echo "❌ Failed to install libzstd-dev!"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Step 1: Check/Install nvCOMP
echo "========================================="
echo "STEP 1: Checking nvCOMP installation"
echo "========================================="
echo ""

NVCOMP_LIB=$(find /usr/lib /usr/local/lib -name "libnvcomp.so" 2>/dev/null | head -n 1)

if [ -z "$NVCOMP_LIB" ]; then
    echo "nvCOMP not found. Installing..."
    bash SETUP_NVCOMP.sh
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ nvCOMP installation failed!"
        echo ""
        echo "Manual installation steps:"
        echo "  1. Download from: https://developer.nvidia.com/nvcomp-download"
        echo "  2. Or use: wget https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz"
        echo "  3. Extract: tar -xzf nvcomp_3.0.6_x86_64_12.x.tgz"
        echo "  4. Install: sudo cp -r lib/* /usr/local/lib/ && sudo cp -r include/* /usr/local/include/"
        echo "  5. Update cache: sudo ldconfig"
        exit 1
    fi
    
    # Re-check after installation
    NVCOMP_LIB=$(find /usr/lib /usr/local/lib -name "libnvcomp.so" 2>/dev/null | head -n 1)
    if [ -z "$NVCOMP_LIB" ]; then
        echo "❌ nvCOMP still not found after installation!"
        exit 1
    fi
fi

echo "✓ nvCOMP found: $NVCOMP_LIB"
echo ""

# Step 2: Build codec
echo "========================================="
echo "STEP 2: Building codec"
echo "========================================="
echo ""

rm -rf build
mkdir -p build
cd build

# Find paths
NVCOMP_INC=$(find /usr/include /usr/local/include -name "nvcomp" -type d 2>/dev/null | head -n 1)
if [ -z "$NVCOMP_INC" ]; then
    NVCOMP_INC="/usr/local/include"
fi

echo "Using:"
echo "  Library: $NVCOMP_LIB"
echo "  Include: $NVCOMP_INC"
echo ""

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCOMP_INCLUDE_DIR="$NVCOMP_INC" \
  -DNVCOMP_LIBRARY="$NVCOMP_LIB"

make -j$(nproc)

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed!"
    echo ""
    echo "Check build/CMakeFiles/codec_core.dir/ for error details"
    exit 1
fi

echo ""
echo "✓ Build complete!"
echo ""

cd ..

# Step 3: Test round-trip
echo "========================================="
echo "STEP 3: Testing compression round-trip"
echo "========================================="
echo ""

python test_roundtrip.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Round-trip test failed!"
    echo "This means compression/decompression is not working correctly."
    exit 1
fi

echo ""
echo "✓ Round-trip test passed!"
echo ""

# Step 4: Test LLM inference
echo "========================================="
echo "STEP 4: Testing LLM inference"
echo "========================================="
echo ""

python test_zstd_inference.py

echo ""
echo "================================================================================"
echo "  ALL TESTS COMPLETE!"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  1. nvCOMP: Installed and working"
echo "  2. Codec: Built successfully"
echo "  3. Round-trip: BIT-EXACT compression/decompression"
echo "  4. LLM test: Check output above for results"
echo ""
echo "Expected LLM results:"
echo "  - Output should match baseline (no garbage)"
echo "  - Compression: ~2.5-3.0x"
echo "  - Time: ~60-135s (slower than baseline, but with low VRAM)"
echo ""

