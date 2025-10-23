#!/bin/bash

echo "========================================="
echo "  INSTALL nvCOMP 3.0.6 + BUILD + TEST"
echo "========================================="
echo ""

# Check if nvCOMP is already installed
echo "[1/6] Checking for existing nvCOMP..."
if [ -f "/usr/local/lib/libnvcomp.so" ] || [ -f "/usr/lib/x86_64-linux-gnu/libnvcomp.so" ]; then
    echo "✓ nvCOMP already installed"
else
    echo "nvCOMP not found, installing..."
    
    # Download nvCOMP 3.0.6 binaries
    echo ""
    echo "[2/6] Downloading nvCOMP 3.0.6..."
    cd /tmp
    wget -q https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz
    
    if [ $? -ne 0 ]; then
        echo "❌ Download failed, trying alternate URL..."
        # If direct download fails, try GitHub releases (though unlikely)
        wget -q https://github.com/NVIDIA/nvcomp/archive/refs/tags/v3.0.1.tar.gz -O nvcomp-3.0.1.tar.gz
        
        if [ $? -ne 0 ]; then
            echo "❌ All download methods failed!"
            echo "Manual installation required:"
            echo "  Visit: https://developer.nvidia.com/nvcomp-download"
            echo "  Download nvCOMP 3.0.6 for CUDA 12.x"
            exit 1
        fi
        
        # Build from source
        echo "Building from source..."
        tar -xzf nvcomp-3.0.1.tar.gz
        cd nvcomp-3.0.1
        mkdir build && cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF
        make -j$(nproc)
        sudo make install
    else
        # Extract binaries
        echo "[3/6] Extracting..."
        mkdir -p /tmp/nvcomp_extract
        cd /tmp/nvcomp_extract
        tar -xzf /tmp/nvcomp_3.0.6_x86_64_12.x.tgz
        
        echo "[4/6] Installing to /usr/local..."
        sudo cp -r lib/* /usr/local/lib/
        sudo cp -r include/* /usr/local/include/
        
        echo "[5/6] Updating library cache..."
        sudo ldconfig
    fi
    
    # Verify installation
    if [ -f "/usr/local/lib/libnvcomp.so" ]; then
        echo "✓ nvCOMP installed successfully"
    else
        echo "❌ Installation verification failed!"
        exit 1
    fi
fi

# Set library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Now build codec
echo ""
echo "[6/6] Building codec..."
cd /workspace/CodecLLM/core

rm -rf build
mkdir -p build
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

# Run tests
echo "========================================="
echo "  RUNNING TESTS"
echo "========================================="
echo ""

echo "Test 1: Round-trip..."
python test_roundtrip.py
echo ""

echo "Test 2: LLM inference..."
python test_zstd_inference.py

echo ""
echo "========================================="
echo "  ALL TESTS COMPLETE"
echo "========================================="

