#!/bin/bash

echo "========================================="
echo "  NVCOMP 3.0.6 INSTALLATION"
echo "========================================="
echo ""

# Check if already installed
if [ -f "/usr/local/lib/libnvcomp.so" ]; then
    echo "✓ nvCOMP already installed at /usr/local/lib/libnvcomp.so"
    ls -la /usr/local/lib/libnvcomp*
    exit 0
fi

echo "[1/3] Downloading nvCOMP 3.0.6 binaries..."
cd /tmp
wget -q https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz

if [ ! -f "nvcomp_3.0.6_x86_64_12.x.tgz" ]; then
    echo "❌ Download failed! Trying alternate URL..."
    # Try GitHub releases
    wget -q https://github.com/NVIDIA/nvcomp/releases/download/v3.0.6/nvcomp_3.0.6_x86_64_12.x.tgz
fi

if [ ! -f "nvcomp_3.0.6_x86_64_12.x.tgz" ]; then
    echo "❌ Download failed from all sources!"
    echo "Please download manually from:"
    echo "https://developer.nvidia.com/nvcomp-download"
    exit 1
fi

echo ""
echo "[2/3] Extracting and installing..."
mkdir -p nvcomp_extract
cd nvcomp_extract
tar -xzf ../nvcomp_3.0.6_x86_64_12.x.tgz

# Copy to system directories
sudo cp -r lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/

# Update library cache
sudo ldconfig

echo ""
echo "[3/3] Verifying installation..."
if [ -f "/usr/local/lib/libnvcomp.so" ]; then
    echo "✓ nvCOMP installed successfully!"
    ls -la /usr/local/lib/libnvcomp*
    ls -la /usr/local/include/nvcomp/
else
    echo "❌ Installation failed!"
    exit 1
fi

# Cleanup
cd /tmp
rm -rf nvcomp_extract nvcomp_3.0.6_x86_64_12.x.tgz

echo ""
echo "========================================="
echo "✓ nvCOMP 3.0.6 ready!"
echo "========================================="

