#!/bin/bash
set -e

echo "================================"
echo "Downgrading to nvCOMP 3.0.6"
echo "================================"
echo ""

echo "[1/4] Removing nvCOMP 5.0..."
sudo apt-get remove -y nvcomp
sudo apt-get autoremove -y

echo ""
echo "[2/4] Installing nvCOMP 3.0.6..."
cd /tmp
# Try to download from NVIDIA
wget https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp-local-repo-ubuntu2204-3.0.6_3.0.6-1_amd64.deb || \
wget https://developer.download.nvidia.com/compute/nvcomp/redist/libnvcomp/linux-x86_64/libnvcomp-linux-x86_64-3.0.6_cuda12-archive.tar.xz || \
echo "⚠ Download failed - will try building from source"

if [ -f "libnvcomp-linux-x86_64-3.0.6_cuda12-archive.tar.xz" ]; then
    echo "Found tarball, extracting..."
    tar -xf libnvcomp-linux-x86_64-3.0.6_cuda12-archive.tar.xz
    sudo cp -r libnvcomp-linux-x86_64-3.0.6_cuda12-archive/include/* /usr/include/
    sudo cp -r libnvcomp-linux-x86_64-3.0.6_cuda12-archive/lib/* /usr/lib/x86_64-linux-gnu/
    sudo ldconfig
elif [ -f "nvcomp-local-repo-ubuntu2204-3.0.6_3.0.6-1_amd64.deb" ]; then
    echo "Found .deb package, installing..."
    sudo dpkg -i nvcomp-local-repo-ubuntu2204-3.0.6_3.0.6-1_amd64.deb
    sudo cp /var/nvcomp-local-repo-ubuntu2204-3.0.6/nvcomp-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get install -y nvcomp
else
    echo "⚠ Could not download nvCOMP 3.0.6 from NVIDIA"
    echo "Manual download required from:"
    echo "  https://developer.nvidia.com/nvcomp"
    exit 1
fi

echo ""
echo "[3/4] Verifying installation..."
find /usr -name "libnvcomp.so*" 2>/dev/null || echo "⚠ libnvcomp.so not found"
find /usr -name "nvcomp/zstd.h" 2>/dev/null | head -1 || echo "⚠ nvcomp headers not found"

echo ""
echo "[4/4] Rebuilding codec..."
cd /workspace/CodecLLM/core
rm -rf build
mkdir build
cd build
cmake .. \
    -DNVCOMP_INCLUDE_DIR=/usr/include/nvcomp_12 \
    -DNVCOMP_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcomp.so
make -j$(nproc)

echo ""
echo "================================"
echo "✓ Downgrade complete!"
echo "================================"
echo ""
echo "Check version:"
ldd build/libcodec_core.so | grep nvcomp
strings /usr/lib/x86_64-linux-gnu/libnvcomp.so* | grep -i "version\|nvcomp" | head -5

