#!/bin/bash
# Install nvCOMP 3.0.6 (last known working version)

echo "==================================="
echo "Installing nvCOMP 3.0.6"
echo "==================================="

# Remove nvCOMP 5
echo "[1/5] Removing nvCOMP 5.0..."
sudo apt-get remove -y nvcomp 2>/dev/null || true
sudo rm -rf /var/nvcomp-local-repo-ubuntu2404-5.0.0.6 2>/dev/null || true

# Clean up previous installations
sudo rm -rf /workspace/nvcomp_install 2>/dev/null || true
sudo rm -rf /workspace/nvcomp 2>/dev/null || true

# Build nvCOMP 3.0.6 from source
echo "[2/5] Cloning nvCOMP repository..."
cd /workspace
git clone https://github.com/NVIDIA/nvcomp.git
cd nvcomp

echo "[3/5] Checking out v3.0.6..."
git checkout v3.0.6

echo "[4/5] Building nvCOMP 3.0.6..."
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_EXAMPLES=OFF

make -j$(nproc)

echo "[5/5] Installing nvCOMP 3.0.6..."
sudo make install

# Update library cache
sudo ldconfig

echo ""
echo "==================================="
echo "nvCOMP 3.0.6 Installation Complete"
echo "==================================="
echo ""
echo "Verifying installation..."
ls -la /usr/local/lib/libnvcomp* 2>/dev/null && echo "✓ Libraries installed" || echo "✗ Libraries not found"
ls -la /usr/local/include/nvcomp/ 2>/dev/null && echo "✓ Headers installed" || echo "✗ Headers not found"

echo ""
echo "Next steps:"
echo "1. cd /workspace/CodecLLM/core"
echo "2. rm -rf build && mkdir build && cd build"
echo "3. cmake .. -DNVCOMP_INCLUDE_DIR=/usr/local/include -DNVCOMP_LIBRARY=/usr/local/lib/libnvcomp.so"
echo "4. make -j\$(nproc)"
echo "5. bash TEST_NVCOMP5.sh  # Should work with nvCOMP 3.0.6"

