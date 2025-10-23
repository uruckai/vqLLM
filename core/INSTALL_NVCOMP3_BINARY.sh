#!/bin/bash
set -e  # Exit on any error

echo "========================================="
echo "Installing nvCOMP 3.0.6 (Binary Package)"
echo "========================================="

# Step 1: Remove nvCOMP 5
echo ""
echo "[1/5] Removing nvCOMP 5..."
sudo apt-get remove -y nvcomp nvcomp-cuda-11 nvcomp-cuda-12 nvcomp-cuda-13 \
    libnvcomp5-cuda-11 libnvcomp5-cuda-12 libnvcomp5-cuda-13 \
    libnvcomp5-dev-cuda-11 libnvcomp5-dev-cuda-12 libnvcomp5-dev-cuda-13 \
    libnvcomp5-static-cuda-11 libnvcomp5-static-cuda-12 libnvcomp5-static-cuda-13 \
    nvcomp0 2>/dev/null || true

sudo apt-get autoremove -y

echo "Cleaning up nvCOMP 5 files..."
sudo rm -rf /usr/lib/x86_64-linux-gnu/nvcomp 2>/dev/null || true
sudo rm -rf /usr/lib/x86_64-linux-gnu/libnvcomp* 2>/dev/null || true
sudo rm -rf /usr/include/nvcomp_* 2>/dev/null || true

# Step 2: Clean up any previous installations
echo ""
echo "[2/5] Cleaning previous installations..."
sudo rm -rf /workspace/nvcomp_install 2>/dev/null || true
sudo rm -rf /workspace/nvcomp 2>/dev/null || true
sudo rm -rf /usr/local/lib/libnvcomp* 2>/dev/null || true
sudo rm -rf /usr/local/include/nvcomp* 2>/dev/null || true
rm -f nvcomp_3.0.6_x86_64_12.x.tgz 2>/dev/null || true

# Step 3: Download nvCOMP 3.0.6 binary
echo ""
echo "[3/5] Downloading nvCOMP 3.0.6 binary..."
cd /workspace
wget https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz

# Step 4: Extract
echo ""
echo "[4/5] Extracting..."
# Extract to a temporary directory to avoid polluting /workspace
mkdir -p nvcomp_temp
cd nvcomp_temp
tar -xzf ../nvcomp_3.0.6_x86_64_12.x.tgz

echo "Contents extracted:"
ls -la

# Step 5: Install
echo ""
echo "[5/5] Installing to /usr/local..."

# The tarball extracts directly (no subdirectory), so lib and include are right here
if [ ! -d "lib" ] || [ ! -d "include" ]; then
    echo "❌ Could not find lib or include directories!"
    ls -la
    exit 1
fi

echo "Installing libraries and headers..."
# Copy libraries
sudo cp -r lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/

# Update library cache
echo "Updating ldconfig..."
sudo ldconfig

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""

# Verification
echo "Verifying installation..."
echo ""
echo "Libraries installed:"
ls -lh /usr/local/lib/libnvcomp* 2>/dev/null || echo "  ❌ No libraries found!"

echo ""
echo "Headers installed:"
ls -lh /usr/local/include/nvcomp.h 2>/dev/null || echo "  ❌ No main header found!"
ls -d /usr/local/include/nvcomp 2>/dev/null && echo "  ✓ nvcomp directory found" || echo "  ❌ nvcomp directory not found!"

echo ""
echo "ldconfig check:"
ldconfig -p | grep nvcomp || echo "  ❌ Not in ldconfig cache!"

echo ""
echo "========================================="
echo "Next Steps:"
echo "========================================="
echo "1. cd /workspace/CodecLLM/core"
echo "2. rm -rf build && mkdir build && cd build"
echo "3. cmake .."
echo "4. make -j\$(nproc)"
echo "5. cd .."
echo "6. bash test_nvcomp_versions.sh  # Test if v3.0.6 works!"
echo ""

