#!/bin/bash
set -e  # Exit on any error

echo "========================================="
echo "Clean nvCOMP 3.0.6 Installation"
echo "========================================="

# Step 1: Complete removal of nvCOMP 5
echo ""
echo "[1/6] Removing all nvCOMP 5 packages..."
sudo apt-get remove -y nvcomp nvcomp-cuda-11 nvcomp-cuda-12 nvcomp-cuda-13 \
    libnvcomp5-cuda-11 libnvcomp5-cuda-12 libnvcomp5-cuda-13 \
    libnvcomp5-dev-cuda-11 libnvcomp5-dev-cuda-12 libnvcomp5-dev-cuda-13 \
    libnvcomp5-static-cuda-11 libnvcomp5-static-cuda-12 libnvcomp5-static-cuda-13 \
    nvcomp0 2>/dev/null || true

sudo apt-get autoremove -y

echo "Removing nvCOMP 5 repositories..."
sudo rm -rf /var/nvcomp-local-repo-ubuntu2404-5.0.0.6 2>/dev/null || true
sudo rm -f /etc/apt/sources.list.d/nvcomp*.list 2>/dev/null || true

echo "Cleaning up nvCOMP 5 files..."
sudo rm -rf /usr/lib/x86_64-linux-gnu/nvcomp 2>/dev/null || true
sudo rm -rf /usr/lib/x86_64-linux-gnu/libnvcomp* 2>/dev/null || true
sudo rm -rf /usr/include/nvcomp_* 2>/dev/null || true
sudo rm -rf /usr/share/doc/libnvcomp* 2>/dev/null || true

# Step 2: Clean up any previous nvCOMP 3 attempts
echo ""
echo "[2/6] Cleaning previous installation attempts..."
sudo rm -rf /workspace/nvcomp 2>/dev/null || true
sudo rm -rf /usr/local/lib/libnvcomp* 2>/dev/null || true
sudo rm -rf /usr/local/include/nvcomp* 2>/dev/null || true

# Step 3: Clone nvCOMP - we need the examples repo which has the build system
echo ""
echo "[3/6] Cloning nvCOMP repository..."
cd /workspace

# The NVIDIA/nvcomp repo IS the correct one, but v3.0.1 has a circular dependency
# We need to build it without the find_package check
git clone https://github.com/NVIDIA/nvcomp.git
cd nvcomp

# Step 4: Checkout v3.0.1 (latest 3.x version)
echo ""
echo "[4/6] Checking out v3.0.1..."
git fetch --all --tags
git checkout v3.0.1

echo "Verifying checkout..."
git log -1 --oneline

# Step 5: Patch CMakeLists.txt to remove circular dependency
echo ""
echo "[5/6] Patching build system..."
# Comment out the find_package(nvcomp) line that causes circular dependency
sed -i 's/^find_package(nvcomp/#find_package(nvcomp/' CMakeLists.txt

echo "Building nvCOMP 3.0.1..."
mkdir -p build
cd build

# Build as a standalone library
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"

echo ""
echo "Compiling (this may take a few minutes)..."
make -j$(nproc) 2>&1 | tee build.log || {
    echo "❌ Build failed! Check build.log for details"
    exit 1
}

# Step 6: Install
echo ""
echo "[6/6] Installing nvCOMP 3.0.1..."
sudo make install

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
ls -lh /usr/local/include/nvcomp* 2>/dev/null || echo "  ❌ No headers found!"
ls -lh /usr/local/include/nvcomp/*.h 2>/dev/null | head -5 || true

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
echo "6. bash test_nvcomp_versions.sh  # Test if v3.0.1 works"
echo ""

