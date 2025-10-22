#!/bin/bash
set -e

echo "================================"
echo "RunPod Setup for CodecLLM"
echo "================================"
echo ""

# 1. Install system dependencies
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq cmake build-essential git wget libzstd-dev

# 2. Install nvCOMP 3.0 (has stable Zstd support)
echo "[2/6] Installing nvCOMP 3.0..."
cd /workspace
wget -q https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp-local-repo-ubuntu2204-3.0.6_3.0.6-1_amd64.deb
dpkg -i nvcomp-local-repo-ubuntu2204-3.0.6_3.0.6-1_amd64.deb
cp /var/nvcomp-local-repo-ubuntu2204-3.0.6/nvcomp-*-keyring.gpg /usr/share/keyrings/
apt-get update -qq
apt-get install -y -qq nvcomp
rm nvcomp-local-repo-ubuntu2204-3.0.6_3.0.6-1_amd64.deb

# 3. Install Python dependencies
echo "[3/6] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q torch transformers accelerate zstandard

# 4. Clone/update repo
echo "[4/6] Cloning CodecLLM repo..."
cd /workspace
if [ -d "CodecLLM" ]; then
    cd CodecLLM
    git pull
else
    git clone https://github.com/uruckai/vqLLM.git CodecLLM
    cd CodecLLM
fi

# 5. Build C++ codec library
echo "[5/6] Building codec library..."
cd core
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 6. Verify installation
echo "[6/6] Verifying installation..."
echo ""
if ldd libcodec_core.so | grep -q nvcomp; then
    echo "✓ nvCOMP GPU decode: ENABLED"
else
    echo "✗ nvCOMP GPU decode: NOT FOUND"
    exit 1
fi

cd /workspace/CodecLLM/core
python3 -c "from bindings_zstd import ZstdGPUDecoder; print('✓ Python bindings: OK')"
python3 -c "from bindings_zstd import ZstdGPUDecoder; print(f'✓ GPU decoder available: {ZstdGPUDecoder.is_available()}')"

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "================================"
echo ""
echo "Test with:"
echo "  cd /workspace/CodecLLM/core"
echo "  python test_gpu_direct_simple.py"
echo ""

