#!/bin/bash
# Quick test script for nvCOMP 5.0 on RunPod

echo "================================"
echo "nvCOMP 5.0 Test Script"
echo "================================"

cd /workspace/CodecLLM

# Pull latest code
echo ""
echo "[1/5] Pulling latest code..."
git pull

# Rebuild
echo ""
echo "[2/5] Rebuilding codec..."
cd core
rm -rf build
mkdir build
cd build

cmake .. \
  -DNVCOMP_INCLUDE_DIR=/usr/include/nvcomp_12 \
  -DNVCOMP_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcomp.so

make -j$(nproc)

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "✓ Build successful!"

# Verify nvCOMP linkage
echo ""
echo "[3/5] Verifying nvCOMP linkage..."
ldd libcodec_core.so | grep nvcomp

# Quick encoder test
echo ""
echo "[4/5] Testing Zstd encoder..."
cd ..
python3 -c "
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder
import numpy as np

print('Creating encoder...')
enc = ZstdEncoder(compression_level=3)

print('Creating test data (256x256)...')
data = np.random.randint(-128, 127, size=(256, 256), dtype=np.int8)

print('Compressing...')
compressed, ratio = enc.encode_layer(data)
print(f'  Original: {data.size} bytes')
print(f'  Compressed: {len(compressed)} bytes')
print(f'  Ratio: {ratio:.2f}x')

print('Creating GPU decoder...')
dec = ZstdGPUDecoder()
print(f'  GPU available: {dec.is_available()}')

print('')
print('✓ Encoder test passed!')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Encoder test failed!"
    exit 1
fi

# Full GPU decode test
echo ""
echo "[5/5] Testing GPU decode..."
python3 test_gpu_direct_simple.py

echo ""
echo "================================"
echo "Test complete!"
echo "================================"

