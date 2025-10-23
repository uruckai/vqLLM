#!/bin/bash
# Progressive compression test runner for RunPod

set -e

echo "=================================="
echo "PROGRESSIVE COMPRESSION TEST"
echo "=================================="
echo ""

# Pull latest code
echo "[1/3] Pulling latest code..."
cd /workspace/CodecLLM
git pull
cd core

# Build if needed
if [ ! -f "build/libcodec_core.so" ]; then
    echo "[2/3] Building codec..."
    bash build.sh
else
    echo "[2/3] Codec already built, skipping..."
fi

# Run progressive test
echo "[3/3] Running progressive compression test..."
echo ""
python3 test_progressive_compression.py

echo ""
echo "=================================="
echo "TEST COMPLETE!"
echo "=================================="

