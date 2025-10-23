#!/bin/bash
# Rebuild and test nvCOMP 3.0.6 integration

set -e

echo "========================================="
echo "Pulling latest changes..."
echo "========================================="
cd /workspace/CodecLLM
git pull

echo ""
echo "========================================="
echo "Rebuilding codec..."
echo "========================================="
cd core
bash REBUILD_NVCOMP3.sh

echo ""
echo "========================================="
echo "Running basic test..."
echo "========================================="
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
python test_gpu_direct_simple.py

echo ""
echo "========================================="
echo "✓ All done!"
echo "========================================="

