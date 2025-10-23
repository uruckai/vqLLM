#!/bin/bash
# Run basic compress/decompress test with nvCOMP 3.0.6

set -e

echo "========================================="
echo "Running Basic GPU Decode Test"
echo "========================================="
echo ""

cd /workspace/CodecLLM/core

# Ensure library path is set
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Run the test
python test_gpu_direct_simple.py

echo ""
echo "========================================="
echo "Test complete!"
echo "========================================="

