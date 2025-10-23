#!/bin/bash
# Run LLM inference test with nvCOMP 3.0.6 Zstd compression

set -e

echo "========================================="
echo "LLM Inference Test with GPU Compression"
echo "========================================="
echo ""
echo "This will:"
echo "  - Load TinyLlama-1.1B (~2GB)"
echo "  - Compress 20 Linear layers with Zstd"
echo "  - Run inference with GPU-direct decode"
echo "  - Measure VRAM savings"
echo ""
echo "Starting test..."
echo ""

cd /workspace/CodecLLM/core

# Ensure library path is set
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Run the test
python test_zstd_inference.py

echo ""
echo "========================================="
echo "✓ Test complete!"
echo "========================================="

