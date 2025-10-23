#!/bin/bash

# Quick test script for per-forward-pass caching
# Run this on RunPod after rebuilding

set -e

echo "========================================"
echo "Testing Per-Forward-Pass Caching"
echo "========================================"
echo ""

cd /workspace/CodecLLM/core

echo "Running test..."
echo ""

python test_zstd_inference.py

echo ""
echo "========================================"
echo "✓ Test complete!"
echo "========================================"
echo ""
echo "Expected results:"
echo "  - Forward passes: ~11 (for 10 tokens + prompt)"
echo "  - Decompressions: ~220 (20 layers × 11 passes)"
echo "  - Peak VRAM: 2.5-3 GB (similar to baseline 2.06 GB)"
echo "  - Speed: 20-50x slower than baseline (vs 478x before)"
echo ""

