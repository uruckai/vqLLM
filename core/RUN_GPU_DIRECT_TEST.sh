#!/bin/bash

echo "========================================="
echo "  GPU-DIRECT DECODE TEST"
echo "========================================="

cd /workspace/CodecLLM
git pull

cd core
python test_zstd_inference.py

echo ""
echo "========================================="
echo "  Test complete!"
echo "========================================="
echo ""
echo "Expected results:"
echo "  - Time: ~50-100 seconds (2-5x faster than before!)"
echo "  - VRAM: ~2.08 GB (unchanged)"
echo "  - Output: Should see '[DECODER] ✓ GPU direct decode SUCCESS'"
echo ""
echo "If it works, this means:"
echo "  ✓ Eliminated CPU→GPU copy overhead"
echo "  ✓ Zero-copy tensor wrapping working"
echo "  ✓ GPU dequantization working"
echo ""

