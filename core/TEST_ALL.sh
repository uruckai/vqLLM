#!/bin/bash
# One-command complete test runner
# Pulls code, checks environment, runs progressive test

set -e

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                  CodecLLM Complete Test Suite                         ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Get to the right directory
cd /workspace/CodecLLM
echo "[Step 1/4] Pulling latest code..."
git pull
cd core
echo "✓ Code updated"
echo ""

# Build if needed
if [ ! -f "build/libcodec_core.so" ]; then
    echo "[Step 2/4] Building codec library..."
    bash build.sh
    echo "✓ Build complete"
else
    echo "[Step 2/4] Codec library already built"
    echo "✓ Skipping build"
fi
echo ""

# Check environment
echo "[Step 3/4] Checking environment..."
python3 check_runpod_status.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Environment check failed!"
    echo "Please fix the issues above and try again."
    exit 1
fi
echo ""

# Run progressive test
echo "[Step 4/4] Running progressive compression test..."
echo "This will test 1, 5, 10, and 20 compressed layers."
echo "Expected runtime: ~10 minutes"
echo ""
python3 test_progressive_compression.py

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                         Test Complete!                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Review the FINAL SUMMARY table above"
echo "  2. Share results with your dev team"
echo "  3. If quality is good (✓ PERFECT), scale to more layers"
echo "  4. If quality is bad (✗ MAJOR DIFF), run: python3 test_quantization_debug.py"

