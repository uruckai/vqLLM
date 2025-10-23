#!/bin/bash
# Run tests with filtered output (removes repetitive C++ debug messages)

set -e

echo "=================================="
echo "QUIET TEST RUNNER"
echo "=================================="
echo ""

cd /workspace/CodecLLM
git pull
cd core

# Method 1: Use quiet Python test (less verbose on Python side)
echo "Method 1: Running quiet test (Python suppression)..."
echo "Note: C++ encoder/decoder messages will still appear"
echo ""
python3 test_all_layers_quiet.py 2>&1 | grep -v "^\[ENCODER\] Temp:" | grep -v "^\[ENCODER\] Starting" | head -500

echo ""
echo "=================================="
echo "Method 2: Full filtering..."
echo "=================================="
echo ""

# Method 2: Run with log filtering (removes repetitive lines)
python3 test_all_layers_quiet.py 2>&1 | python3 filter_test_logs.py | head -500

echo ""
echo "=================================="
echo "COMPLETE!"
echo "=================================="

