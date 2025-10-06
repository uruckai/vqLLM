#!/bin/bash
# Run all test suites

set -e

echo "=============================================="
echo "Weight Codec - Full Test Suite"
echo "=============================================="
echo ""

# Check build first
echo "[1/3] Verifying build..."
bash scripts/check_build.sh
echo ""

# Run compression roundtrip tests
echo "[2/3] Running compression roundtrip tests..."
echo "=============================================="
python3 tests/test_compression_roundtrip.py
TEST_RESULT=$?
echo ""

# Run analysis tests
echo "[3/3] Running analysis tests..."
echo "=============================================="
python3 tests/test_week2_week3.py
echo ""

# Summary
echo "=============================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✓ ALL TEST SUITES PASSED"
    echo "=============================================="
    echo ""
    echo "Week 2+3 Implementation VERIFIED:"
    echo "  ✓ Predictive coding working"
    echo "  ✓ rANS entropy coding working"
    echo "  ✓ Encode/decode roundtrip bit-exact"
    echo "  ✓ Compression achieved (see ratios above)"
    echo "  ✓ Transform coding implemented"
    echo "  ✓ Bitplane operations implemented"
    echo "  ✓ Container format ready"
    echo ""
    echo "Next steps: Week 4 - GPU acceleration"
    exit 0
else
    echo "✗ SOME TESTS FAILED"
    exit 1
fi

