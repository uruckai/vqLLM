#!/bin/bash
# Script to extract nvCOMP API signatures from header files

echo "==================================="
echo "nvCOMP Zstd API Header Inspection"
echo "==================================="

HEADER="/usr/include/nvcomp_12/nvcomp/zstd.h"

if [ ! -f "$HEADER" ]; then
    echo "ERROR: Cannot find $HEADER"
    exit 1
fi

echo ""
echo "1. nvcompBatchedZstdCompressGetTempSizeSync signature:"
echo "======================================================="
grep -A 20 "nvcompBatchedZstdCompressGetTempSizeSync" "$HEADER" | head -30

echo ""
echo "2. nvcompBatchedZstdCompressAsync signature:"
echo "============================================="
grep -A 20 "nvcompBatchedZstdCompressAsync" "$HEADER" | head -30

echo ""
echo "3. nvcompBatchedZstdDecompressGetTempSizeAsync signature:"
echo "=========================================================="
grep -A 20 "nvcompBatchedZstdDecompressGetTempSizeAsync" "$HEADER" | head -30

echo ""
echo "4. nvcompBatchedZstdDecompressAsync signature:"
echo "==============================================="
grep -A 20 "nvcompBatchedZstdDecompressAsync" "$HEADER" | head -30

echo ""
echo "5. nvcompBatchedZstdCompressOpts_t structure:"
echo "=============================================="
grep -B 5 -A 10 "nvcompBatchedZstdCompressOpts_t" "$HEADER" | head -20

echo ""
echo "6. nvcompBatchedZstdDecompressOpts_t structure:"
echo "================================================"
grep -B 5 -A 10 "nvcompBatchedZstdDecompressOpts_t" "$HEADER" | head -20

