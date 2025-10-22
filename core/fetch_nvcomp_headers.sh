#!/bin/bash
# Fetch nvCOMP 5.0 headers and commit them to repo for reference

echo "Fetching nvCOMP 5.0 API headers..."

cd /workspace/CodecLLM/core

# Create directory
mkdir -p nvcomp_api_reference

# Copy zstd header
echo "Copying zstd.h..."
cat /usr/include/nvcomp_12/nvcomp/zstd.h > nvcomp_api_reference/zstd.h 2>/dev/null || \
cat /usr/include/nvcomp_13/nvcomp/zstd.h > nvcomp_api_reference/zstd.h

# Copy main header
echo "Copying nvcomp.h..."
cat /usr/include/nvcomp_12/nvcomp.h > nvcomp_api_reference/nvcomp.h 2>/dev/null || \
cat /usr/include/nvcomp_13/nvcomp.h > nvcomp_api_reference/nvcomp.h

echo ""
echo "Headers saved to nvcomp_api_reference/"
echo ""
echo "Key Zstd function signatures:"
echo "=============================="
echo ""
grep -B2 -A15 "nvcompBatchedZstdCompressGetTempSize" nvcomp_api_reference/zstd.h | head -40
echo ""
echo "=============================="
echo ""
grep -B2 -A15 "nvcompBatchedZstdDecompressGetTempSize" nvcomp_api_reference/zstd.h | head -40

echo ""
echo "Complete! Now commit these files:"
echo "  git add nvcomp_api_reference/"
echo "  git commit -m 'Add nvCOMP 5.0 API reference headers'"
echo "  git push"

