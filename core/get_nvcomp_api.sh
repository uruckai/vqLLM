#!/bin/bash
# Download nvCOMP 5.0 header files for API reference

echo "Downloading nvCOMP 5.0 API headers from RunPod..."

mkdir -p nvcomp_headers

# Copy all nvCOMP headers
echo "Copying Zstd header..."
cp /usr/include/nvcomp_12/nvcomp/zstd.h nvcomp_headers/ 2>/dev/null || cp /usr/include/nvcomp_13/nvcomp/zstd.h nvcomp_headers/

echo "Copying main nvcomp header..."
cp /usr/include/nvcomp_12/nvcomp.h nvcomp_headers/ 2>/dev/null || cp /usr/include/nvcomp_13/nvcomp.h nvcomp_headers/

echo "Copying LZ4 header for reference..."
cp /usr/include/nvcomp_12/nvcomp/lz4.h nvcomp_headers/ 2>/dev/null || cp /usr/include/nvcomp_13/nvcomp/lz4.h nvcomp_headers/

echo "Copying common headers..."
cp /usr/include/nvcomp_12/nvcomp/*.h nvcomp_headers/ 2>/dev/null || cp /usr/include/nvcomp_13/nvcomp/*.h nvcomp_headers/ 2>/dev/null

echo ""
echo "Headers copied to nvcomp_headers/"
ls -lh nvcomp_headers/

echo ""
echo "Displaying Zstd compression API signatures:"
echo "==========================================="
grep -A 30 "nvcompBatchedZstdCompressGetTempSize" nvcomp_headers/zstd.h || echo "Not found"

echo ""
echo "Displaying Zstd decompression API signatures:"
echo "=============================================="
grep -A 30 "nvcompBatchedZstdDecompressGetTempSize" nvcomp_headers/zstd.h || echo "Not found"

echo ""
echo "Complete!"

