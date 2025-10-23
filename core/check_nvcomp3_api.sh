#!/bin/bash
echo "=== nvCOMP 3.0.6 API Inspection ==="
echo ""
echo "Looking for Zstd header..."
find /usr/local/include -name "*.h" -path "*nvcomp*" 2>/dev/null
echo ""
echo "=== Checking for Zstd API in nvCOMP 3.0.6 ==="
if [ -f "/usr/local/include/nvcomp/zstd.h" ]; then
    echo "✓ zstd.h found"
    echo ""
    echo "=== Compression functions ==="
    grep -A 5 "Compress" /usr/local/include/nvcomp/zstd.h | head -30
    echo ""
    echo "=== Decompression functions ==="
    grep -A 5 "Decompress" /usr/local/include/nvcomp/zstd.h | head -30
else
    echo "❌ zstd.h not found!"
    echo ""
    echo "Available headers:"
    ls -la /usr/local/include/nvcomp/
    echo ""
    echo "Checking main nvcomp.h:"
    grep -i "zstd" /usr/local/include/nvcomp.h || echo "No Zstd references in main header"
fi

