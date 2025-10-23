#!/bin/bash
echo "=== nvCOMP 5.0 API Inspection ==="
echo ""
echo "Looking for GetTempSizeSync signature..."
grep -A 30 "nvcompBatchedZstdCompressGetTempSizeSync" /usr/include/nvcomp_12/nvcomp/zstd.h
echo ""
echo "=== Looking for opts struct ==="
grep -B 5 -A 10 "nvcompBatchedZstdCompressOpts_t" /usr/include/nvcomp_12/nvcomp/zstd.h
echo ""
echo "=== Looking for DefaultOpts ==="
grep "Default" /usr/include/nvcomp_12/nvcomp/zstd.h
echo ""
echo "=== Looking for required alignments ==="
grep -A 5 "nvcompBatchedZstdCompressGetRequiredAlignments" /usr/include/nvcomp_12/nvcomp/zstd.h
grep -A 5 "nvcompBatchedZstdDecompressGetRequiredAlignments" /usr/include/nvcomp_12/nvcomp/zstd.h
