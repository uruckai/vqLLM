#!/bin/bash
# Restore the WORKING rANS implementation from backup

set -e

BACKUP="c:/Users/cfisc/OneDrive/Documents/llm project backups/CodecLLM/core"
DEST="core_rans"

echo "Restoring working rANS implementation..."
echo "  From: $BACKUP"
echo "  To:   $DEST/"
echo

# Create directory
mkdir -p "$DEST"

# Copy core files (simple rANS implementation)
echo "Copying source files..."
cp "$BACKUP/encoder.h" "$DEST/"
cp "$BACKUP/encoder_simple.cpp" "$DEST/"
cp "$BACKUP/decoder_host.h" "$DEST/"
cp "$BACKUP/decoder_host.cpp" "$DEST/"
cp "$BACKUP/rans.h" "$DEST/"
cp "$BACKUP/rans.cpp" "$DEST/"
cp "$BACKUP/format.h" "$DEST/"
cp "$BACKUP/c_api.cpp" "$DEST/"
cp "$BACKUP/decoder_gpu.cu" "$DEST/"
cp "$BACKUP/CMakeLists.txt" "$DEST/"

echo "✓ Copied 10 files"
echo
echo "Build instructions:"
echo "  cd core_rans"
echo "  mkdir -p build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  make -j\$(nproc)"
echo
echo "This will create: core_rans/build/libcodec_core.so"
echo
echo "Then run: python3 test_rans_simple_core.py"

