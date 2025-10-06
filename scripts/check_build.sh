#!/bin/bash
# Quick build verification script

set -e

echo "=============================================="
echo "Checking C++ Library Build"
echo "=============================================="

# Check if library exists
if [ -f "build/libwcodec.so" ]; then
    echo "✓ Library found: build/libwcodec.so"
    ls -lh build/libwcodec.so
    echo ""
    
    # Check symbols
    echo "Key symbols:"
    nm -D build/libwcodec.so | grep -E "(wcodec_encoder|wcodec_decoder|wcodec_encode_layer|wcodec_decode_layer)" | head -10
    echo ""
    
    # Try to load it in Python
    echo "Testing Python import..."
    python3 -c "from python.wcodec.bindings import Encoder, Decoder; print('✓ Python bindings load successfully!')"
    echo ""
    
    echo "=============================================="
    echo "✓ Build verification PASSED"
    echo "=============================================="
    exit 0
else
    echo "✗ Library not found: build/libwcodec.so"
    echo ""
    echo "Please build first:"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make -j8"
    exit 1
fi

