#!/bin/bash

echo "Checking Zstd installation..."
echo ""

echo "Looking for zstd.h:"
find /usr/include -name "zstd.h" 2>/dev/null

echo ""
echo "Looking for libzstd.so:"
find /usr/lib -name "libzstd.so*" 2>/dev/null

echo ""
echo "Checking pkg-config:"
pkg-config --cflags --libs libzstd 2>/dev/null || echo "pkg-config not available or libzstd.pc not found"

echo ""
echo "Done!"

