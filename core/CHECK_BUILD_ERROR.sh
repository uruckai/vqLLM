#!/bin/bash

echo "========================================="
echo "  CHECKING BUILD ERROR"
echo "========================================="
echo ""

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

cd /workspace/CodecLLM/core

echo "Cleaning old build..."
rm -rf build
mkdir -p build
cd build

echo ""
echo "Running CMake with verbose output..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCOMP_INCLUDE_DIR=/usr/local/include \
  -DNVCOMP_LIBRARY=/usr/local/lib/libnvcomp.so

echo ""
echo "Building with verbose output (showing actual error)..."
make VERBOSE=1 2>&1 | tee build.log

echo ""
echo "========================================="
echo "Build log saved to build/build.log"
echo "Showing last 50 lines of errors:"
echo "========================================="
tail -n 50 build.log

