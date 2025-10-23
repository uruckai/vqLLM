#!/bin/bash
echo "=== Checking nvCOMP Installation ==="
echo ""
echo "Looking for nvCOMP libraries..."
find /usr -name "libnvcomp*" 2>/dev/null
echo ""
echo "Looking for nvCOMP headers..."
find /usr -name "nvcomp.h" 2>/dev/null
find /usr -name "zstd.h" -path "*/nvcomp/*" 2>/dev/null
echo ""
echo "Checking /workspace..."
find /workspace -name "libnvcomp*" 2>/dev/null
find /workspace -name "nvcomp.h" 2>/dev/null
echo ""
echo "ldconfig cache:"
ldconfig -p | grep nvcomp

