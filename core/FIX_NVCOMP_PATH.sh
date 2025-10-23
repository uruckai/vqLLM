#!/bin/bash

echo "========================================="
echo "  FINDING nvCOMP LIBRARY"
echo "========================================="
echo ""

echo "Searching for libnvcomp.so..."
find /usr -name "libnvcomp.so*" 2>/dev/null
find /workspace -name "libnvcomp.so*" 2>/dev/null

echo ""
echo "Checking ldconfig cache..."
ldconfig -p | grep nvcomp

echo ""
echo "Checking common locations..."
ls -la /usr/lib/x86_64-linux-gnu/libnvcomp* 2>/dev/null || echo "Not in /usr/lib/x86_64-linux-gnu/"
ls -la /usr/local/lib/libnvcomp* 2>/dev/null || echo "Not in /usr/local/lib/"
ls -la /usr/lib/libnvcomp* 2>/dev/null || echo "Not in /usr/lib/"

echo ""
echo "========================================="
echo "If found, use that path in cmake command"
echo "========================================="

