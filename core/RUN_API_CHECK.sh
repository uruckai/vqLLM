#!/bin/bash
# Complete script to pull latest code and inspect nvCOMP API

set -e  # Exit on error

echo "==================================="
echo "nvCOMP API Inspection - Full Setup"
echo "==================================="

echo ""
echo "[1/3] Pulling latest code from GitHub..."
cd /workspace/CodecLLM
git fetch origin
git reset --hard origin/main
git pull

echo ""
echo "[2/3] Making script executable..."
cd core
chmod +x check_nvcomp_api.sh

echo ""
echo "[3/3] Inspecting nvCOMP API from headers..."
bash check_nvcomp_api.sh

echo ""
echo "==================================="
echo "✓ Complete!"
echo "==================================="

