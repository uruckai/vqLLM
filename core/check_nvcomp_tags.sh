#!/bin/bash
echo "=== Checking available nvCOMP tags ==="
cd /workspace/nvcomp 2>/dev/null || { echo "nvcomp not cloned yet"; exit 1; }
echo ""
echo "All tags:"
git tag | sort -V | tail -20
echo ""
echo "Tags matching 3.0:"
git tag | grep "^v3\.0" | sort -V
echo ""
echo "Tags matching 2.x:"
git tag | grep "^v2\." | sort -V | tail -10
echo ""
echo "Latest 10 tags:"
git tag | sort -V | tail -10

