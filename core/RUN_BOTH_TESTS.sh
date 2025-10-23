#!/bin/bash

echo "=== TEST 1: GPU Decode Fix Test ==="
cd /workspace/CodecLLM/core
python test_gpu_decode_fix.py

echo ""
echo "=== TEST 2: LLM Test with 1 Layer ==="
cd /workspace/CodecLLM && git pull
cd /workspace/CodecLLM/core
python test_zstd_inference.py
