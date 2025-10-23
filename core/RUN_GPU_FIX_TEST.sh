#!/bin/bash

cd /workspace/CodecLLM/core

echo "Testing GPU decode fix..."
python test_gpu_decode_fix.py
