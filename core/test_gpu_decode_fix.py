#!/usr/bin/env python3
"""
Quick test to verify GPU decode fix
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace/CodecLLM/core')

from bindings_zstd import ZstdEncoder, ZstdGPUDecoder

print("Testing GPU decode with device actual_sizes...")

# Create test data
np.random.seed(42)
rows, cols = 2048, 2048
weight_original = np.random.randn(rows, cols).astype(np.float16)

# Quantize
scales = np.abs(weight_original).max(axis=1, keepdims=True) / 127.0
scales = np.maximum(scales, 1e-8)
weight_int8 = np.clip(np.round(weight_original / scales), -127, 127).astype(np.int8)

# Compress
encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()

print("Compressing...")
compressed, ratio = encoder.encode_layer(weight_int8)
print(f"Compressed: {len(compressed)} bytes, ratio: {ratio:.2f}x")

# Test GPU decode
print("Testing GPU decode...")
try:
    gpu_ptr, rows_gpu, cols_gpu, dtype_gpu = decoder.decode_layer_to_gpu(compressed)
    print(f"GPU decode successful: {rows_gpu}x{cols_gpu}, ptr: 0x{gpu_ptr:x}")

    # Copy back to CPU to verify
    import ctypes
    cudart = ctypes.CDLL('libcudart.so')
    weight_from_gpu = np.empty((rows_gpu, cols_gpu), dtype=np.int8)
    cudart.cudaMemcpy(
        weight_from_gpu.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_void_p(gpu_ptr),
        ctypes.c_size_t(rows_gpu * cols_gpu),
        ctypes.c_int(2)  # cudaMemcpyDeviceToHost
    )

    # Check if we got actual data
    matches = np.array_equal(weight_int8, weight_from_gpu)
    print(f"GPU decode matches CPU: {matches}")
    print(f"GPU data range: [{weight_from_gpu.min()}, {weight_from_gpu.max()}]")

    if not matches:
        print("Differences found:")
        diff = np.sum(weight_int8 != weight_from_gpu)
        print(f"  Total differences: {diff:,} / {weight_int8.size:,} ({diff/weight_int8.size*100:.2f}%)")
        print(f"  Max diff: {np.abs(weight_int8.astype(np.int16) - weight_from_gpu.astype(np.int16)).max()}")

    cudart.cudaFree(ctypes.c_void_p(gpu_ptr))

except Exception as e:
    print(f"GPU decode failed: {e}")
    import traceback
    traceback.print_exc()

print("Test complete!")
