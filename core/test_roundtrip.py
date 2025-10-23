#!/usr/bin/env python3
"""
Test quantization + compression + decompression round-trip
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace/CodecLLM/core')

from bindings_zstd import ZstdEncoder, ZstdGPUDecoder

print("=" * 80)
print("ROUND-TRIP TEST: Quantize -> Compress -> Decompress -> Dequantize")
print("=" * 80)
print()

# Create test data
np.random.seed(42)
rows, cols = 2048, 2048
weight_original = np.random.randn(rows, cols).astype(np.float16)

print(f"Original weight:")
print(f"  Shape: {weight_original.shape}")
print(f"  Dtype: {weight_original.dtype}")
print(f"  Range: [{weight_original.min():.6f}, {weight_original.max():.6f}]")
print(f"  Mean: {weight_original.mean():.6f}")
print()

# Step 1: Per-channel quantization
print("-" * 80)
print("STEP 1: Per-channel quantization")
scales = np.abs(weight_original).max(axis=1, keepdims=True) / 127.0
scales = np.maximum(scales, 1e-8)
weight_int8 = np.clip(np.round(weight_original / scales), -127, 127).astype(np.int8)

print(f"  Scales shape: {scales.shape}")
print(f"  Scales range: [{scales.min():.8f}, {scales.max():.8f}]")
print(f"  INT8 range: [{weight_int8.min()}, {weight_int8.max()}]")
print(f"  INT8 unique values: {len(np.unique(weight_int8))}")
print()

# Step 2: Compress
print("-" * 80)
print("STEP 2: Compress with Zstd")
encoder = ZstdEncoder(compression_level=9)
compressed, ratio = encoder.encode_layer(weight_int8)

print(f"  Original size: {weight_int8.nbytes:,} bytes ({weight_int8.nbytes/1024**2:.2f} MB)")
print(f"  Compressed size: {len(compressed):,} bytes ({len(compressed)/1024**2:.2f} MB)")
print(f"  Ratio: {ratio:.2f}x")
print()

# Step 3: Decompress (CPU)
print("-" * 80)
print("STEP 3: Decompress (CPU)")
decoder = ZstdGPUDecoder()
weight_int8_decoded = decoder.decode_layer(compressed)

print(f"  Decoded shape: {weight_int8_decoded.shape}")
print(f"  Decoded dtype: {weight_int8_decoded.dtype}")
print(f"  Decoded range: [{weight_int8_decoded.min()}, {weight_int8_decoded.max()}]")
print()

# Step 4: Check bit-exactness
print("-" * 80)
print("STEP 4: Verify bit-exact reconstruction")
matches = np.array_equal(weight_int8, weight_int8_decoded)
if matches:
    print(f"  ✓ BIT-EXACT MATCH!")
else:
    print(f"  ✗ MISMATCH!")
    diff = np.sum(weight_int8 != weight_int8_decoded)
    print(f"  Differences: {diff:,} / {weight_int8.size:,} ({diff/weight_int8.size*100:.2f}%)")
    print(f"  Max diff: {np.abs(weight_int8.astype(np.int16) - weight_int8_decoded.astype(np.int16)).max()}")
print()

# Step 5: Dequantize
print("-" * 80)
print("STEP 5: Dequantize (per-channel)")
weight_dequant = weight_int8_decoded.astype(np.float32) * scales

print(f"  Dequantized shape: {weight_dequant.shape}")
print(f"  Dequantized range: [{weight_dequant.min():.6f}, {weight_dequant.max():.6f}]")
print()

# Step 6: Compare to original
print("-" * 80)
print("STEP 6: Compare to original")
error = np.abs(weight_original.astype(np.float32) - weight_dequant)
print(f"  Mean error: {error.mean():.8f}")
print(f"  Max error: {error.max():.8f}")
print(f"  Relative error: {(error.mean() / np.abs(weight_original).mean() * 100):.4f}%")
print()

# Step 7: Test GPU decompress
print("-" * 80)
print("STEP 7: GPU Decompress")
try:
    gpu_ptr, rows_gpu, cols_gpu, dtype_gpu = decoder.decode_layer_to_gpu(compressed)
    print(f"  ✓ GPU decode successful")
    print(f"  GPU ptr: 0x{gpu_ptr:x}")
    print(f"  Shape: ({rows_gpu}, {cols_gpu})")
    
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
    cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
    
    gpu_matches = np.array_equal(weight_int8, weight_from_gpu)
    if gpu_matches:
        print(f"  ✓ GPU decode matches CPU decode!")
    else:
        print(f"  ✗ GPU decode MISMATCH!")
        diff_gpu = np.sum(weight_int8 != weight_from_gpu)
        print(f"  Differences: {diff_gpu:,} / {weight_int8.size:,}")
        
except Exception as e:
    print(f"  ✗ GPU decode failed: {e}")
print()

# Summary
print("=" * 80)
print("SUMMARY:")
print("=" * 80)
if matches:
    print("✓ Compression/decompression: BIT-EXACT")
else:
    print("✗ Compression/decompression: CORRUPTED")

print(f"✓ Quantization error: {(error.mean() / np.abs(weight_original).mean() * 100):.4f}% (expected <1%)")
print(f"✓ Compression ratio: {ratio:.2f}x")
print()

if not matches:
    print("⚠️  CRITICAL: Compression is corrupting data!")
    print("This explains the garbage output in the LLM test.")
else:
    print("Compression pipeline is working correctly.")
    print("If LLM output is still garbage, issue is elsewhere.")

