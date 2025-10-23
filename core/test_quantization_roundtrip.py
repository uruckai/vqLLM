#!/usr/bin/env python3
"""
Quick test to verify quantization round-trip without LLM inference
Isolates whether the problem is quantization or model interaction
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder

print("="*80)
print("QUANTIZATION ROUND-TRIP TEST")
print("="*80)
print()
print("Testing if quantization/dequantization is working correctly")
print("This isolates the codec from the LLM model")
print()

# Create test data that mimics LLM weights
np.random.seed(42)
rows, cols = 2048, 2048
weight_original = np.random.randn(rows, cols).astype(np.float16)

print(f"Original weight: shape={weight_original.shape}, dtype={weight_original.dtype}")
print(f"  Range: [{weight_original.min():.6f}, {weight_original.max():.6f}]")
print(f"  Mean: {weight_original.mean():.6f}, Std: {weight_original.std():.6f}")
print()

# Per-channel quantization (what we use in inference)
print("Step 1: Per-channel quantization...")
scales = np.abs(weight_original).max(axis=1, keepdims=True) / 127.0
scales = np.maximum(scales, 1e-8)
scales = scales.astype(np.float32)  # CRITICAL: float32 for precision

print(f"  Scales: shape={scales.shape}, dtype={scales.dtype}")
print(f"  Scale range: [{scales.min():.6f}, {scales.max():.6f}]")
print(f"  First 5 scales: {scales[:5, 0]}")
print()

# Quantize
weight_int8 = np.clip(np.round(weight_original / scales), -127, 127).astype(np.int8)
print(f"Step 2: Quantized to INT8...")
print(f"  INT8 range: [{weight_int8.min()}, {weight_int8.max()}]")
print(f"  Unique values: {len(np.unique(weight_int8))}")
print()

# Compress
print("Step 3: Compress with Zstd...")
encoder = ZstdEncoder(compression_level=9)
compressed, ratio = encoder.encode_layer(weight_int8)
print(f"  Compressed size: {len(compressed)} bytes")
print(f"  Compression ratio: {ratio:.2f}x")
print()

# Decompress to CPU
print("Step 4: Decompress to CPU...")
decoder = ZstdGPUDecoder()
weight_int8_decoded = decoder.decode_layer(compressed)
print(f"  Decoded shape: {weight_int8_decoded.shape}")
print(f"  Decoded range: [{weight_int8_decoded.min()}, {weight_int8_decoded.max()}]")
print()

# Check bit-exact
if np.array_equal(weight_int8, weight_int8_decoded):
    print("✓ Bit-exact INT8 reconstruction (CPU decode)")
else:
    errors = np.sum(weight_int8 != weight_int8_decoded)
    print(f"✗ INT8 mismatch: {errors} errors (CPU decode)")
    sys.exit(1)

print()

# Decompress to GPU (what we use in inference)
print("Step 5: Decompress to GPU (inference path)...")
gpu_ptr, rows_gpu, cols_gpu, dtype_gpu = decoder.decode_layer_to_gpu(compressed)
print(f"  GPU pointer: {hex(gpu_ptr)}")
print(f"  Dimensions: {rows_gpu}x{cols_gpu}")
print()

# Copy back to CPU to verify
import ctypes
cudart = ctypes.CDLL('libcudart.so')
weight_int8_from_gpu = np.empty((rows_gpu, cols_gpu), dtype=np.int8)
cudart.cudaMemcpy(
    weight_int8_from_gpu.ctypes.data_as(ctypes.c_void_p),
    ctypes.c_void_p(gpu_ptr),
    ctypes.c_size_t(rows_gpu * cols_gpu),
    ctypes.c_int(2)  # cudaMemcpyDeviceToHost
)
cudart.cudaFree(ctypes.c_void_p(gpu_ptr))

print(f"  GPU decode range: [{weight_int8_from_gpu.min()}, {weight_int8_from_gpu.max()}]")

if np.array_equal(weight_int8, weight_int8_from_gpu):
    print("✓ Bit-exact INT8 reconstruction (GPU decode)")
else:
    errors = np.sum(weight_int8 != weight_int8_from_gpu)
    print(f"✗ INT8 mismatch: {errors} errors (GPU decode)")
    sys.exit(1)

print()

# Dequantize (simulating inference)
print("Step 6: Dequantize (simulate inference)...")
print("  Method 1: NumPy (reference)")
weight_fp_numpy = weight_int8_decoded.astype(np.float32) * scales
error_numpy = np.abs(weight_original.astype(np.float32) - weight_fp_numpy)
print(f"    Reconstruction error: mean={error_numpy.mean():.6f}, max={error_numpy.max():.6f}")

print()
print("  Method 2: PyTorch (inference path)")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_int8_torch = torch.from_numpy(weight_int8_from_gpu).to(device)
scales_torch = torch.from_numpy(scales.squeeze()).to(torch.float16).to(device)

# Convert INT8 to FP16
weight_fp_unscaled = weight_int8_torch.to(torch.float16)
print(f"    INT8→FP16: range=[{weight_fp_unscaled.min():.1f}, {weight_fp_unscaled.max():.1f}]")

# Broadcast scales
scale_expanded = scales_torch.view(-1, 1)
print(f"    Scales shape: {scale_expanded.shape}")

# Dequantize
weight_fp_torch = weight_fp_unscaled * scale_expanded
print(f"    Dequantized: range=[{weight_fp_torch.min():.6f}, {weight_fp_torch.max():.6f}]")

# Check error
weight_original_torch = torch.from_numpy(weight_original).to(torch.float16).to(device)
error_torch = torch.abs(weight_original_torch - weight_fp_torch.to(torch.float16))
print(f"    Reconstruction error: mean={error_torch.mean():.6f}, max={error_torch.max():.6f}")

print()
print("="*80)
print("RESULTS")
print("="*80)
print()

if error_torch.mean() < 0.01:
    print("✓ QUANTIZATION WORKING CORRECTLY")
    print()
    print("The quantization/dequantization pipeline is working.")
    print("The problem must be in the hybrid model (some layers compressed, some not).")
    print()
    print("RECOMMENDATION: Run test_all_layers_compressed.py")
    print("Compressing ALL layers should eliminate error amplification.")
else:
    print("✗ QUANTIZATION HAS ISSUES")
    print()
    print(f"Mean reconstruction error: {error_torch.mean():.6f} (should be < 0.01)")
    print(f"Max reconstruction error: {error_torch.max():.6f}")
    print()
    print("The quantization itself may need tuning:")
    print("  - Check if scales are reasonable")
    print("  - Verify INT8 range utilization")
    print("  - Consider asymmetric quantization")

print()

