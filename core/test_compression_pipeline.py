#!/usr/bin/env python3
"""
Debug: Test the compression pipeline in isolation
Find exactly where the 255x error is coming from
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder

print("="*80)
print("DEBUG: COMPRESSION PIPELINE ISOLATION")
print("="*80)
print()

# Test 1: Simple weight matrix compression
print("=== TEST 1: Simple matrix compression ===")

# Create a simple test matrix
original_weight = torch.randn(128, 256, dtype=torch.float16).numpy()
print(f"Original weight shape: {original_weight.shape}")
print(f"Original weight dtype: {original_weight.dtype}")
print(f"Original range: [{original_weight.min()".6f"}, {original_weight.max()".6f"}]")
print(f"Original mean: {original_weight.mean()".6f"}")
print()

# Step 1: Per-channel quantization (as in MLP code)
print("Step 1: Per-channel quantization...")
scales = np.abs(original_weight).max(axis=1, keepdims=True) / 127.0
scales = np.maximum(scales, 1e-8).astype(np.float32)
print(f"  Scales shape: {scales.shape}")
print(f"  Scale range: [{scales.min()".6f"}, {scales.max()".6f"}]")
print(f"  First 5 scales: {scales[:5].flatten()}")
print()

# Step 2: Quantize to INT8
print("Step 2: Quantize to INT8...")
weight_int8 = np.clip(np.round(original_weight / scales), -127, 127).astype(np.int8)
print(f"  INT8 range: [{weight_int8.min()}, {weight_int8.max()}]")
print(f"  Unique values: {len(np.unique(weight_int8))}")
print()

# Step 3: Compress with Zstd
print("Step 3: Compress with Zstd...")
encoder = ZstdEncoder(compression_level=9)
compressed, ratio = encoder.encode_layer(weight_int8)
print(f"  Compressed size: {len(compressed)} bytes")
print(f"  Compression ratio: {ratio".2f"}x")
print()

# Step 4: Decompress to CPU
print("Step 4: Decompress to CPU...")
decoder = ZstdGPUDecoder()
decoded_cpu = decoder.decode_layer(compressed)
print(f"  Decoded shape: {decoded_cpu.shape}")
print(f"  Decoded range: [{decoded_cpu.min()}, {decoded_cpu.max()}]")
print()

# Check CPU reconstruction
print("Step 5: CPU dequantization...")
reconstructed_cpu = decoded_cpu * scales
print(f"  Reconstructed range: [{reconstructed_cpu.min()".6f"}, {reconstructed_cpu.max()".6f"}]")
print(f"  Reconstructed mean: {reconstructed_cpu.mean()".6f"}")
print()

# Check accuracy
cpu_diff = np.abs(original_weight - reconstructed_cpu)
cpu_max_diff = cpu_diff.max()
cpu_rel_diff = (cpu_diff / (np.abs(original_weight) + 1e-8)).max()

print(f"  Max absolute difference: {cpu_max_diff".8f"}")
print(f"  Max relative difference: {cpu_rel_diff".2f"}")

if np.allclose(original_weight, reconstructed_cpu, rtol=1e-3, atol=1e-3):
    print("  ✓ CPU: Reconstruction is accurate")
else:
    print("  ✗ CPU: Reconstruction has errors")
print()

# Step 6: Decompress to GPU (our problematic path)
print("Step 6: GPU decompression (inference path)...")
import ctypes

gpu_ptr, rows, cols, actual_size = decoder.decode_layer_to_gpu(compressed)
print(f"  GPU pointer: 0x{gpu_ptr:x}")
print(f"  Dimensions: {rows}x{cols}")
print(f"  Actual size: {actual_size}")
print()

# Copy from GPU
weight_bytes_gpu = torch.empty((rows, cols), dtype=torch.int8, device='cuda')
cudart = ctypes.CDLL('libcudart.so')
result = cudart.cudaMemcpy(
    ctypes.c_void_p(weight_bytes_gpu.data_ptr()),
    ctypes.c_void_p(gpu_ptr),
    ctypes.c_size_t(rows * cols),
    ctypes.c_int(1)
)
cudart.cudaFree(ctypes.c_void_p(gpu_ptr))

print(f"  GPU copy result: {result}")
print(f"  GPU decoded range: [{weight_bytes_gpu.min()}, {weight_bytes_gpu.max()}]")
print()

# Step 7: Dequantize on GPU
print("Step 7: GPU dequantization...")
weight_fp_gpu = weight_bytes_gpu.to(torch.float16) * torch.from_numpy(scales).to(torch.float16).to('cuda')
print(f"  GPU dequantized range: [{weight_fp_gpu.min()".6f"}, {weight_fp_gpu.max()".6f"}]")
print(f"  GPU dequantized mean: {weight_fp_gpu.mean()".6f"}")
print()

# Compare with original
weight_fp_gpu_np = weight_fp_gpu.cpu().numpy()
gpu_diff = np.abs(original_weight - weight_fp_gpu_np)
gpu_max_diff = gpu_diff.max()
gpu_rel_diff = (gpu_diff / (np.abs(original_weight) + 1e-8)).max()

print(f"  Max absolute difference: {gpu_max_diff".8f"}")
print(f"  Max relative difference: {gpu_rel_diff".2f"}")

if np.allclose(original_weight, weight_fp_gpu_np, rtol=1e-3, atol=1e-3):
    print("  ✓ GPU: Reconstruction is accurate")
else:
    print("  ✗ GPU: Reconstruction has errors")
print()

# Compare CPU vs GPU paths
if np.allclose(reconstructed_cpu, weight_fp_gpu_np, rtol=1e-3, atol=1e-3):
    print("  ✓ CPU and GPU paths match")
else:
    print("  ✗ CPU and GPU paths differ")

cpu_gpu_diff = np.abs(reconstructed_cpu - weight_fp_gpu_np).max()
print(f"  Max CPU-GPU difference: {cpu_gpu_diff".8f"}")
print()

print("="*80)
print("VERDICT")
print("="*80)
print()

if gpu_rel_diff < 1.0:
    print("✓ COMPRESSION PIPELINE IS WORKING")
    print("The issue must be in the MLP integration or layer replacement.")
    print()
    print("Possible issues:")
    print("  1. Bias terms not handled correctly")
    print("  2. Layer replacement affecting forward pass")
    print("  3. Dtype mismatches in the model")
else:
    print("✗ COMPRESSION PIPELINE HAS ERRORS")
    print("The GPU decompression path is broken.")
    print()
    print("Most likely issues:")
    print("  1. GPU memory copy is wrong")
    print("  2. Scale application is wrong")
    print("  3. Tensor dtypes are mismatched")
    print("  4. GPU decode buffer size is wrong")

print()

