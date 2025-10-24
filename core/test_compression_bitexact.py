#!/usr/bin/env python3
"""
Test: Is FP16 compression bit-exact?
Critical test to see if Zstd compression introduces ANY differences
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
print("TEST: BIT-EXACT COMPRESSION VERIFICATION")
print("="*80)
print()

# Create test weight matrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_original = torch.randn(2048, 2048, dtype=torch.float16).numpy()

print("Step 1: Original FP16 weight")
print(f"  Shape: {weight_original.shape}")
print(f"  Dtype: {weight_original.dtype}")
print(f"  First 5 values: {weight_original.flatten()[:5]}")
print()

# Convert to bytes (as int8 view)
print("Step 2: Convert to byte representation")
weight_bytes = weight_original.tobytes()
weight_int8_view = np.frombuffer(weight_bytes, dtype=np.int8).reshape(
    weight_original.shape[0], weight_original.shape[1] * 2
)
print(f"  Byte shape: {weight_int8_view.shape}")
print(f"  Total bytes: {len(weight_bytes)}")
print()

# Compress
print("Step 3: Compress with Zstd")
encoder = ZstdEncoder(compression_level=9)
compressed, ratio = encoder.encode_layer(weight_int8_view)
print(f"  Compressed size: {len(compressed)} bytes")
print(f"  Ratio: {ratio:.2f}x")
print()

# Decompress to CPU
print("Step 4: Decompress to CPU")
decoder = ZstdGPUDecoder()
decoded_cpu = decoder.decode_layer(compressed)
print(f"  Decoded shape: {decoded_cpu.shape}")
print()

# Reconstruct FP16
weight_bytes_decoded = decoded_cpu.tobytes()
weight_reconstructed_cpu = np.frombuffer(weight_bytes_decoded, dtype=np.float16).reshape(weight_original.shape)

print("Step 5: Check CPU reconstruction")
print(f"  Reconstructed shape: {weight_reconstructed_cpu.shape}")
print(f"  First 5 values: {weight_reconstructed_cpu.flatten()[:5]}")
print()

if np.array_equal(weight_original, weight_reconstructed_cpu):
    print("✓ CPU: Bit-exact match!")
else:
    print("✗ CPU: NOT bit-exact")
    diff = np.abs(weight_original - weight_reconstructed_cpu)
    print(f"  Max difference: {diff.max()}")
    print(f"  Mean difference: {diff.mean()}")
    print(f"  Different elements: {(diff > 0).sum()} / {weight_original.size}")
print()

# Decompress to GPU
print("Step 6: Decompress to GPU (inference path)")
import ctypes

gpu_ptr, rows, cols, actual_size = decoder.decode_layer_to_gpu(compressed)
print(f"  GPU pointer: 0x{gpu_ptr:x}")
print(f"  Dimensions: {rows}x{cols}")
print()

# Copy from GPU
weight_bytes_gpu = torch.empty((rows, cols), dtype=torch.int8, device=device)
cudart = ctypes.CDLL('libcudart.so')
cudart.cudaMemcpy(
    ctypes.c_void_p(weight_bytes_gpu.data_ptr()),
    ctypes.c_void_p(gpu_ptr),
    ctypes.c_size_t(rows * cols),
    ctypes.c_int(1)
)
cudart.cudaFree(ctypes.c_void_p(gpu_ptr))

# Reinterpret as FP16
weight_as_bytes = weight_bytes_gpu.flatten()
weight_as_int16 = weight_as_bytes.view(torch.int16)
weight_reconstructed_gpu = weight_as_int16.view(torch.float16).reshape(weight_original.shape)

print("Step 7: Check GPU reconstruction")
weight_reconstructed_gpu_np = weight_reconstructed_gpu.cpu().numpy()
print(f"  Reconstructed shape: {weight_reconstructed_gpu_np.shape}")
print(f"  First 5 values: {weight_reconstructed_gpu_np.flatten()[:5]}")
print()

if np.array_equal(weight_original, weight_reconstructed_gpu_np):
    print("✓ GPU: Bit-exact match!")
else:
    print("✗ GPU: NOT bit-exact")
    diff = np.abs(weight_original - weight_reconstructed_gpu_np)
    print(f"  Max difference: {diff.max()}")
    print(f"  Mean difference: {diff.mean()}")
    print(f"  Different elements: {(diff > 0).sum()} / {weight_original.size}")
print()

# Final verdict
print("="*80)
print("VERDICT")
print("="*80)
print()

cpu_exact = np.array_equal(weight_original, weight_reconstructed_cpu)
gpu_exact = np.array_equal(weight_original, weight_reconstructed_gpu_np)

if cpu_exact and gpu_exact:
    print("✓✓✓ COMPRESSION IS BIT-EXACT ✓✓✓")
    print()
    print("Zstd compression/decompression preserves FP16 values perfectly.")
    print("The KV cache issue must be something else:")
    print("  - Attention mechanism sensitivity to weight load order?")
    print("  - PyTorch caching/fusion differences?")
    print("  - Need to compress O projection differently?")
elif cpu_exact and not gpu_exact:
    print("✗ GPU PATH HAS ERRORS")
    print()
    print("CPU decode is perfect, but GPU-direct decode introduces differences.")
    print("This is the root cause! GPU decode path needs fixing.")
elif not cpu_exact:
    print("✗ COMPRESSION ITSELF HAS ERRORS")
    print()
    print("Zstd is not preserving the data correctly.")
    print("This is the root cause! Need to fix encoder/decoder.")
else:
    print("✗ UNEXPECTED STATE")
    print("Both paths have errors but differently?")

print()

