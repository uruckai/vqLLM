#!/usr/bin/env python3
"""
Test nvCOMP with RAW Zstd data (no custom header)
"""

import sys
from pathlib import Path
import numpy as np
import ctypes

sys.path.insert(0, str(Path(__file__).parent))

print("Testing RAW nvCOMP Zstd decode...")
print()

# Compress with standard Python zstd
import zstandard as zstd

data = np.random.randint(-127, 127, size=(256, 256), dtype=np.int8)
print(f"Original shape: {data.shape}")
print(f"Original size: {data.nbytes} bytes")

# Compress with zstandard
cctx = zstd.ZstdCompressor(level=9)
compressed = cctx.compress(data.tobytes())

print(f"Compressed size: {len(compressed)} bytes")
print(f"Ratio: {data.nbytes / len(compressed):.2f}x")
print()

# Try nvCOMP batched API manually
cuda = ctypes.CDLL('libcudart.so')
nvcomp = ctypes.CDLL('/workspace/nvcomp_install/lib/libnvcomp.so')

# Allocate GPU memory for compressed data
d_compressed = ctypes.c_void_p()
cuda.cudaMalloc(ctypes.byref(d_compressed), len(compressed))

# Copy compressed to GPU
cuda.cudaMemcpy(d_compressed, compressed, len(compressed), 1)  # H2D

# Get temp size
temp_size = ctypes.c_size_t()
print(f"Calling nvcompBatchedZstdDecompressGetTempSize...")
print(f"  max_uncompressed_chunk_size: {data.nbytes}")
print(f"  batch_size: 1")

ret = nvcomp.nvcompBatchedZstdDecompressGetTempSize(
    ctypes.c_size_t(data.nbytes),  # max uncompressed size
    ctypes.c_size_t(1),  # batch
    ctypes.byref(temp_size)
)

print(f"  Return: {ret}")
print(f"  Temp size: {temp_size.value} bytes ({temp_size.value/1024/1024:.1f} MB)")
print()

if ret != 0:
    print(f"❌ nvcompBatchedZstdDecompressGetTempSize failed: {ret}")
    sys.exit(1)

# Allocate GPU buffers
d_decompressed = ctypes.c_void_p()
cuda.cudaMalloc(ctypes.byref(d_decompressed), data.nbytes)

d_temp = ctypes.c_void_p()
cuda.cudaMalloc(ctypes.byref(d_temp), temp_size.value)

# Allocate GPU arrays for pointers
d_compressed_ptrs = ctypes.c_void_p()
d_decompressed_ptrs = ctypes.c_void_p()
d_compressed_sizes = ctypes.c_void_p()
d_decompressed_sizes = ctypes.c_void_p()

cuda.cudaMalloc(ctypes.byref(d_compressed_ptrs), 8)
cuda.cudaMalloc(ctypes.byref(d_decompressed_ptrs), 8)
cuda.cudaMalloc(ctypes.byref(d_compressed_sizes), 8)
cuda.cudaMalloc(ctypes.byref(d_decompressed_sizes), 8)

# Copy values to GPU arrays
cuda.cudaMemcpy(d_compressed_ptrs, ctypes.byref(d_compressed), 8, 1)
cuda.cudaMemcpy(d_decompressed_ptrs, ctypes.byref(d_decompressed), 8, 1)

comp_size = ctypes.c_size_t(len(compressed))
decomp_size = ctypes.c_size_t(data.nbytes)
cuda.cudaMemcpy(d_compressed_sizes, ctypes.byref(comp_size), 8, 1)
cuda.cudaMemcpy(d_decompressed_sizes, ctypes.byref(decomp_size), 8, 1)

print(f"Calling nvcompBatchedZstdDecompressAsync...")
print(f"  compressed_size: {len(compressed)}")
print(f"  uncompressed_size: {data.nbytes}")

ret = nvcomp.nvcompBatchedZstdDecompressAsync(
    d_compressed_ptrs,
    d_compressed_sizes,
    d_decompressed_sizes,
    None,  # actual sizes out
    ctypes.c_size_t(1),  # batch
    d_temp,
    temp_size,
    d_decompressed_ptrs,
    None,  # statuses
    None   # stream
)

print(f"  Return: {ret}")

if ret != 0:
    print(f"❌ nvcompBatchedZstdDecompressAsync failed: {ret}")
    sys.exit(1)

cuda.cudaDeviceSynchronize()
print("✓ Decompression succeeded!")

# Copy back
result = np.zeros(data.nbytes, dtype=np.int8)
cuda.cudaMemcpy(result.ctypes.data, d_decompressed, data.nbytes, 2)  # D2H

result = result.reshape(256, 256)

if np.array_equal(data, result):
    print("✓ Bit-exact match!")
else:
    errors = np.sum(data != result)
    print(f"✗ {errors} errors")

# Cleanup
cuda.cudaFree(d_compressed)
cuda.cudaFree(d_decompressed)
cuda.cudaFree(d_temp)
cuda.cudaFree(d_compressed_ptrs)
cuda.cudaFree(d_decompressed_ptrs)
cuda.cudaFree(d_compressed_sizes)
cuda.cudaFree(d_decompressed_sizes)

print("Done!")

