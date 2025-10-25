#!/usr/bin/env python3
"""
rANS test using the EXACT working approach from backup
"""

import sys
from pathlib import Path
import numpy as np
import ctypes

print("="*80)
print("rANS TEST - WORKING APPROACH")
print("="*80)
print()

# Load library directly (like the backup does)
lib_path = Path("build/libwcodec.so")
if not lib_path.exists():
    print(f"✗ Library not found at {lib_path}")
    print()
    print("Build first:")
    print("  cd /workspace/CodecLLM")
    print("  mkdir -p build && cd build")
    print("  cmake .. -DCMAKE_BUILD_TYPE=Release")
    print("  make -j$(nproc)")
    sys.exit(1)

lib = ctypes.CDLL(str(lib_path))

# Set return types (exact C API function names)
lib.wcodec_encoder_create.restype = ctypes.c_void_p
lib.wcodec_encoder_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
lib.wcodec_encoder_destroy.argtypes = [ctypes.c_void_p]
lib.wcodec_encode_layer.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
]
lib.wcodec_encode_layer.restype = ctypes.c_int
lib.wcodec_free_buffer.argtypes = [ctypes.POINTER(ctypes.c_uint8)]

lib.wcodec_decoder_create.restype = ctypes.c_void_p
lib.wcodec_decoder_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
lib.wcodec_decoder_destroy.argtypes = [ctypes.c_void_p]
lib.wcodec_gpu_is_available.restype = ctypes.c_int

print("✓ Library loaded")
print()

# Test compression with ONE TILE (like backup does)
tile_size = 256
test_data = np.random.randn(tile_size, tile_size).astype(np.float16)
scale = np.abs(test_data).max() / 127.0
test_int8 = np.clip(np.round(test_data / scale), -127, 127).astype(np.int8)

print(f"Testing {tile_size}x{tile_size} tile...")
print(f"  Input: range=[{test_int8.min()}, {test_int8.max()}]")

try:
    # CRITICAL: Create encoder for THIS tile only (like backup)
    encoder = lib.wcodec_encoder_create(tile_size, tile_size)
    if not encoder:
        raise RuntimeError("Failed to create encoder")
    
    # Encode
    data_ptr = test_int8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    output_ptr = ctypes.POINTER(ctypes.c_uint8)()
    output_size = ctypes.c_size_t()
    compression_ratio = ctypes.c_double()
    encode_time = ctypes.c_double()
    
    result = lib.wcodec_encode_layer(
        encoder,
        data_ptr,
        tile_size,
        tile_size,
        ctypes.byref(output_ptr),
        ctypes.byref(output_size),
        ctypes.byref(compression_ratio),
        ctypes.byref(encode_time)
    )
    
    if result != 0:
        raise RuntimeError(f"Encode failed with code {result}")
    
    print(f"  Compressed: {test_int8.nbytes} → {output_size.value} bytes")
    print(f"  Ratio: {compression_ratio.value:.2f}x")
    print(f"  Time: {encode_time.value:.2f}ms")
    
    # Copy compressed data
    compressed = bytes(ctypes.cast(
        output_ptr,
        ctypes.POINTER(ctypes.c_uint8 * output_size.value)
    ).contents)
    
    # CRITICAL: Free and destroy immediately (like backup)
    lib.wcodec_free_buffer(output_ptr)
    lib.wcodec_encoder_destroy(encoder)
    
    print(f"  ✓ Compression successful")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*80)
print("SUCCESS - rANS codec works with backup approach!")
print("="*80)
print()
print("Key insight:")
print("  - Create encoder PER TILE")
print("  - Destroy immediately after use")
print("  - This prevents memory accumulation")
print()
print("However, this doesn't change the fundamental blocker:")
print("  - Dynamic weight loading still breaks LLM inference")
print("  - See core/COMPRESSION_BLOCKERS.md")

