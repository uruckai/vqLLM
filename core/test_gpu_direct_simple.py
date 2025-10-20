#!/usr/bin/env python3
"""
Simple test of GPU-direct decode
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder

print("Testing GPU-direct decode...")
print()

if not ZstdGPUDecoder.is_available():
    print("❌ GPU decoder not available")
    sys.exit(1)

print("✓ GPU decoder available")
print()

# Create test data
print("Creating test data (256x256)...")
data = np.random.randint(-127, 127, size=(256, 256), dtype=np.int8)
print(f"  Data shape: {data.shape}")
print(f"  Data size: {data.nbytes} bytes")
print()

# Compress
print("Compressing...")
encoder = ZstdEncoder(compression_level=9)
compressed, ratio = encoder.encode_layer(data)
print(f"  Compressed size: {len(compressed)} bytes")
print(f"  Ratio: {ratio:.2f}x")
print()

# Decode to CPU (for comparison)
print("Decoding to CPU...")
decoder = ZstdGPUDecoder()
try:
    cpu_result = decoder.decode_layer(compressed)
    print(f"  ✓ CPU decode successful")
    print(f"  Shape: {cpu_result.shape}")
    if np.array_equal(data, cpu_result):
        print(f"  ✓ Bit-exact match")
    else:
        print(f"  ✗ Mismatch!")
except Exception as e:
    print(f"  ✗ CPU decode failed: {e}")
print()

# Decode to GPU
print("Decoding to GPU...")
try:
    gpu_ptr, rows, cols, dtype = decoder.decode_layer_to_gpu(compressed)
    print(f"  ✓ GPU decode successful")
    print(f"  GPU pointer: 0x{gpu_ptr:x}")
    print(f"  Rows: {rows}, Cols: {cols}")
    print(f"  Dtype: {dtype}")
    
    # Now try to read it back
    print()
    print("Reading back from GPU...")
    import torch
    
    # Create a PyTorch tensor to copy into
    result_gpu = torch.empty((rows, cols), dtype=torch.int8, device='cuda')
    
    # Copy from our GPU buffer to PyTorch
    import ctypes
    cuda = ctypes.CDLL('libcudart.so')
    
    dst_ptr = result_gpu.data_ptr()
    src_ptr = gpu_ptr
    size = rows * cols
    
    print(f"  Copying {size} bytes from 0x{src_ptr:x} to 0x{dst_ptr:x}")
    
    ret = cuda.cudaMemcpy(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(size),
        ctypes.c_int(3)  # cudaMemcpyDeviceToDevice
    )
    
    if ret != 0:
        print(f"  ✗ cudaMemcpy failed with error {ret}")
    else:
        print(f"  ✓ cudaMemcpy succeeded")
        
        # Check result
        result_cpu = result_gpu.cpu().numpy()
        if np.array_equal(data, result_cpu):
            print(f"  ✓ Bit-exact match!")
        else:
            errors = np.sum(data != result_cpu)
            print(f"  ✗ {errors} errors")
    
    # Free
    print()
    print("Freeing GPU memory...")
    ret = cuda.cudaFree(ctypes.c_void_p(gpu_ptr))
    if ret != 0:
        print(f"  ✗ cudaFree failed with error {ret}")
    else:
        print(f"  ✓ cudaFree succeeded")
    
except Exception as e:
    print(f"  ✗ GPU decode failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("Test complete!")

