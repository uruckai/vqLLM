#!/usr/bin/env python3
"""
Quick diagnostic to check RunPod environment is ready
"""

import sys
from pathlib import Path

print("="*80)
print("RUNPOD ENVIRONMENT CHECK")
print("="*80)
print()

# 1. Check CUDA
print("[1/6] Checking CUDA...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"  ✗ CUDA not available!")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ PyTorch not available: {e}")
    sys.exit(1)

print()

# 2. Check codec library
print("[2/6] Checking codec library...")
lib_path = Path(__file__).parent / "build" / "libcodec_core.so"
if lib_path.exists():
    print(f"  ✓ Library found: {lib_path}")
    print(f"  Size: {lib_path.stat().st_size / 1024:.1f} KB")
else:
    print(f"  ✗ Library not found: {lib_path}")
    print(f"  Run: bash build.sh")
    sys.exit(1)

print()

# 3. Check GPU decoder availability
print("[3/6] Checking GPU decoder...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from bindings_zstd import ZstdGPUDecoder, ZstdEncoder
    
    if ZstdGPUDecoder.is_available():
        print(f"  ✓ GPU decoder available")
    else:
        print(f"  ✗ GPU decoder not available!")
        print(f"  Check nvCOMP installation")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed to import: {e}")
    sys.exit(1)

print()

# 4. Test compression round-trip
print("[4/6] Testing compression round-trip...")
try:
    import numpy as np
    
    encoder = ZstdEncoder(compression_level=9)
    decoder = ZstdGPUDecoder()
    
    # Create test data
    test_data = np.random.randint(-127, 127, size=(256, 256), dtype=np.int8)
    
    # Compress
    compressed, ratio = encoder.encode_layer(test_data)
    print(f"  Test compression: {len(compressed)} bytes, ratio={ratio:.2f}x")
    
    # Decompress
    decompressed = decoder.decode_layer(compressed)
    
    # Verify
    if np.array_equal(test_data, decompressed):
        print(f"  ✓ Round-trip test passed (bit-exact)")
    else:
        errors = np.sum(test_data != decompressed)
        print(f"  ✗ Round-trip test failed: {errors} errors")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Round-trip test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 5. Check Transformers
print("[5/6] Checking Transformers...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  ✓ Transformers available")
except Exception as e:
    print(f"  ✗ Transformers not available: {e}")
    sys.exit(1)

print()

# 6. GPU decode to device pointer test
print("[6/6] Testing GPU decode to device pointer...")
try:
    import ctypes
    
    test_data = np.random.randint(-127, 127, size=(128, 128), dtype=np.int8)
    compressed, ratio = encoder.encode_layer(test_data)
    
    # Decode to GPU
    gpu_ptr, rows, cols, dtype = decoder.decode_layer_to_gpu(compressed)
    
    if gpu_ptr and rows == 128 and cols == 128:
        print(f"  ✓ GPU decode returned valid pointer: {hex(gpu_ptr)}")
        print(f"  Dimensions: {rows}x{cols}")
        
        # Free the GPU memory
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
        print(f"  ✓ Memory freed successfully")
    else:
        print(f"  ✗ GPU decode failed")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ GPU decode test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*80)
print("✓ ALL CHECKS PASSED - READY FOR TESTING")
print("="*80)
print()
print("Next steps:")
print("  1. Run progressive test: python3 test_progressive_compression.py")
print("  2. Or use the runner: bash RUN_PROGRESSIVE_TEST.sh")

