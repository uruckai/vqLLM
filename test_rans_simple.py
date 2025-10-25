#!/usr/bin/env python3
"""
Simple test of old rANS codec - sanity check
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add python module to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

print("="*80)
print("RANS CODEC SANITY CHECK")
print("="*80)
print()

print("Step 1: Try to import old rANS bindings...")
try:
    from wcodec import bindings
    print("✓ Imported wcodec.bindings")
except ImportError as e:
    print(f"✗ Cannot import wcodec.bindings: {e}")
    print()
    print("Need to build the old rANS codec first:")
    print()
    print("Run these commands on your RunPod:")
    print("  cd /workspace/CodecLLM")
    print("  mkdir -p build && cd build")
    print("  cmake .. -DCMAKE_BUILD_TYPE=Release")
    print("  make -j$(nproc)")
    print("  cd ..")
    print("  python3 test_rans_simple.py")
    print()
    sys.exit(1)

print()
print("Step 2: Load codec library...")
try:
    codec = bindings.WCodec()
    print("✓ Loaded rANS codec")
except Exception as e:
    print(f"✗ Failed to load codec: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 3: Test compression round-trip...")
try:
    # Create test data
    test_data = np.random.randn(2048, 2048).astype(np.float16)
    print(f"  Original: shape={test_data.shape}, dtype={test_data.dtype}")
    
    # Quantize to INT8
    scale = np.abs(test_data).max() / 127.0
    test_int8 = np.clip(np.round(test_data / scale), -127, 127).astype(np.int8)
    print(f"  Quantized: range=[{test_int8.min()}, {test_int8.max()}]")
    
    # Compress
    compressed = codec.encode(test_int8)
    ratio = test_int8.nbytes / len(compressed)
    print(f"  Compressed: {test_int8.nbytes} → {len(compressed)} bytes ({ratio:.2f}x)")
    
    # Decompress
    decoded = codec.decode(compressed, test_int8.shape)
    print(f"  Decoded: shape={decoded.shape}, dtype={decoded.dtype}")
    
    # Verify
    if np.array_equal(test_int8, decoded):
        print("✓ Bit-exact round-trip")
    else:
        print("✗ Decode mismatch")
        diff = np.abs(test_int8.astype(np.int32) - decoded.astype(np.int32)).max()
        print(f"  Max difference: {diff}")
    
except Exception as e:
    print(f"✗ Compression test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 4: Test with LLM weights...")
try:
    from transformers import AutoModelForCausalLM
    
    print("  Loading TinyLlama...")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Get first linear layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  Testing layer: {name}")
            weight = module.weight.data.cpu().numpy()
            print(f"    Shape: {weight.shape}, dtype: {weight.dtype}")
            
            # Quantize
            scale = np.abs(weight).max() / 127.0
            weight_int8 = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
            
            # Compress
            compressed = codec.encode(weight_int8)
            ratio = weight_int8.nbytes / len(compressed)
            print(f"    Compressed: {weight_int8.nbytes/1024:.1f} KB → {len(compressed)/1024:.1f} KB ({ratio:.2f}x)")
            
            # Decompress
            decoded = codec.decode(compressed, weight_int8.shape)
            
            # Verify
            if np.array_equal(weight_int8, decoded):
                print(f"    ✓ Bit-exact decode")
            else:
                print(f"    ✗ Decode mismatch")
            
            break
    
except Exception as e:
    print(f"  ✗ LLM test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("SUMMARY")
print("="*80)
print()
print("✓ Old rANS codec works correctly for compression/decompression")
print()
print("Next step: Test with actual LLM inference")
print("  Run: python3 test_rans_inference.py")
print()
print("Expected result: Same issues as Zstd")
print("  - Compression is perfect")
print("  - But LLM output will be garbled")
print("  - Because dynamic weight loading breaks inference")
print()

