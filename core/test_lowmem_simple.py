#!/usr/bin/env python3
"""
Simple test for low-memory inference approach
Tests the basic compression/decompression flow with PyTorch
"""

import numpy as np
import sys
from pathlib import Path

def test_compressed_tensor():
    """Test CompressedTensor class"""
    print("="*80)
    print("TEST 1: CompressedTensor Compression/Decompression")
    print("="*80)
    
    # Import after path check
    sys.path.insert(0, str(Path(__file__).parent))
    from test_inference_lowmem import load_codec, CompressedTensor
    
    try:
        import torch
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Load codec
    print("\n[1/4] Loading codec...")
    lib = load_codec()
    if lib is None:
        return False
    print("✓ Codec loaded")
    
    # Create test tensor
    print("\n[2/4] Creating test tensor...")
    test_tensor = torch.randn(2048, 2048, dtype=torch.float16)
    original_size = test_tensor.element_size() * test_tensor.numel()
    print(f"  Shape: {test_tensor.shape}")
    print(f"  Dtype: {test_tensor.dtype}")
    print(f"  Size: {original_size / 1024**2:.2f} MB")
    
    # Compress
    print("\n[3/4] Compressing tensor...")
    try:
        compressed = CompressedTensor(lib, test_tensor)
        print(f"  Original size: {compressed.original_size / 1024**2:.2f} MB")
        print(f"  Compressed size: {compressed.compressed_size / 1024**2:.2f} MB")
        print(f"  Compression ratio: {compressed.get_compression_ratio():.2f}x")
        print(f"  Space saved: {(1 - 1/compressed.get_compression_ratio())*100:.1f}%")
    except Exception as e:
        print(f"✗ Compression failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Decompress
    print("\n[4/4] Decompressing tensor...")
    try:
        decompressed = compressed.decompress()
        print(f"  Decompressed shape: {decompressed.shape}")
        print(f"  Decompressed dtype: {decompressed.dtype}")
        
        # Check reconstruction quality (allowing for quantization error)
        diff = (test_tensor - decompressed).abs()
        max_error = diff.max().item()
        mean_error = diff.mean().item()
        
        print(f"  Max error: {max_error:.6f}")
        print(f"  Mean error: {mean_error:.6f}")
        
        # With INT8 quantization, expect some error but should be small
        if mean_error < 0.01:  # Less than 1% average error
            print("✅ Reconstruction quality: GOOD")
            return True
        else:
            print("⚠️  Reconstruction error higher than expected")
            return False
            
    except Exception as e:
        print(f"✗ Decompression failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compressed_linear():
    """Test CompressedLinear layer"""
    print("\n" + "="*80)
    print("TEST 2: CompressedLinear Layer")
    print("="*80)
    
    sys.path.insert(0, str(Path(__file__).parent))
    from test_inference_lowmem import load_codec, CompressedLinear
    
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Load codec
    print("\n[1/5] Loading codec...")
    lib = load_codec()
    if lib is None:
        return False
    print("✓ Codec loaded")
    
    # Create test linear layer
    print("\n[2/5] Creating test linear layer...")
    original_layer = nn.Linear(2048, 2048, bias=True)
    original_layer.weight.data = torch.randn_like(original_layer.weight) * 0.01
    original_layer.bias.data = torch.randn_like(original_layer.bias) * 0.01
    print(f"  Input features: {original_layer.in_features}")
    print(f"  Output features: {original_layer.out_features}")
    
    # Compress layer
    print("\n[3/5] Compressing layer...")
    try:
        compressed_layer = CompressedLinear(original_layer, lib)
        ratio = compressed_layer.compressed_weight.get_compression_ratio()
        print(f"  Compression ratio: {ratio:.2f}x")
    except Exception as e:
        print(f"✗ Compression failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass
    print("\n[4/5] Testing forward pass...")
    test_input = torch.randn(8, 2048)  # Batch of 8
    
    try:
        # Original layer output
        with torch.no_grad():
            original_output = original_layer(test_input)
        
        # Compressed layer output
        with torch.no_grad():
            compressed_output = compressed_layer.forward(test_input)
        
        print(f"  Original output shape: {original_output.shape}")
        print(f"  Compressed output shape: {compressed_output.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare outputs
    print("\n[5/5] Comparing outputs...")
    diff = (original_output - compressed_output).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    
    if mean_error < 0.1:  # Reasonable error threshold
        print("✅ Forward pass quality: GOOD")
        print(f"  Decode count: {compressed_layer.decode_count}")
        print(f"  Decode time: {compressed_layer.decode_time*1000:.1f}ms")
        return True
    else:
        print("⚠️  Forward pass error higher than expected")
        return False

def main():
    print("\n" + "="*80)
    print("SIMPLE LOW-MEMORY INFERENCE TESTS")
    print("="*80)
    print("\nThis tests the basic building blocks for low-memory inference:")
    print("1. CompressedTensor: Compress/decompress individual tensors")
    print("2. CompressedLinear: Replace nn.Linear with on-demand decompression")
    print()
    
    # Run tests
    test1_pass = test_compressed_tensor()
    test2_pass = test_compressed_linear()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Test 1 (CompressedTensor): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (CompressedLinear):  {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n✅ ALL TESTS PASSED!")
        print("\nNext step: Run full inference test with test_inference_lowmem.py")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

