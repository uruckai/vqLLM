#!/usr/bin/env python3
"""
End-to-end test for core codec
"""

import numpy as np
import time
from bindings import Encoder, GPUDecoder

def test_small():
    """Quick test on small data"""
    print("\n=== Test 1: Small Data (64x64) ===")
    
    data = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
    print(f"Original: {data.shape}, {data.nbytes} bytes")
    
    # Encode
    encoder = Encoder(tile_size=16)
    compressed, ratio = encoder.encode(data)
    print(f"Compressed: {len(compressed)} bytes ({ratio:.2f}x)")
    
    # Decode
    if GPUDecoder.is_available():
        decoder = GPUDecoder()
        decoded, time_ms = decoder.decode(compressed, 64, 64)
        print(f"Decode time: {time_ms:.3f} ms")
        
        # Verify
        match = np.array_equal(data, decoded)
        print(f"Bit-exact: {match}")
        
        if not match:
            diff = np.sum(data != decoded)
            print(f"  Errors: {diff} / {data.size}")
            print(f"  First error at: {np.where(data != decoded)}")
            return False
        
        return True
    else:
        print("GPU not available")
        return False


def test_medium():
    """Test on medium data (256x256)"""
    print("\n=== Test 2: Medium Data (256x256) ===")
    
    data = np.random.randint(-128, 127, (256, 256), dtype=np.int8)
    print(f"Original: {data.shape}, {data.nbytes / 1024:.1f} KB")
    
    # Encode
    encoder = Encoder(tile_size=16)
    start = time.time()
    compressed, ratio = encoder.encode(data)
    encode_time = (time.time() - start) * 1000
    print(f"Compressed: {len(compressed) / 1024:.1f} KB ({ratio:.2f}x)")
    print(f"Encode time: {encode_time:.1f} ms")
    
    # Decode
    if GPUDecoder.is_available():
        decoder = GPUDecoder()
        decoded, decode_time = decoder.decode(compressed, 256, 256)
        print(f"Decode time: {decode_time:.1f} ms")
        print(f"Speedup: {encode_time / decode_time:.1f}x faster")
        
        # Verify
        match = np.array_equal(data, decoded)
        print(f"Bit-exact: {match}")
        
        return match
    else:
        print("GPU not available")
        return False


def test_llm_like():
    """Test on LLM-like dimensions (4096x4096)"""
    print("\n=== Test 3: LLM-like Data (4096x4096) ===")
    
    # Simulate LLM weight statistics
    # Real LLM weights have specific distributions
    data = np.random.normal(0, 20, (4096, 4096)).astype(np.int8)
    print(f"Original: {data.shape}, {data.nbytes / (1024*1024):.1f} MB")
    
    # Encode
    encoder = Encoder(tile_size=16)
    start = time.time()
    compressed, ratio = encoder.encode(data)
    encode_time = (time.time() - start) * 1000
    print(f"Compressed: {len(compressed) / (1024*1024):.1f} MB ({ratio:.2f}x)")
    print(f"Encode time: {encode_time:.0f} ms")
    
    # Decode
    if GPUDecoder.is_available():
        decoder = GPUDecoder()
        decoded, decode_time = decoder.decode(compressed, 4096, 4096)
        print(f"Decode time: {decode_time:.1f} ms")
        print(f"Throughput: {data.nbytes / (decode_time * 1e6):.1f} GB/s")
        print(f"Speedup: {encode_time / decode_time:.0f}x faster")
        
        # Verify
        match = np.array_equal(data, decoded)
        print(f"Bit-exact: {match}")
        
        return match
    else:
        print("GPU not available")
        return False


def test_compression_quality():
    """Test compression on different data patterns"""
    print("\n=== Test 4: Compression Quality ===")
    
    encoder = Encoder(tile_size=16)
    
    patterns = [
        ("Random", np.random.randint(-128, 127, (512, 512), dtype=np.int8)),
        ("Smooth", np.fromfunction(lambda i, j: (i + j) // 4, (512, 512), dtype=np.int8)),
        ("Sparse", np.random.choice([0, 0, 0, 1, -1], (512, 512)).astype(np.int8)),
    ]
    
    for name, data in patterns:
        compressed, ratio = encoder.encode(data)
        print(f"{name:10s}: {ratio:.2f}x ({data.nbytes} -> {len(compressed)} bytes)")
    
    return True


def main():
    print("=" * 60)
    print("Core Codec - End-to-End Test")
    print("=" * 60)
    
    # Check GPU
    if not GPUDecoder.is_available():
        print("\n⚠️  GPU not available - tests will be limited")
    
    # Run tests
    tests = [
        ("Small Data", test_small),
        ("Medium Data", test_medium),
        ("LLM-like Data", test_llm_like),
        ("Compression Quality", test_compression_quality),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {name} PASSED")
            else:
                failed += 1
                print(f"✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! Core codec is working.")
        print("\nNext steps:")
        print("1. Test on real LLM checkpoints")
        print("2. Optimize GPU kernels for throughput")
        print("3. Integrate with PyTorch model loading")
    else:
        print("\n⚠️  Some tests failed - check implementation")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)

