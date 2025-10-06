#!/usr/bin/env python3
"""
Complete compression roundtrip test
Tests actual encode→decode with compression measurement
"""

import sys
import numpy as np
from pathlib import Path
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    from wcodec.bindings import Encoder, Decoder
except Exception as e:
    print(f"✗ Failed to import bindings: {e}")
    print("\nMake sure the library is built:")
    print("  cd build && make -j8")
    sys.exit(1)


def test_simple_roundtrip():
    """Test encode/decode on small random data"""
    print("\n" + "="*60)
    print("[TEST 1] Simple 64×64 Roundtrip")
    print("="*60)
    
    # Create test data
    np.random.seed(42)
    original = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
    
    print(f"\nOriginal data:")
    print(f"  Shape: {original.shape}")
    print(f"  Size: {original.nbytes} bytes")
    print(f"  Range: [{original.min()}, {original.max()}]")
    
    # Encode
    encoder = Encoder(tile_size=16)
    encoded, encode_stats = encoder.encode_layer(original)
    
    print(f"\nEncoded:")
    print(f"  Compressed size: {len(encoded)} bytes")
    print(f"  Compression ratio: {encode_stats['compression_ratio']:.2f}x")
    print(f"  Space savings: {encode_stats['compression_percent']:.1f}%")
    print(f"  Encode time: {encode_stats['encode_time_ms']:.2f}ms")
    
    # Decode
    decoder = Decoder(tile_size=16)
    decoded, decode_stats = decoder.decode_layer(encoded, 64, 64)
    
    print(f"\nDecoded:")
    print(f"  Decode time: {decode_stats['decode_time_ms']:.2f}ms")
    
    # Verify bit-exact
    if np.array_equal(original, decoded):
        print(f"\n✓ Bit-exact reconstruction!")
        return True, encode_stats
    else:
        diff = np.sum(original != decoded)
        max_diff = np.abs(original - decoded).max()
        print(f"\n✗ FAILED: {diff} mismatched values (max diff: {max_diff})")
        return False, encode_stats


def test_various_sizes():
    """Test different matrix sizes"""
    print("\n" + "="*60)
    print("[TEST 2] Various Sizes")
    print("="*60)
    
    sizes = [
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (64, 128),
        (256, 256)
    ]
    
    encoder = Encoder(tile_size=16)
    decoder = Decoder(tile_size=16)
    
    all_passed = True
    all_stats = []
    
    for rows, cols in sizes:
        np.random.seed(42)
        original = np.random.randint(-128, 127, (rows, cols), dtype=np.int8)
        
        encoded, stats = encoder.encode_layer(original)
        decoded, _ = decoder.decode_layer(encoded, rows, cols)
        
        passed = np.array_equal(original, decoded)
        status = "✓" if passed else "✗"
        
        print(f"  {status} {rows:3d}×{cols:3d}: "
              f"{stats['compression_ratio']:5.2f}x compression, "
              f"{stats['compression_percent']:5.1f}% savings")
        
        all_passed = all_passed and passed
        all_stats.append(stats)
    
    # Average stats
    avg_ratio = np.mean([s['compression_ratio'] for s in all_stats])
    avg_savings = np.mean([s['compression_percent'] for s in all_stats])
    
    print(f"\nAverage: {avg_ratio:.2f}x compression, {avg_savings:.1f}% savings")
    
    return all_passed


def test_patterns():
    """Test on different data patterns"""
    print("\n" + "="*60)
    print("[TEST 3] Data Patterns")
    print("="*60)
    
    encoder = Encoder(tile_size=16)
    decoder = Decoder(tile_size=16)
    
    tests = [
        ("Zeros", np.zeros((64, 64), dtype=np.int8)),
        ("Ones", np.ones((64, 64), dtype=np.int8)),
        ("Constant 42", np.full((64, 64), 42, dtype=np.int8)),
        ("Random", np.random.randint(-128, 127, (64, 64), dtype=np.int8)),
        ("Structured", (np.arange(64*64, dtype=np.int16) % 256 - 128).astype(np.int8).reshape(64, 64)),
        ("Smooth gradient", np.outer(np.arange(64), np.arange(64)).astype(np.int8)),
    ]
    
    all_passed = True
    for name, data in tests:
        encoded, stats = encoder.encode_layer(data)
        decoded, _ = decoder.decode_layer(encoded, 64, 64)
        
        passed = np.array_equal(data, decoded)
        status = "✓" if passed else "✗"
        ratio = stats['compression_ratio']
        savings = stats['compression_percent']
        
        print(f"  {status} {name:20s}: {ratio:6.2f}x ({savings:5.1f}% savings)")
        
        all_passed = all_passed and passed
    
    return all_passed


def test_synthetic_checkpoint():
    """Test on synthetic checkpoint layer"""
    print("\n" + "="*60)
    print("[TEST 4] Synthetic Checkpoint Layer")
    print("="*60)
    
    # Simulate a typical LLM layer (attention projection)
    size = (4096, 4096)
    print(f"\nSimulating {size[0]}×{size[1]} attention layer")
    print(f"Original size: {size[0] * size[1] / (1024**2):.2f} MB")
    
    np.random.seed(42)
    # More realistic: centered around 0, limited range
    layer = np.random.normal(0, 30, size).astype(np.int8)
    
    encoder = Encoder(tile_size=16)
    decoder = Decoder(tile_size=16)
    
    print("\nEncoding...")
    start = time.time()
    encoded, stats = encoder.encode_layer(layer)
    encode_time = time.time() - start
    
    print(f"  Original: {layer.nbytes / (1024**2):.2f} MB")
    print(f"  Compressed: {len(encoded) / (1024**2):.2f} MB")
    print(f"  Compression: {stats['compression_ratio']:.2f}x ({stats['compression_percent']:.1f}% savings)")
    print(f"  Time: {encode_time:.2f}s ({layer.nbytes / (1024**2) / encode_time:.1f} MB/s)")
    
    print("\nDecoding...")
    start = time.time()
    decoded, _ = decoder.decode_layer(encoded, size[0], size[1])
    decode_time = time.time() - start
    
    print(f"  Time: {decode_time:.2f}s ({layer.nbytes / (1024**2) / decode_time:.1f} MB/s)")
    
    # Verify
    if np.array_equal(layer, decoded):
        print(f"\n✓ Bit-exact reconstruction on {layer.nbytes / (1024**2):.1f} MB layer!")
        return True, stats
    else:
        diff = np.sum(layer != decoded)
        print(f"\n✗ FAILED: {diff} mismatches")
        return False, stats


def main():
    print("="*60)
    print("Weight Codec - Compression Roundtrip Tests")
    print("="*60)
    
    results = []
    all_stats = []
    
    try:
        # Test 1
        result, stats = test_simple_roundtrip()
        results.append(result)
        all_stats.append(stats)
        
        # Test 2
        results.append(test_various_sizes())
        
        # Test 3
        results.append(test_patterns())
        
        # Test 4
        result, stats = test_synthetic_checkpoint()
        results.append(result)
        all_stats.append(stats)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if all(results):
            print("✓ ALL TESTS PASSED!")
            print("\nCompression Results:")
            for i, stats in enumerate(all_stats, 1):
                if 'compression_ratio' in stats:
                    print(f"  Test {i}: {stats['compression_ratio']:.2f}x compression "
                          f"({stats['compression_percent']:.1f}% savings)")
            
            avg_ratio = np.mean([s['compression_ratio'] for s in all_stats])
            avg_savings = np.mean([s['compression_percent'] for s in all_stats])
            
            print(f"\n  Average: {avg_ratio:.2f}x compression ({avg_savings:.1f}% savings)")
            print("\n" + "="*60)
            print("Week 2+3 VERIFIED: Codec works with real compression!")
            print("="*60)
            return 0
        else:
            print("✗ SOME TESTS FAILED")
            failed = sum(1 for r in results if not r)
            print(f"  {failed}/{len(results)} tests failed")
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

