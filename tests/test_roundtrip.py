"""
Integration test: Encode -> Decode roundtrip
Tests Week 2 implementation
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from wcodec.bindings import Encoder, Decoder


def test_simple_roundtrip():
    """Test encode/decode on small random data"""
    print("\n[TEST 1] Simple 64x64 roundtrip")
    
    # Create test data
    np.random.seed(42)
    original = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
    
    # Encode
    encoder = Encoder(tile_size=16)
    encoded, encode_stats = encoder.encode_layer(original)
    
    print(f"  Original: {encode_stats['original_bytes']} bytes")
    print(f"  Compressed: {encode_stats['compressed_bytes']} bytes")
    print(f"  Ratio: {encode_stats['compression_ratio']:.2f}x")
    print(f"  Encode time: {encode_stats['encode_time_ms']:.2f}ms")
    
    # Decode
    decoder = Decoder(tile_size=16)
    decoded, decode_stats = decoder.decode_layer(encoded, 64, 64)
    
    print(f"  Decode time: {decode_stats['decode_time_ms']:.2f}ms")
    
    # Verify bit-exact
    if np.array_equal(original, decoded):
        print("  ✓ Bit-exact reconstruction!")
        return True
    else:
        diff = np.sum(original != decoded)
        print(f"  ✗ FAILED: {diff} mismatched values")
        return False


def test_various_sizes():
    """Test different matrix sizes"""
    print("\n[TEST 2] Various sizes")
    
    sizes = [(16, 16), (32, 32), (64, 128), (100, 200)]
    encoder = Encoder(tile_size=16)
    decoder = Decoder(tile_size=16)
    
    all_passed = True
    for rows, cols in sizes:
        np.random.seed(42)
        original = np.random.randint(-128, 127, (rows, cols), dtype=np.int8)
        
        encoded, stats = encoder.encode_layer(original)
        decoded, _ = decoder.decode_layer(encoded, rows, cols)
        
        passed = np.array_equal(original, decoded)
        status = "✓" if passed else "✗"
        print(f"  {status} {rows}x{cols}: {stats['compression_ratio']:.2f}x compression")
        
        all_passed = all_passed and passed
    
    return all_passed


def test_patterns():
    """Test on different data patterns"""
    print("\n[TEST 3] Data patterns")
    
    encoder = Encoder(tile_size=16)
    decoder = Decoder(tile_size=16)
    
    tests = [
        ("Zeros", np.zeros((64, 64), dtype=np.int8)),
        ("Ones", np.ones((64, 64), dtype=np.int8)),
        ("Constant", np.full((64, 64), 42, dtype=np.int8)),
        ("Random", np.random.randint(-128, 127, (64, 64), dtype=np.int8)),
        ("Structured", np.arange(64*64, dtype=np.int8).reshape(64, 64)),
    ]
    
    all_passed = True
    for name, data in tests:
        encoded, stats = encoder.encode_layer(data)
        decoded, _ = decoder.decode_layer(encoded, 64, 64)
        
        passed = np.array_equal(data, decoded)
        status = "✓" if passed else "✗"
        ratio = stats['compression_ratio']
        print(f"  {status} {name:12s}: {ratio:6.2f}x compression")
        
        all_passed = all_passed and passed
    
    return all_passed


def test_edge_cases():
    """Test edge cases"""
    print("\n[TEST 4] Edge cases")
    
    encoder = Encoder(tile_size=16)
    decoder = Decoder(tile_size=16)
    
    # Min/max values
    data = np.array([[-128, 127], [0, -1]], dtype=np.int8)
    encoded, _ = encoder.encode_layer(data)
    decoded, _ = decoder.decode_layer(encoded, 2, 2)
    
    if np.array_equal(data, decoded):
        print("  ✓ Min/max values")
        return True
    else:
        print("  ✗ Min/max values FAILED")
        return False


def main():
    print("="*60)
    print("Weight Codec - Week 2 Integration Tests")
    print("="*60)
    
    try:
        results = []
        
        results.append(test_simple_roundtrip())
        results.append(test_various_sizes())
        results.append(test_patterns())
        results.append(test_edge_cases())
        
        print("\n" + "="*60)
        if all(results):
            print("✓ ALL TESTS PASSED!")
            print("="*60)
            print("\nWeek 2 encoder/decoder working correctly!")
            print("Bit-exact reconstruction confirmed.")
            return 0
        else:
            print("✗ SOME TESTS FAILED")
            print("="*60)
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

