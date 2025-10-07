#!/usr/bin/env python3
"""
End-to-end integration tests
"""

import sys
import numpy as np
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from wcodec.bindings import Encoder, Decoder
from wcodec.encoder_api import encode_layer_standalone
from wcodec.decoder_api import decode_layer_standalone


def test_single_layer_roundtrip():
    """Test encode → write → read → decode pipeline"""
    print("\n" + "="*60)
    print("[TEST 1] Single Layer Roundtrip")
    print("="*60)
    
    # Create test data
    np.random.seed(42)
    original = np.random.randint(-128, 127, (256, 256), dtype=np.int8)
    
    print(f"\nOriginal: {original.shape}, {original.nbytes} bytes")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        compressed_path = Path(tmpdir) / "layer.wcodec"
        
        # Encode and write
        print(f"Encoding to {compressed_path.name}...")
        stats = encode_layer_standalone(original, compressed_path, tile_size=16)
        
        print(f"  Compressed: {stats['compressed_bytes']} bytes")
        print(f"  Ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Time: {stats['encode_time_ms']:.1f}ms")
        
        # Read and decode
        print("Decoding...")
        decoded = decode_layer_standalone(compressed_path, 256, 256, tile_size=16)
        
        # Verify
        if np.array_equal(original, decoded):
            print("✓ Bit-exact reconstruction!")
            return True
        else:
            diff = np.sum(original != decoded)
            print(f"✗ FAILED: {diff} mismatches")
            return False


def test_multiple_layers():
    """Test encoding multiple layers"""
    print("\n" + "="*60)
    print("[TEST 2] Multiple Layers")
    print("="*60)
    
    layer_sizes = [
        (64, 64),
        (128, 128),
        (256, 256),
    ]
    
    encoder = Encoder(tile_size=16)
    decoder = Decoder(tile_size=16)
    
    all_passed = True
    total_original = 0
    total_compressed = 0
    
    for i, (rows, cols) in enumerate(layer_sizes):
        np.random.seed(i)
        original = np.random.randint(-128, 127, (rows, cols), dtype=np.int8)
        
        # Encode
        compressed, stats = encoder.encode_layer(original)
        
        # Decode
        decoded, _ = decoder.decode_layer(compressed, rows, cols)
        
        # Verify
        passed = np.array_equal(original, decoded)
        status = "✓" if passed else "✗"
        
        print(f"  {status} Layer {i}: {rows}×{cols} → "
              f"{stats['compression_ratio']:.2f}x compression")
        
        all_passed = all_passed and passed
        total_original += stats['original_bytes']
        total_compressed += stats['compressed_bytes']
    
    overall_ratio = total_original / max(1, total_compressed)
    print(f"\nOverall: {overall_ratio:.2f}x compression")
    
    return all_passed


def test_large_layer():
    """Test realistic checkpoint layer size"""
    print("\n" + "="*60)
    print("[TEST 3] Large Layer (1024×1024)")
    print("="*60)
    
    np.random.seed(42)
    original = np.random.randint(-128, 127, (1024, 1024), dtype=np.int8)
    
    print(f"Size: {original.nbytes / (1024**2):.1f} MB")
    
    encoder = Encoder(tile_size=16)
    decoder = Decoder(tile_size=16)
    
    # Encode
    print("Encoding...", end=" ", flush=True)
    compressed, stats = encoder.encode_layer(original)
    print(f"{stats['encode_time_ms']:.0f}ms")
    
    print(f"  Original: {stats['original_bytes'] / (1024**2):.2f} MB")
    print(f"  Compressed: {stats['compressed_bytes'] / (1024**2):.2f} MB")
    print(f"  Ratio: {stats['compression_ratio']:.2f}x")
    
    # Decode
    print("Decoding...", end=" ", flush=True)
    decoded, decode_stats = decoder.decode_layer(compressed, 1024, 1024)
    print(f"{decode_stats['decode_time_ms']:.0f}ms")
    
    # Verify
    if np.array_equal(original, decoded):
        print("✓ Bit-exact reconstruction!")
        return True
    else:
        diff = np.sum(original != decoded)
        print(f"✗ FAILED: {diff} mismatches")
        return False


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*60)
    print("[TEST 4] Edge Cases")
    print("="*60)
    
    encoder = Encoder(tile_size=16)
    decoder = Decoder(tile_size=16)
    
    test_cases = [
        ("All zeros", np.zeros((64, 64), dtype=np.int8)),
        ("All ones", np.ones((64, 64), dtype=np.int8)),
        ("Max values", np.full((64, 64), 127, dtype=np.int8)),
        ("Min values", np.full((64, 64), -128, dtype=np.int8)),
        ("Checkerboard", np.array([[(-1)**((i+j)%2)*64 for j in range(64)] for i in range(64)], dtype=np.int8)),
    ]
    
    all_passed = True
    
    for name, data in test_cases:
        compressed, stats = encoder.encode_layer(data)
        decoded, _ = decoder.decode_layer(compressed, 64, 64)
        
        passed = np.array_equal(data, decoded)
        status = "✓" if passed else "✗"
        
        print(f"  {status} {name:15s}: {stats['compression_ratio']:6.2f}x")
        
        all_passed = all_passed and passed
    
    return all_passed


def main():
    print("="*60)
    print("End-to-End Integration Tests")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(test_single_layer_roundtrip())
    results.append(test_multiple_layers())
    results.append(test_large_layer())
    results.append(test_edge_cases())
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("\nWeek 5 Status:")
        print("  ✓ Low-level encode/decode working")
        print("  ✓ File I/O working")
        print("  ✓ Bit-exact reconstruction verified")
        print("  ⚠ Container format: partial (full integration pending)")
        print("  ⚠ GPU decode: infrastructure ready (wiring pending)")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        failed = sum(1 for r in results if not r)
        print(f"  {failed}/{len(results)} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

