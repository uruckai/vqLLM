#!/usr/bin/env python3
"""
Test GPU-accelerated decode (with CPU fallback)
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from wcodec.bindings import Encoder, Decoder, GPUDecoder, is_gpu_available


def test_gpu_availability():
    """Test GPU detection"""
    print("\n" + "="*60)
    print("[TEST 1] GPU Availability")
    print("="*60)
    
    available = is_gpu_available()
    
    if available:
        print("✓ GPU available for decode")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        except:
            pass
    else:
        print("⚠ GPU not available - will use CPU fallback")
    
    return True


def test_gpu_decode_small():
    """Test GPU decode on small data"""
    print("\n" + "="*60)
    print("[TEST 2] GPU Decode (64×64)")
    print("="*60)
    
    # Create test data
    np.random.seed(42)
    original = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
    
    # Encode with CPU
    print("Encoding...")
    encoder = Encoder(tile_size=16)
    compressed, encode_stats = encoder.encode_layer(original)
    
    print(f"  Compressed: {len(compressed)} bytes ({encode_stats['compression_ratio']:.2f}x)")
    
    # Decode with GPU
    print("Decoding with GPU decoder...")
    gpu_decoder = GPUDecoder(tile_size=16)
    decoded, decode_stats = gpu_decoder.decode_layer(compressed, 64, 64)
    
    print(f"  Device: {decode_stats['device']}")
    print(f"  Time: {decode_stats['decode_time_ms']:.2f}ms")
    
    # Verify
    if np.array_equal(original, decoded):
        print("✓ Bit-exact reconstruction!")
        return True
    else:
        diff = np.sum(original != decoded)
        print(f"✗ FAILED: {diff} mismatches")
        return False


def test_gpu_vs_cpu_speed():
    """Compare GPU vs CPU decode speed"""
    print("\n" + "="*60)
    print("[TEST 3] GPU vs CPU Speed Comparison")
    print("="*60)
    
    sizes = [(64, 64), (128, 128), (256, 256)]
    
    for rows, cols in sizes:
        np.random.seed(42)
        original = np.random.randint(-128, 127, (rows, cols), dtype=np.int8)
        
        # Encode
        encoder = Encoder(tile_size=16)
        compressed, _ = encoder.encode_layer(original)
        
        # CPU decode
        cpu_decoder = Decoder(tile_size=16)
        start = time.time()
        decoded_cpu, _ = cpu_decoder.decode_layer(compressed, rows, cols)
        cpu_time = (time.time() - start) * 1000  # ms
        
        # GPU decode
        gpu_decoder = GPUDecoder(tile_size=16)
        decoded_gpu, gpu_stats = gpu_decoder.decode_layer(compressed, rows, cols)
        gpu_time = gpu_stats['decode_time_ms']
        
        # Verify both match
        cpu_match = np.array_equal(original, decoded_cpu)
        gpu_match = np.array_equal(original, decoded_gpu)
        
        if cpu_match and gpu_match:
            speedup = cpu_time / max(gpu_time, 0.001)
            status = "✓"
            device = gpu_stats['device']
            print(f"  {status} {rows}×{cols}: CPU {cpu_time:.1f}ms, {device} {gpu_time:.1f}ms "
                  f"({speedup:.1f}x)")
        else:
            print(f"  ✗ {rows}×{cols}: Decode mismatch!")
            return False
    
    return True


def test_gpu_decode_large():
    """Test GPU decode on larger data"""
    print("\n" + "="*60)
    print("[TEST 4] GPU Decode (512×512)")
    print("="*60)
    
    np.random.seed(42)
    original = np.random.randint(-128, 127, (512, 512), dtype=np.int8)
    
    print(f"Size: {original.nbytes / 1024:.1f} KB")
    
    # Encode
    encoder = Encoder(tile_size=16)
    compressed, encode_stats = encoder.encode_layer(original)
    
    print(f"Compressed: {len(compressed) / 1024:.1f} KB ({encode_stats['compression_ratio']:.2f}x)")
    
    # GPU decode
    print("Decoding...")
    gpu_decoder = GPUDecoder(tile_size=16)
    decoded, decode_stats = gpu_decoder.decode_layer(compressed, 512, 512)
    
    print(f"  Device: {decode_stats['device']}")
    print(f"  Time: {decode_stats['decode_time_ms']:.2f}ms")
    
    throughput = (original.nbytes / 1024) / (decode_stats['decode_time_ms'] / 1000)  # KB/s
    print(f"  Throughput: {throughput:.1f} KB/s")
    
    # Verify
    if np.array_equal(original, decoded):
        print("✓ Bit-exact reconstruction!")
        return True
    else:
        print("✗ FAILED")
        return False


def main():
    print("="*60)
    print("GPU-Accelerated Decode Tests")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(test_gpu_availability())
    results.append(test_gpu_decode_small())
    results.append(test_gpu_vs_cpu_speed())
    results.append(test_gpu_decode_large())
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("\nStatus:")
        print("  ✓ GPU decoder API working")
        print("  ✓ CPU fallback working")
        print("  ✓ Bit-exact reconstruction")
        
        if is_gpu_available():
            print("  ✓ GPU acceleration available")
        else:
            print("  ⚠ GPU not available (using CPU fallback)")
        
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

