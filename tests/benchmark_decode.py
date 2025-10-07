#!/usr/bin/env python3
"""
Decode performance benchmark
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from wcodec.bindings import Encoder, Decoder


def benchmark_layer_size(rows, cols, num_runs=5):
    """Benchmark a specific layer size"""
    # Create test data
    np.random.seed(42)
    data = np.random.randint(-128, 127, (rows, cols), dtype=np.int8)
    
    # Encode once
    encoder = Encoder(tile_size=16)
    compressed, encode_stats = encoder.encode_layer(data)
    
    # Decode multiple times
    decoder = Decoder(tile_size=16)
    
    decode_times = []
    for _ in range(num_runs):
        start = time.time()
        decoded, _ = decoder.decode_layer(compressed, rows, cols)
        decode_times.append(time.time() - start)
    
    # Verify correctness
    decoded_final, _ = decoder.decode_layer(compressed, rows, cols)
    correct = np.array_equal(data, decoded_final)
    
    avg_time = np.mean(decode_times)
    std_time = np.std(decode_times)
    throughput = (rows * cols) / (1024**2) / avg_time  # MB/s
    
    return {
        'rows': rows,
        'cols': cols,
        'size_mb': (rows * cols) / (1024**2),
        'compressed_mb': len(compressed) / (1024**2),
        'compression_ratio': encode_stats['compression_ratio'],
        'avg_decode_time': avg_time,
        'std_decode_time': std_time,
        'throughput_mbps': throughput,
        'correct': correct
    }


def main():
    print("="*80)
    print("Decode Performance Benchmark (CPU)")
    print("="*80)
    
    # Test various layer sizes
    sizes = [
        (256, 256),      # Small layer
        (512, 512),      # Medium layer
        (1024, 1024),    # Large layer
        (2048, 2048),    # Very large layer
        (4096, 4096),    # Checkpoint layer
        (4096, 11008),   # MLP layer (Llama-style)
    ]
    
    print("\nBenchmarking different layer sizes...")
    print("-"*80)
    print(f"{'Size':<15} {'MB':<8} {'Comp':<8} {'Time (s)':<12} {'Throughput':<15} {'Status'}")
    print("-"*80)
    
    all_results = []
    
    for rows, cols in sizes:
        result = benchmark_layer_size(rows, cols, num_runs=3)
        all_results.append(result)
        
        status = "✓" if result['correct'] else "✗"
        
        print(f"{rows}×{cols:<10} "
              f"{result['size_mb']:6.2f}  "
              f"{result['compression_ratio']:6.2f}x  "
              f"{result['avg_decode_time']:6.3f} ± {result['std_decode_time']:5.3f}  "
              f"{result['throughput_mbps']:8.2f} MB/s  "
              f"{status}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_mb = sum(r['size_mb'] for r in all_results)
    total_compressed_mb = sum(r['compressed_mb'] for r in all_results)
    avg_ratio = np.mean([r['compression_ratio'] for r in all_results])
    avg_throughput = np.mean([r['throughput_mbps'] for r in all_results])
    
    print(f"\nTotal data processed: {total_mb:.1f} MB")
    print(f"Total compressed: {total_compressed_mb:.1f} MB")
    print(f"Average compression: {avg_ratio:.2f}x")
    print(f"Average throughput: {avg_throughput:.2f} MB/s (CPU)")
    
    # Estimate full model decode time
    print("\n" + "-"*80)
    print("Estimated Full Model Decode Times (CPU):")
    print("-"*80)
    
    model_sizes = {
        "Llama-2-7B": 7 * 1024,      # 7 GB
        "Llama-2-13B": 13 * 1024,    # 13 GB
        "Llama-2-70B": 70 * 1024,    # 70 GB
    }
    
    for model_name, size_mb in model_sizes.items():
        decode_time = size_mb / avg_throughput
        print(f"  {model_name:<20} {size_mb/1024:5.0f} GB  →  {decode_time/60:6.1f} minutes")
    
    print("\n" + "="*80)
    print("Next Step: GPU Acceleration")
    print("="*80)
    print("\nTarget with GPU (Week 4 goal):")
    print(f"  Average throughput: ~500 MB/s (100x faster)")
    print(f"  Llama-2-7B decode:  < 60 seconds")
    print(f"  Llama-2-70B decode: < 10 minutes")
    print("\nThis demonstrates why GPU acceleration is critical! 🚀")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

