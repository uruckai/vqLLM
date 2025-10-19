#!/usr/bin/env python3
"""
Test batched codec with LLM-like data (not random!)
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from bindings_batched import BatchedEncoder, BatchedGPUDecoder

print("=== Batched Codec Test (LLM-like Data) ===")

# Generate LLM-like data (small values, correlated)
# Real LLM weights after quantization have:
# - Most values near 0
# - High correlation (neighboring weights are similar)
# - Range typically -100 to +100

np.random.seed(42)
# Generate correlated data (like LLM weights)
# Method: Start with smooth base + small noise
base = np.zeros((2048, 2048), dtype=np.float32)
for i in range(0, 2048, 64):
    for j in range(0, 2048, 64):
        # Each 64x64 block has a random base value
        base_val = np.random.randn() * 0.3
        base[i:i+64, j:j+64] = base_val + np.random.randn(64, 64) * 0.1

# Quantize to int8
data = np.clip(base * 100, -127, 127).astype(np.int8)

print(f"Original: {data.shape}, {data.nbytes} bytes")
print(f"Value range: [{data.min()}, {data.max()}]")
print(f"Mean: {data.mean():.2f}, Std: {data.std():.2f}")

# Histogram
hist, bins = np.histogram(data, bins=20)
print(f"Distribution (histogram):")
for i in range(len(hist)):
    bar = '█' * int(hist[i] / hist.max() * 40)
    print(f"  [{bins[i]:4.0f}, {bins[i+1]:4.0f}): {bar} {hist[i]}")

# Encode
encoder = BatchedEncoder(tile_size=256)
compressed, ratio = encoder.encode_layer(data)
print(f"\nCompressed: {len(compressed)} bytes")
print(f"Ratio: {ratio:.2f}x")

# Decode
if BatchedGPUDecoder.is_available():
    decoder = BatchedGPUDecoder()
    decoded, time_ms = decoder.decode_layer(compressed, 2048, 2048)
    print(f"Decode time: {time_ms:.2f} ms ({time_ms/64:.2f} ms/tile for 64 tiles)")
    print(f"Bit-exact: {np.array_equal(data, decoded)}")
    
    if not np.array_equal(data, decoded):
        diff = np.sum(data != decoded)
        print(f"  Errors: {diff} / {data.size}")
else:
    print("GPU not available, skipping decode test")

print("\n" + "="*50)
print("Compression effectiveness:")
print(f"  Original size:    {data.nbytes:>10,} bytes")
print(f"  Compressed size:  {len(compressed):>10,} bytes")
print(f"  Savings:          {data.nbytes - len(compressed):>10,} bytes ({(1 - len(compressed)/data.nbytes)*100:.1f}%)")
print(f"  Compression ratio: {ratio:.2f}x")
print("="*50)

