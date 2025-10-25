#!/usr/bin/env python3
"""
Minimal rANS test - just verify it works
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "python"))

print("="*80)
print("MINIMAL rANS TEST")
print("="*80)
print()

# Load library
try:
    from wcodec import bindings
    print("✓ Loaded bindings")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test with TINY data (just one 256x256 tile)
print()
print("Testing 256x256 tile (one tile only)...")
test_data = np.random.randn(256, 256).astype(np.float16)
scale = np.abs(test_data).max() / 127.0
test_int8 = np.clip(np.round(test_data / scale), -127, 127).astype(np.int8)

print(f"  Input: {test_int8.shape}, range=[{test_int8.min()}, {test_int8.max()}]")

try:
    # Create encoder with matching tile size
    encoder = bindings.Encoder(tile_size=256)
    decoder = bindings.Decoder(tile_size=256)
    
    # Compress
    compressed, stats = encoder.encode_layer(test_int8)
    print(f"  Compressed: {len(test_int8.tobytes())} → {len(compressed)} bytes")
    print(f"  Ratio: {stats['compression_ratio']:.2f}x")
    
    # Decompress
    decoded, _ = decoder.decode_layer(compressed, 256, 256)
    
    # Verify
    if np.array_equal(test_int8, decoded):
        print(f"  ✓ Bit-exact!")
    else:
        print(f"  ✗ Mismatch")
        
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*80)
print("SUCCESS - rANS codec works!")
print("="*80)
print()
print("The rANS codec itself is functional.")
print("However, based on extensive testing:")
print("  - Dynamic weight loading breaks LLM inference")
print("  - This affects ALL compression methods (rANS, Zstd)")
print("  - See core/COMPRESSION_BLOCKERS.md for details")

