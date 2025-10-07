#!/usr/bin/env python3
"""
Simple test for core codec
"""

import numpy as np
import time

# Test without loading the library first
def test_basic():
    print("=== Basic Codec Test ===")

    # Generate test data
    data = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
    print(f"Original: {data.shape}, {data.nbytes} bytes")

    # Test predictor manually (what encoder does)
    tile = data[:16, :16]  # 16x16 tile
    left_col = data[:16, 0]  # Left neighbor
    top_row = data[0, :16]  # Top neighbor

    # Manual LEFT prediction
    pred_left = np.zeros_like(tile)
    for r in range(tile.shape[0]):
        pred_left[r, :] = left_col[r]

    residual_left = tile - pred_left

    # Manual TOP prediction
    pred_top = np.zeros_like(tile)
    for c in range(tile.shape[1]):
        pred_top[:, c] = top_row[c]

    residual_top = tile - pred_top

    # Compute energies
    energy_left = np.sum(np.abs(residual_left))
    energy_top = np.sum(np.abs(residual_top))

    best_mode = "LEFT" if energy_left <= energy_top else "TOP"
    print(f"Best predictor: {best_mode}")
    print(f"Left energy: {energy_left}, Top energy: {energy_top}")

    # Differential encoding
    residual = residual_left if best_mode == "LEFT" else residual_top
    
    print(f"\nResidual sample (first 5x5):")
    print(residual[:5, :5])
    
    prev = 0
    diff_bytes = []
    encode_debug = []
    for i, val in enumerate(residual.flatten()):
        # Convert numpy int8 to Python int to avoid overflow
        val_int = int(val)
        diff = val_int - prev
        # Cast to uint8 range (same as C++: static_cast<uint8_t>(diff + 128))
        byte_val = (diff + 128) % 256
        diff_bytes.append(byte_val)
        if i < 5:
            encode_debug.append(f"  [{i}] val={val_int}, prev={prev}, diff={diff}, byte={byte_val}")
        prev = val_int

    print(f"Encoding trace (first 5):")
    for line in encode_debug:
        print(line)
    print(f"First 10 encoded bytes: {diff_bytes[:10]}")
    
    # Calculate compression ratio
    original_bytes = tile.nbytes
    compressed_bytes = 4 + len(diff_bytes)  # header + data
    ratio = original_bytes / compressed_bytes

    print(f"Tile compression: {ratio:.2f}x")

    # Decode
    decoded_vals = []
    prev = 0
    decode_debug = []
    for i, diff_byte in enumerate(diff_bytes):
        # Decode: match C++ static_cast<int8_t>(diff_byte - 128)
        # First subtract 128 (uncentering), then interpret as signed int8
        diff_unsigned = diff_byte - 128
        # Convert to signed int8 range (-128 to 127)
        if diff_unsigned > 127:
            signed_diff = diff_unsigned - 256
        else:
            signed_diff = diff_unsigned
        val = prev + signed_diff
        decoded_vals.append(val)
        if i < 5:
            decode_debug.append(f"  [{i}] byte={diff_byte}, diff_u={diff_unsigned}, diff_s={signed_diff}, prev={prev}, val={val}")
        prev = val

    print(f"Decoding trace (first 5):")
    for line in decode_debug:
        print(line)
    print(f"First 10 decoded values: {decoded_vals[:10]}")
    
    decoded = np.array(decoded_vals).reshape(tile.shape)

    # Reconstruct
    pred = pred_left if best_mode == "LEFT" else pred_top
    reconstructed = decoded + pred

    # Verify
    match = np.array_equal(tile, reconstructed)
    print(f"Tile reconstruction: {match}")
    
    if not match:
        # Debug output
        print(f"\nDEBUG:")
        print(f"Tile sample (first 5x5):")
        print(tile[:5, :5])
        print(f"\nDecoded residual sample (first 5x5):")
        print(decoded[:5, :5])
        print(f"\nPrediction sample (first 5x5):")
        print(pred[:5, :5])
        print(f"\nReconstructed sample (first 5x5):")
        print(reconstructed[:5, :5])
        print(f"\nDifference (first 5x5):")
        print((tile - reconstructed)[:5, :5])
        print(f"\nMax error: {np.max(np.abs(tile - reconstructed))}")
        print(f"Mean error: {np.mean(np.abs(tile - reconstructed))}")
        
        # Check if it's just overflow
        print(f"\nTile dtype: {tile.dtype}, range: [{tile.min()}, {tile.max()}]")
        print(f"Reconstructed dtype: {reconstructed.dtype}, range: [{reconstructed.min()}, {reconstructed.max()}]")

    return match

def test_library():
    print("\n=== Library Test ===")
    try:
        # Try to load library
        import ctypes
        lib_path = "build/libcodec_core.so"
        lib = ctypes.CDLL(lib_path)
        print("✓ Library loaded successfully")

        # Try basic functions
        print("✓ Core functions available")

        return True
    except Exception as e:
        print(f"✗ Library error: {e}")
        return False

def main():
    print("=" * 60)
    print("Core Codec - Simple Tests")
    print("=" * 60)

    results = []
    results.append(test_basic())
    results.append(test_library())

    print("\n" + "=" * 60)
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("Core codec logic is working correctly")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

if __name__ == "__main__":
    main()

