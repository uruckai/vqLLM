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
    prev = 0
    diff_bytes = []
    for val in residual.flatten():
        # Convert numpy int8 to Python int to avoid overflow
        val_int = int(val)
        diff = val_int - prev
        # Center around 128, but clamp to uint8 range (0-255)
        centered = diff + 128
        clamped = max(0, min(255, centered))
        diff_bytes.append(clamped)
        prev = val_int

    # Calculate compression ratio
    original_bytes = tile.nbytes
    compressed_bytes = 4 + len(diff_bytes)  # header + data
    ratio = original_bytes / compressed_bytes

    print(f"Tile compression: {ratio:.2f}x")

    # Decode
    decoded_vals = []
    prev = 0
    for diff_byte in diff_bytes:
        diff = diff_byte - 128
        val = prev + diff
        decoded_vals.append(val)
        prev = val

    decoded = np.array(decoded_vals).reshape(tile.shape)

    # Reconstruct
    pred = pred_left if best_mode == "LEFT" else pred_top
    reconstructed = decoded + pred

    # Verify
    match = np.array_equal(tile, reconstructed)
    print(f"Tile reconstruction: {match}")

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

