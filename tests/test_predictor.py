"""
Unit tests for predictive coding (Week 2)
"""

import pytest
import numpy as np


def test_predictor_placeholder():
    """
    Placeholder test - will test C++ predictor via Python bindings
    once pybind11 wrapper is added.
    """
    # For now, just test numpy operations that will be needed
    tile = np.random.randint(-128, 127, (16, 16), dtype=np.int8)
    left_col = np.random.randint(-128, 127, (16,), dtype=np.int8)
    top_row = np.random.randint(-128, 127, (16,), dtype=np.int8)
    
    # Simple prediction tests
    pred_left = np.tile(left_col.reshape(-1, 1), (1, 16))
    pred_top = np.tile(top_row.reshape(1, -1), (16, 1))
    pred_avg = ((pred_left.astype(np.int16) + pred_top.astype(np.int16)) // 2).astype(np.int8)
    
    # Residual
    residual = tile - pred_avg
    
    # Reconstruction
    reconstructed = residual + pred_avg
    
    # Should match exactly
    assert np.array_equal(tile, reconstructed)
    print("✓ Basic predictor math works")


def test_roundtrip_placeholder():
    """
    Test that encode -> decode gives bit-exact reconstruction.
    Will use C++ library once bindings are added.
    """
    # Generate test data
    data = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
    
    # TODO: Once C++ bindings work:
    # import wcodec
    # encoded = wcodec.encode_layer_numpy(data)
    # decoded = wcodec.decode_layer_numpy(encoded, 64, 64)
    # assert np.array_equal(data, decoded)
    
    print("✓ Roundtrip test placeholder ready")


if __name__ == "__main__":
    test_predictor_placeholder()
    test_roundtrip_placeholder()
    print("\nAll tests passed! (placeholders)")

