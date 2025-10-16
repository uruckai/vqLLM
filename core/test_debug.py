#!/usr/bin/env python3
"""
Debug test to isolate the exact crash point
"""

import numpy as np
import ctypes
import sys
from pathlib import Path

def load_codec():
    """Load the codec library"""
    lib_path = Path("build/libcodec_core.so")
    if not lib_path.exists():
        print(f"✗ Codec library not found at {lib_path}")
        return None
    
    lib = ctypes.CDLL(str(lib_path))
    lib.encoder_create.restype = ctypes.c_void_p
    lib.decoder_create.restype = ctypes.c_void_p
    lib.decoder_is_available.restype = ctypes.c_bool
    lib.encoder_encode.restype = ctypes.c_float
    
    return lib

def test_problematic_pattern():
    """Test the exact pattern that causes segfault"""
    
    print("="*80)
    print("DEBUG TEST: Problematic Data Pattern")
    print("="*80)
    
    lib = load_codec()
    if lib is None:
        return False
    
    # Create data pattern similar to Layer 2 (highly concentrated around 0, ±1, ±2)
    # This mimics the distribution that causes the crash
    print("\nCreating data with concentrated distribution...")
    
    rng = np.random.RandomState(42)
    
    # Create distribution similar to problematic layer:
    # 10% each for 0, -1, 1, -2, 2
    # Rest distributed randomly
    data = np.zeros((256, 256), dtype=np.int8)
    
    choices = [0, -1, 1, -2, 2]
    probs = [0.105, 0.10, 0.10, 0.09, 0.09]  # Total 58.5%
    probs.append(1.0 - sum(probs))  # Remaining 41.5%
    choices.append(-128)  # Placeholder for "other" values
    
    for i in range(256):
        for j in range(256):
            val_idx = rng.choice(len(choices), p=probs)
            if choices[val_idx] == -128:
                # Random other value
                data[i, j] = rng.randint(-90, 128, dtype=np.int8)
            else:
                data[i, j] = choices[val_idx]
    
    print(f"  Data stats:")
    print(f"    Range: [{data.min()}, {data.max()}]")
    print(f"    Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    
    unique, counts = np.unique(data, return_counts=True)
    top_5 = sorted(zip(counts, unique), reverse=True)[:5]
    print(f"    Most common values:")
    for count, val in top_5:
        freq = count / data.size * 100
        print(f"      {val:4d}: {freq:5.2f}% ({count} occurrences)")
    
    # Test encode
    print("\n[1/3] Creating encoder...")
    encoder = lib.encoder_create(256)
    print("✓ Encoder created")
    
    print("\n[2/3] Encoding...")
    sys.stdout.flush()
    
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    output_ptr = ctypes.POINTER(ctypes.c_uint8)()
    output_size = ctypes.c_size_t()
    
    try:
        ratio = lib.encoder_encode(encoder, data_ptr, 256, 256,
                                  ctypes.byref(output_ptr), ctypes.byref(output_size))
        print(f"✓ Encoding succeeded")
        print(f"  Input: {data.nbytes} bytes")
        print(f"  Compressed: {output_size.value} bytes")
        print(f"  Ratio: {data.nbytes / output_size.value:.2f}x")
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        return False
    
    print("\n[3/3] Decoding...")
    sys.stdout.flush()
    
    if not lib.decoder_is_available():
        print("✗ GPU decoder not available")
        return False
    
    print("  Creating decoder...")
    sys.stdout.flush()
    
    try:
        decoder = lib.decoder_create()
        print("  ✓ Decoder created")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Failed to create decoder: {e}")
        return False
    
    decoded = np.zeros((256, 256), dtype=np.int8)
    decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    
    print("  Calling decoder_decode...")
    sys.stdout.flush()
    
    try:
        lib.decoder_decode(decoder, output_ptr, output_size.value, decoded_ptr)
        print("  ✓ Decoding succeeded!")
    except Exception as e:
        print(f"  ✗ Decoding failed: {e}")
        lib.free_buffer(output_ptr)
        lib.encoder_destroy(encoder)
        lib.decoder_destroy(decoder)
        return False
    
    # Verify
    print("\nVerifying reconstruction...")
    if np.array_equal(data, decoded):
        print("✅ SUCCESS! Bit-exact reconstruction")
        matches = True
    else:
        errors = np.abs(data.astype(np.int32) - decoded.astype(np.int32))
        print(f"✗ Reconstruction error:")
        print(f"  Max error: {errors.max()}")
        print(f"  Mean error: {errors.mean():.2f}")
        print(f"  Num errors: {np.sum(errors > 0)}")
        matches = False
    
    # Cleanup
    lib.free_buffer(output_ptr)
    lib.encoder_destroy(encoder)
    lib.decoder_destroy(decoder)
    
    return matches

if __name__ == "__main__":
    print("\nThis test reproduces the exact data pattern that causes the segfault.")
    print("If it crashes, we know the issue is in the codec.")
    print("If it works, the issue is in how the Python test script prepares data.\n")
    
    success = test_problematic_pattern()
    sys.exit(0 if success else 1)

