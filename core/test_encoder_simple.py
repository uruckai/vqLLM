#!/usr/bin/env python3
"""
Simplest possible test - just create encoder and check if it's valid
"""
import ctypes

lib = ctypes.CDLL('./build/libcodec_core.so')

# Set argtypes FIRST
lib.batched_encoder_create.argtypes = [ctypes.c_uint16]
lib.batched_encoder_create.restype = ctypes.c_void_p

print("Creating encoder with tile_size=256...")
encoder = lib.batched_encoder_create(256)

if encoder:
    print(f"✓ Encoder created: {hex(encoder)}")
else:
    print("✗ Failed to create encoder")

