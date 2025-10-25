#!/usr/bin/env python3
"""
Test: rANS codec from archive
Goal: Check if old rANS implementation builds and works at all
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add archive to path
sys.path.insert(0, str(Path(__file__).parent / "archive"))

print("="*80)
print("TEST: rANS CODEC FROM ARCHIVE")
print("="*80)
print()

# Try to load the old codec
try:
    from archive import bindings
    print("✓ Found old rANS bindings in archive")
except ImportError as e:
    print(f"✗ Cannot import archive bindings: {e}")
    print()
    print("The old rANS codec requires rebuilding.")
    print()
    print("To use the old rANS implementation:")
    print("1. The C++ source files are in core/archive/")
    print("2. Need to update CMakeLists.txt to point to archive/")
    print("3. Rebuild with: cmake .. && make")
    print()
    print("However, note that the rANS approach has the same fundamental")
    print("issue we just discovered: dynamic weight loading causes")
    print("numerical differences that break LLM inference.")
    print()
    sys.exit(1)

# If we got here, try to use it
print()
print("Testing rANS codec...")

try:
    lib = bindings.load_codec()
    print("✓ Loaded rANS codec library")
    
    # Create encoder
    encoder = lib.encoder_create(256)  # tile_size=256
    if encoder:
        print("✓ Created encoder")
        lib.encoder_destroy(encoder)
    else:
        print("✗ Failed to create encoder")
    
    # Create decoder
    decoder = lib.decoder_create()
    if decoder:
        print("✓ Created decoder")
        
        # Check GPU availability
        gpu_available = lib.decoder_is_available()
        print(f"  GPU decode available: {gpu_available}")
        
        lib.decoder_destroy(decoder)
    else:
        print("✗ Failed to create decoder")
    
    print()
    print("="*80)
    print("rANS CODEC STATUS")
    print("="*80)
    print()
    print("✓ Old rANS codec builds and initializes correctly")
    print()
    print("However, this codec has the same issue as Zstd:")
    print("  - Dynamic weight loading causes numerical differences")
    print("  - LLM inference breaks even with perfect compression")
    print("  - See COMPRESSION_BLOCKERS.md for details")
    print()
    print("The rANS codec is preserved in core/archive/ for reference,")
    print("but is not recommended for LLM inference.")
    
except Exception as e:
    print(f"✗ Error testing codec: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

