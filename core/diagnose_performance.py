#!/usr/bin/env python3
"""
Performance Diagnostic Script

Analyzes where time is being spent in the inference pipeline
"""

import numpy as np
import ctypes
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn

# Import from existing test
from test_inference_lowmem import load_codec

def time_operation(name, func):
    """Time an operation and print results"""
    start = time.time()
    result = func()
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.3f}s")
    return result, elapsed

def analyze_layer_sizes():
    """Analyze typical layer sizes to understand tiling strategy"""
    print("Analyzing layer sizes...")

    # Load a small model to get layer info
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to('cpu')

        print(f"Model has {len(list(model.modules()))} modules")

        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module.weight.shape))
                print(f"  {name}: {module.weight.shape} = {module.weight.numel() * module.weight.element_size() / 1024:.1f} KB")

        # Analyze optimal tile sizes
        print("\nOptimal tile size analysis:")
        for name, shape in linear_layers[:5]:  # Just first 5
            rows, cols = shape
            elements = rows * cols

            # Test different tile sizes
            for tile_size in [256, 512, 1024, 2048]:
                tile_elements = tile_size * tile_size
                num_tiles = (elements + tile_elements - 1) // tile_elements
                print(f"  {name} ({rows}x{cols}): tile_size={tile_size}")
                print(f"    Elements: {elements","} -> {num_tiles} tiles")
                print(f"    Overhead: {num_tiles * tile_elements - elements} wasted elements")

        return linear_layers

    except Exception as e:
        print(f"Could not load model for analysis: {e}")
        return []

def test_codec_performance():
    """Test raw codec performance on synthetic data"""
    print("\nTesting raw codec performance...")

    # Load codec
    lib = load_codec()
    if lib is None:
        print("Could not load codec")
        return

    # Test different data sizes and tile sizes
    test_sizes = [
        (512, 512),    # Small layer
        (2048, 2048),  # Medium layer
        (4096, 4096),  # Large layer
    ]

    tile_sizes = [256, 512, 1024, 2048]

    for rows, cols in test_sizes:
        print(f"\nTesting {rows}x{cols} layer:")

        # Create test data
        data = np.random.randn(rows, cols).astype(np.float32)
        scale = np.abs(data).max() / 127.0
        quantized = np.round(data / scale).astype(np.int8)

        for tile_size in tile_sizes:
            # Skip if tile too big for data
            if tile_size > min(rows, cols):
                continue

            tile_elements = tile_size * tile_size
            num_tiles = (rows * cols + tile_elements - 1) // tile_elements

            # Time compression
            def compress_test():
                encoder = lib.encoder_create(tile_size)
                compressed_tiles = []

                flat = quantized.flatten()
                padded_size = num_tiles * tile_elements
                padded = np.zeros(padded_size, dtype=np.int8)
                padded[:len(flat)] = flat

                for i in range(num_tiles):
                    tile_data = padded[i*tile_elements:(i+1)*tile_elements]
                    tile = tile_data.reshape(tile_size, tile_size)
                    tile_contig = np.ascontiguousarray(tile)

                    compressed = ctypes.POINTER(ctypes.c_uint8)()
                    compressed_size = ctypes.c_size_t()

                    lib.encoder_encode(
                        encoder,
                        tile_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        tile_size, tile_size,
                        ctypes.byref(compressed),
                        ctypes.byref(compressed_size)
                    )

                    compressed_bytes = bytes(compressed[:compressed_size.value])
                    lib.free_buffer(compressed)
                    compressed_tiles.append(compressed_bytes)

                lib.encoder_destroy(encoder)
                return compressed_tiles

            compressed_tiles, comp_time = time_operation(f"  Compress {tile_size}x{tile_size}", compress_test)

            # Time decompression
            def decompress_test():
                decoder = lib.decoder_create()
                all_data = []

                for compressed in compressed_tiles:
                    decoded = np.zeros(tile_elements, dtype=np.int8)
                    decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))

                    compressed_array = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
                    compressed_ptr = ctypes.cast(compressed_array, ctypes.POINTER(ctypes.c_uint8))

                    lib.decoder_decode(decoder, compressed_ptr, len(compressed), decoded_ptr)
                    all_data.append(decoded)

                lib.decoder_destroy(decoder)

                # Concatenate
                full_data = np.concatenate(all_data)[:rows * cols]
                return full_data.reshape(rows, cols)

            decompressed, decomp_time = time_operation(f"  Decompress {tile_size}x{tile_size}", decompress_test)

            # Verify correctness
            original_padded = quantized.flatten()[:len(decompressed.flatten())]
            matches = np.array_equal(original_padded, decompressed.flatten())

            total_time = comp_time + decomp_time
            ratio = (rows * cols) / sum(len(t) for t in compressed_tiles)

            print(f"    Total: {total_time:.3f}s ({comp_time:.3f}s + {decomp_time:.3f}s)")
            print(f"    Compression ratio: {ratio:.2f}x")
            print(f"    Correct: {'✓' if matches else '✗'}")
            print(f"    Tiles: {num_tiles}")

def main():
    print("="*80)
    print("PERFORMANCE DIAGNOSTIC")
    print("="*80)

    # Analyze layer sizes
    layers = analyze_layer_sizes()

    # Test codec performance
    test_codec_performance()

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

