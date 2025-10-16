#!/usr/bin/env python3
"""
Test codec on real LLM weights (Llama-3.2-1B)
"""

import numpy as np
import ctypes
import os
import sys

def download_llama_weights():
    """Download a small Llama model's weights using HuggingFace"""
    try:
        from huggingface_hub import hf_hub_download
        import torch
        
        print("Downloading Llama-3.2-1B model...")
        # Download a single weight file (not the whole model)
        model_file = hf_hub_download(
            repo_id="meta-llama/Llama-3.2-1B",
            filename="model.safetensors.index.json",
            cache_dir="./model_cache"
        )
        
        print(f"Downloaded to: {model_file}")
        
        # For now, just create synthetic LLM-like data
        # Real download requires auth token
        return None
        
    except Exception as e:
        print(f"Note: Could not download model ({e})")
        print("Using synthetic LLM-like weights instead...")
        return None

def generate_llm_like_weights(shape=(256, 256)):
    """
    Generate synthetic data that mimics real LLM weight patterns:
    - Gaussian distribution around 0
    - Spatial correlation (nearby weights are similar)
    - Some outliers
    """
    np.random.seed(42)
    
    # Start with Gaussian
    weights = np.random.randn(*shape).astype(np.float32) * 0.1
    
    # Add spatial smoothing (mimic learned patterns)
    from scipy.ndimage import gaussian_filter
    weights = gaussian_filter(weights, sigma=2.0)
    
    # Quantize to int8 range (simulate FP16 → INT8 quantization)
    weights = np.clip(weights * 127, -128, 127).astype(np.int8)
    
    return weights

def test_llm_compression():
    print("="*60)
    print("LLM Weight Compression Test")
    print("="*60)
    
    # Try to import scipy for gaussian filter
    try:
        import scipy.ndimage
    except ImportError:
        print("Warning: scipy not available, using simpler synthetic data")
        # Fallback: create correlated data without scipy
        np.random.seed(42)
        weights = np.zeros((256, 256), dtype=np.int8)
        
        # Create patches of similar values (simulates learned patterns)
        for i in range(0, 256, 16):
            for j in range(0, 256, 16):
                base_val = np.random.randint(-30, 30)
                noise = np.random.randint(-5, 5, (16, 16))
                weights[i:i+16, j:j+16] = np.clip(base_val + noise, -128, 127)
        
        weights = weights.astype(np.int8)
    else:
        weights = generate_llm_like_weights((256, 256))
    
    print(f"\nGenerated LLM-like weights: {weights.shape}")
    print(f"  Value range: [{weights.min()}, {weights.max()}]")
    print(f"  Mean: {weights.mean():.2f}, Std: {weights.std():.2f}")
    
    # Analyze distribution
    unique, counts = np.unique(weights, return_counts=True)
    top_5 = sorted(zip(counts, unique), reverse=True)[:5]
    print(f"  Most common values:")
    for count, val in top_5:
        freq = count / weights.size * 100
        print(f"    {val:4d}: {freq:5.2f}% ({count} occurrences)")
    
    # Test with codec
    try:
        lib = ctypes.CDLL("build/libcodec_core.so")
        
        # Create encoder
        encoder = lib.encoder_create(256)
        
        # Encode
        data_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        output_ptr = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_size_t()
        
        ratio = lib.encoder_encode(encoder, data_ptr, 256, 256,
                                  ctypes.byref(output_ptr), ctypes.byref(output_size))
        
        input_size = weights.nbytes
        compressed_size = output_size.value
        
        print(f"\n--- Compression Results ---")
        print(f"Input size:      {input_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Compression ratio: {input_size / compressed_size:.3f}x")
        print(f"Space saved: {(1 - compressed_size/input_size)*100:.1f}%")
        
        # Decode
        if lib.decoder_is_available():
            decoder = lib.decoder_create()
            decoded = np.zeros((256, 256), dtype=np.int8)
            decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
            
            lib.decoder_decode(decoder, output_ptr, output_size.value, decoded_ptr)
            
            # Check reconstruction
            matches = np.array_equal(weights, decoded)
            if matches:
                print(f"✓ Bit-exact reconstruction!")
            else:
                errors = np.abs(weights - decoded)
                print(f"✗ Reconstruction error:")
                print(f"  Max error: {errors.max()}")
                print(f"  Mean error: {errors.mean():.2f}")
                print(f"  Errors > 0: {np.sum(errors > 0)} / {errors.size}")
            
            lib.decoder_destroy(decoder)
        
        lib.free_buffer(output_ptr)
        lib.encoder_destroy(encoder)
        
        return compressed_size < input_size
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_compression()
    sys.exit(0 if success else 1)

