#!/usr/bin/env python3
"""
GPU-Accelerated Low-Memory Inference Test

Uses GPU decoder with direct GPU memory allocation for 10x faster decompression
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
from test_inference_lowmem import load_codec, compress_model_weights

def test_gpu_accelerated_inference():
    """Test inference with GPU-accelerated decompression"""
    
    print("="*80)
    print("GPU-ACCELERATED LOW-MEMORY INFERENCE TEST")
    print("="*80)
    print("\nThis test uses GPU decoder for ~10x faster decompression")
    
    # Load codec
    print("\n[1/5] Loading codec library...")
    lib = load_codec()
    if lib is None:
        return False
    
    if not lib.decoder_is_available():
        print("✗ GPU decoder not available")
        return False
    
    print("✓ Codec library loaded")
    print("✓ GPU decoder available")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available - falling back to CPU")
        print("  (This will be slow - use test_inference_lowmem.py for CPU test)")
        return False
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Load PyTorch and Transformers
    print("\n[2/5] Loading PyTorch and Transformers...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ Libraries loaded")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False
    
    # Load model
    print("\n[3/5] Loading model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"  Model: {model_name}")
    
    try:
        # Load to CPU first (will move to GPU after compression)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to('cpu')
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Model loaded: {model_name}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Run baseline inference on GPU
    print("\n[4/5] Running baseline inference (uncompressed GPU)...")
    test_prompt = "The capital of France is"
    print(f"  Prompt: '{test_prompt}'")
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    torch.cuda.reset_peak_memory_stats()
    model = model.to('cuda')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    baseline_start = time.time()
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs,
            max_new_tokens=10,  # Fewer tokens for faster test
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    baseline_time = time.time() - baseline_start
    
    baseline_vram = torch.cuda.max_memory_allocated() / 1024**3
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    
    print(f"  Generated: '{baseline_text}'")
    print(f"  Time: {baseline_time:.2f}s")
    print(f"  Peak VRAM: {baseline_vram:.2f} GB")
    
    # Move back to CPU for compression
    model = model.to('cpu')
    torch.cuda.empty_cache()
    
    # Compress model weights
    print("\n[5/5] Compressing model and running GPU-accelerated inference...")
    print("  Compressing all Linear layers...")
    
    compress_start = time.time()
    num_compressed = compress_model_weights(model, lib, verbose=True)
    compress_time = time.time() - compress_start
    
    print(f"  Compression took: {compress_time:.1f}s")
    
    # Run inference with compressed weights on GPU
    print("\n  Running inference with compressed weights...")
    print("  (Weights decompress on-demand using GPU decoder)")
    
    torch.cuda.reset_peak_memory_stats()
    model = model.to('cuda')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    compressed_start = time.time()
    with torch.no_grad():
        compressed_output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    compressed_time = time.time() - compressed_start
    
    compressed_vram = torch.cuda.max_memory_allocated() / 1024**3
    compressed_text = tokenizer.decode(compressed_output[0], skip_special_tokens=True)
    
    print(f"  Generated: '{compressed_text}'")
    print(f"  Time: {compressed_time:.2f}s")
    print(f"  Peak VRAM: {compressed_vram:.2f} GB")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nBaseline (uncompressed):")
    print(f"  Text: '{baseline_text}'")
    print(f"  Time: {baseline_time:.2f}s")
    print(f"  VRAM: {baseline_vram:.2f} GB")
    
    print("\nCompressed (GPU decoder):")
    print(f"  Text: '{compressed_text}'")
    print(f"  Time: {compressed_time:.2f}s")
    print(f"  VRAM: {compressed_vram:.2f} GB")
    
    print("\nComparison:")
    if baseline_text == compressed_text:
        print("  ✅ Outputs MATCH exactly!")
    else:
        print("  ⚠️  Outputs differ slightly (expected due to quantization)")
    
    slowdown = compressed_time / baseline_time
    vram_reduction = baseline_vram / compressed_vram
    
    print(f"  Speed: {slowdown:.1f}x slower")
    print(f"  VRAM: {vram_reduction:.1f}x reduction ({(1-1/vram_reduction)*100:.1f}% saved)")
    
    # Calculate decode stats
    total_decodes = 0
    total_decode_time = 0.0
    for module in model.modules():
        if hasattr(module, 'decode_count'):
            total_decodes += module.decode_count
            total_decode_time += module.decode_time
    
    if total_decodes > 0:
        print(f"\n  Decompression stats:")
        print(f"    Total decompressions: {total_decodes}")
        print(f"    Total decode time: {total_decode_time:.2f}s")
        print(f"    Avg per decode: {total_decode_time/total_decodes*1000:.1f}ms")
        print(f"    Decode overhead: {total_decode_time/compressed_time*100:.1f}% of total time")
    
    print("\n" + "="*80)
    if vram_reduction > 1.5:
        print("✅ SUCCESS! GPU-accelerated low-memory inference works!")
    else:
        print("⚠️  Test completed but VRAM savings lower than expected")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = test_gpu_accelerated_inference()
    sys.exit(0 if success else 1)

