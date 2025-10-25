#!/usr/bin/env python3
"""
rANS LLM Test - Pure FP16 compression (NO INT8 quantization)
Tests if lossless FP16 compression works for LLM inference
"""

import sys
from pathlib import Path
import numpy as np
import ctypes
import torch
import time

print("="*80)
print("rANS FP16 TEST - NO QUANTIZATION")
print("="*80)
print()

# Load codec library
lib_path = Path("core_rans/build/libcodec_core.so")
if not lib_path.exists():
    print(f"✗ Library not found at {lib_path}")
    print("Run: cd core_rans/build && cmake .. && make")
    sys.exit(1)

lib = ctypes.CDLL(str(lib_path))

# Set return types
lib.encoder_create.restype = ctypes.c_void_p
lib.encoder_create.argtypes = [ctypes.c_uint16]
lib.encoder_destroy.argtypes = [ctypes.c_void_p]
lib.encoder_encode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t)
]
lib.encoder_encode.restype = ctypes.c_float
lib.free_buffer.argtypes = [ctypes.POINTER(ctypes.c_uint8)]

lib.decoder_create.restype = ctypes.c_void_p
lib.decoder_destroy.argtypes = [ctypes.c_void_p]
lib.decoder_decode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int8)
]
lib.decoder_decode.restype = ctypes.c_float

print("✓ Loaded libcodec_core.so")
print()

# Load model
print("[1/4] Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded")
print()

# Baseline inference
print("[2/4] Baseline inference...")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs_baseline = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)

baseline_text = tokenizer.decode(outputs_baseline[0], skip_special_tokens=True)
print(f"Output: '{baseline_text}'")
print()

# Compress 1 MLP layer (FP16, no quantization)
print("[3/4] Compressing 1 MLP layer with rANS (FP16 → FP16)...")
layer_name = "model.layers.0.mlp.gate_proj"
print(f"Target: {layer_name}")

# Get the layer
layer = model.model.layers[0].mlp.gate_proj
original_weight = layer.weight.data.clone()
print(f"  Shape: {original_weight.shape}")
print(f"  Dtype: {original_weight.dtype}")

# Convert FP16 to bytes (for compression)
weight_np = original_weight.cpu().numpy()
rows, cols = weight_np.shape

# Reinterpret FP16 as uint8 bytes (2 bytes per FP16)
weight_bytes = weight_np.view(np.uint8).reshape(-1)
print(f"  FP16 data: {rows * cols} values = {len(weight_bytes)} bytes")

# Reinterpret as int8 for rANS (which expects int8_t*)
weight_int8_view = weight_bytes.view(np.int8)

# Compress (tile by tile to avoid OOM)
tile_size = 256
tile_elements = tile_size * tile_size

# Split into tiles
num_tiles = (len(weight_int8_view) + tile_elements - 1) // tile_elements
print(f"  Tiles: {num_tiles} (tile_size={tile_size})")

compressed_tiles = []
encoder = lib.encoder_create(tile_size)

for i in range(num_tiles):
    start = i * tile_elements
    end = min(start + tile_elements, len(weight_int8_view))
    tile_data = weight_int8_view[start:end]
    
    # Pad to tile size
    if len(tile_data) < tile_elements:
        padded = np.zeros(tile_elements, dtype=np.int8)
        padded[:len(tile_data)] = tile_data
        tile_data = padded
    
    tile_2d = tile_data.reshape(tile_size, tile_size)
    
    # Encode tile
    data_ptr = tile_2d.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    output_ptr = ctypes.POINTER(ctypes.c_uint8)()
    output_size = ctypes.c_size_t()
    
    ratio = lib.encoder_encode(encoder, data_ptr, tile_size, tile_size,
                               ctypes.byref(output_ptr), ctypes.byref(output_size))
    
    # Copy compressed data
    compressed = bytes(ctypes.cast(output_ptr, 
                                  ctypes.POINTER(ctypes.c_uint8 * output_size.value)).contents)
    compressed_tiles.append(compressed)
    
    lib.free_buffer(output_ptr)

lib.encoder_destroy(encoder)

original_bytes = len(weight_bytes)
compressed_bytes = sum(len(t) for t in compressed_tiles)
print(f"  Compressed: {original_bytes / 1024**2:.1f} MB → {compressed_bytes / 1024**2:.1f} MB")
print(f"  Ratio: {original_bytes / compressed_bytes:.2f}x")
print()

# Decompress and replace weight
print("[4/4] Decompressing and running inference...")

decoder = lib.decoder_create()
all_data = []

for compressed in compressed_tiles:
    # Decompress tile
    decoded = np.zeros((tile_size, tile_size), dtype=np.int8)
    decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    
    # Convert bytes to ctypes array
    compressed_array = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
    compressed_ptr = ctypes.cast(compressed_array, ctypes.POINTER(ctypes.c_uint8))
    
    lib.decoder_decode(decoder, compressed_ptr, len(compressed), decoded_ptr)
    
    all_data.append(decoded.flatten())

lib.decoder_destroy(decoder)

# Reconstruct weight
full_data = np.concatenate(all_data)[:len(weight_int8_view)]

# CRITICAL: Verify compression is lossless
if not np.array_equal(weight_int8_view, full_data):
    print("✗ COMPRESSION NOT LOSSLESS!")
    diff = np.abs(weight_int8_view.astype(np.int16) - full_data.astype(np.int16))
    print(f"  Max diff: {diff.max()}")
    print(f"  Num diffs: {(diff > 0).sum()} / {diff.size}")
    sys.exit(1)

print("✓ Compression is bit-exact")

# Convert back to FP16
weight_bytes_recovered = full_data.view(np.uint8)
weight_fp16_recovered = weight_bytes_recovered.view(np.float16).reshape(rows, cols)

# Verify FP16 recovery is perfect
if not np.array_equal(weight_np, weight_fp16_recovered):
    print("✗ FP16 RECOVERY MISMATCH!")
    diff = np.abs(weight_np - weight_fp16_recovered)
    print(f"  Max diff: {diff.max()}")
    print(f"  Mean diff: {diff.mean()}")
    print(f"  Num diffs: {(diff != 0).sum()} / {diff.size}")
    sys.exit(1)

print("✓ FP16 recovery is perfect (bit-exact)")

# Replace weight
layer.weight.data = torch.from_numpy(weight_fp16_recovered).to("cuda")

print("✓ Weight replaced")

# Run inference with compressed weight
with torch.no_grad():
    outputs_compressed = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)

compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)

print()
print("="*80)
print("RESULTS")
print("="*80)
print(f"Baseline:   '{baseline_text}'")
print(f"Compressed: '{compressed_text}'")
print()

if baseline_text == compressed_text:
    print("✓✓✓ OUTPUTS MATCH PERFECTLY! ✓✓✓")
    print()
    print("This PROVES:")
    print("  ✓ rANS codec is 100% lossless for FP16 data")
    print("  ✓ FP16 compression works with LLM inference")
    print("  ✓ Static weight loading works perfectly")
    print("  ✓ NO quantization error when using pure FP16")
    print()
    print(f"Compression ratio: {original_bytes / compressed_bytes:.2f}x")
    print()
    print("Next steps:")
    print("  1. Test with more layers (should still match!)")
    print("  2. Test dynamic weight loading (the real blocker)")
    print("  3. If dynamic loading works, we have a solution!")
else:
    print("✗ OUTPUTS DIFFER")
    print()
    print("This is UNEXPECTED because:")
    print("  ✓ Compression is bit-exact (verified above)")
    print("  ✓ FP16 recovery is perfect (verified above)")
    print("  ✗ But outputs differ anyway")
    print()
    print("Possible causes:")
    print("  - Floating point non-determinism")
    print("  - PyTorch module state issue")
    print("  - Need to investigate further")

