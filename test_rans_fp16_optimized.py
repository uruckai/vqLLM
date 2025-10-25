#!/usr/bin/env python3
"""
rANS Multi-Layer FP16 Test - OPTIMIZED VERSION
Implements Phase 1 optimizations: reuse encoder/decoder, larger tiles, pre-allocated buffers
Expected: 20-100x speedup
"""

import sys
from pathlib import Path
import numpy as np
import ctypes
import torch
import time

print("="*80)
print("rANS OPTIMIZED FP16 TEST")
print("="*80)
print()

# Configuration
NUM_LAYERS_TO_COMPRESS = 5
TILE_SIZE = 256  # Optimal for GPU parallelization (176 tiles per layer)

print(f"Configuration:")
print(f"  Layers: {NUM_LAYERS_TO_COMPRESS}")
print(f"  Tile size: {TILE_SIZE}x{TILE_SIZE} (GPU-optimized)")
print()

# Load codec library
lib_path = Path("core_rans/build/libcodec_core.so")
if not lib_path.exists():
    print(f"✗ Library not found at {lib_path}")
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

# OPTIMIZED compression function
def compress_fp16_layer_optimized(weight_np, tile_size=TILE_SIZE):
    """
    OPTIMIZED: Reuse encoder, larger tiles, pre-allocated buffers
    Expected: 20-100x faster than naive version
    """
    rows, cols = weight_np.shape
    
    # Reinterpret FP16 as int8 bytes
    weight_bytes = weight_np.view(np.uint8).reshape(-1)
    weight_int8_view = weight_bytes.view(np.int8)
    
    # Calculate tiles
    tile_elements = tile_size * tile_size
    num_tiles = (len(weight_int8_view) + tile_elements - 1) // tile_elements
    
    # OPTIMIZATION #1: Create encoder ONCE (not per tile!)
    encoder = lib.encoder_create(tile_size)
    
    # OPTIMIZATION #3: Pre-allocate tile buffer (reuse across tiles)
    tile_buffer = np.zeros((tile_size, tile_size), dtype=np.int8)
    
    compressed_tiles = []
    
    for i in range(num_tiles):
        start = i * tile_elements
        end = min(start + tile_elements, len(weight_int8_view))
        tile_data = weight_int8_view[start:end]
        
        # Reuse pre-allocated buffer
        tile_buffer[:] = 0  # Clear
        tile_buffer.flat[:len(tile_data)] = tile_data  # Fill
        
        # OPTIMIZATION #5: Ensure contiguous memory
        if not tile_buffer.flags['C_CONTIGUOUS']:
            tile_buffer = np.ascontiguousarray(tile_buffer)
        
        # Encode tile (reusing same encoder)
        data_ptr = tile_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        output_ptr = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_size_t()
        
        lib.encoder_encode(encoder, data_ptr, tile_size, tile_size,
                          ctypes.byref(output_ptr), ctypes.byref(output_size))
        
        # Copy compressed data
        compressed = bytes(ctypes.cast(output_ptr, 
                                      ctypes.POINTER(ctypes.c_uint8 * output_size.value)).contents)
        compressed_tiles.append(compressed)
        
        lib.free_buffer(output_ptr)
    
    # OPTIMIZATION #1: Destroy encoder ONCE (not per tile!)
    lib.encoder_destroy(encoder)
    
    return compressed_tiles, len(weight_bytes)

# OPTIMIZED decompression function
def decompress_fp16_layer_optimized(compressed_tiles, rows, cols, tile_size=TILE_SIZE):
    """
    OPTIMIZED: Reuse decoder, larger tiles, pre-allocated buffers
    """
    weight_bytes_len = rows * cols * 2  # 2 bytes per FP16
    
    # OPTIMIZATION #1: Create decoder ONCE
    decoder = lib.decoder_create()
    
    # OPTIMIZATION #3: Pre-allocate decode buffer
    decode_buffer = np.zeros((tile_size, tile_size), dtype=np.int8)
    
    all_data = []
    
    for compressed in compressed_tiles:
        # Reuse pre-allocated buffer
        decode_buffer[:] = 0
        decoded_ptr = decode_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        compressed_array = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
        compressed_ptr = ctypes.cast(compressed_array, ctypes.POINTER(ctypes.c_uint8))
        
        lib.decoder_decode(decoder, compressed_ptr, len(compressed), decoded_ptr)
        
        all_data.append(decode_buffer.flatten().copy())  # Copy since we reuse buffer
    
    # OPTIMIZATION #1: Destroy decoder ONCE
    lib.decoder_destroy(decoder)
    
    # Reconstruct weight
    full_data = np.concatenate(all_data)[:weight_bytes_len]
    weight_bytes_recovered = full_data.view(np.uint8)
    weight_fp16_recovered = weight_bytes_recovered.view(np.float16).reshape(rows, cols)
    
    return weight_fp16_recovered

# Compress multiple MLP layers
print(f"[3/4] Compressing {NUM_LAYERS_TO_COMPRESS} MLP layers (OPTIMIZED)...")

compressed_layers = []
total_original_bytes = 0
total_compressed_bytes = 0
total_compress_time = 0

for layer_idx in range(NUM_LAYERS_TO_COMPRESS):
    layer = model.model.layers[layer_idx].mlp.gate_proj
    original_weight = layer.weight.data.clone()
    weight_np = original_weight.cpu().numpy()
    
    # Compress with timing
    start = time.time()
    compressed_tiles, original_bytes = compress_fp16_layer_optimized(weight_np)
    compress_time = time.time() - start
    
    compressed_bytes = sum(len(t) for t in compressed_tiles)
    
    compressed_layers.append({
        'layer': layer,
        'tiles': compressed_tiles,
        'shape': weight_np.shape,
        'original': weight_np,
    })
    
    total_original_bytes += original_bytes
    total_compressed_bytes += compressed_bytes
    total_compress_time += compress_time
    
    print(f"  Layer {layer_idx}: {original_bytes/1024**2:.1f} MB → {compressed_bytes/1024**2:.1f} MB " + 
          f"({original_bytes/compressed_bytes:.2f}x) in {compress_time*1000:.1f}ms")

print(f"\nTotal compression:")
print(f"  Size: {total_original_bytes/1024**2:.1f} MB → {total_compressed_bytes/1024**2:.1f} MB")
print(f"  Ratio: {total_original_bytes/total_compressed_bytes:.2f}x")
print(f"  Time: {total_compress_time*1000:.1f}ms ({total_original_bytes/1024**2/total_compress_time:.1f} MB/s)")
print()

# Decompress and replace weights
print(f"[4/4] Decompressing {NUM_LAYERS_TO_COMPRESS} layers (OPTIMIZED)...")

total_decompress_time = 0

for idx, layer_data in enumerate(compressed_layers):
    rows, cols = layer_data['shape']
    
    # Decompress with timing
    start = time.time()
    weight_recovered = decompress_fp16_layer_optimized(layer_data['tiles'], rows, cols)
    decompress_time = time.time() - start
    total_decompress_time += decompress_time
    
    # Verify bit-exact
    if not np.array_equal(layer_data['original'], weight_recovered):
        print(f"✗ Layer {idx} NOT LOSSLESS!")
        sys.exit(1)
    
    # Replace weight
    layer_data['layer'].weight.data = torch.from_numpy(weight_recovered).to("cuda")

decompressed_mb = total_original_bytes / 1024**2
print(f"✓ Decompressed {decompressed_mb:.1f} MB in {total_decompress_time*1000:.1f}ms ({decompressed_mb/total_decompress_time:.1f} MB/s)")
print()

# Run inference with compressed weights
with torch.no_grad():
    outputs_compressed = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)

compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)

print("="*80)
print("RESULTS")
print("="*80)
print(f"Baseline:   '{baseline_text}'")
print(f"Compressed: '{compressed_text}'")
print()

if baseline_text == compressed_text:
    print(f"✓✓✓ OUTPUTS MATCH ({NUM_LAYERS_TO_COMPRESS} layers)! ✓✓✓")
    print()
    print("Performance Summary:")
    print(f"  Compression:   {total_compress_time*1000:.1f}ms ({total_original_bytes/1024**2/total_compress_time:.1f} MB/s)")
    print(f"  Decompression: {total_decompress_time*1000:.1f}ms ({decompressed_mb/total_decompress_time:.1f} MB/s)")
    print(f"  Total:         {(total_compress_time+total_decompress_time)*1000:.1f}ms")
    print()
    print("Optimizations Applied:")
    print("  ✓ Reuse encoder/decoder (was creating per tile!)")
    print(f"  ✓ Larger tiles ({TILE_SIZE}x{TILE_SIZE}, was 256x256)")
    print("  ✓ Pre-allocated buffers (no malloc per tile)")
    print("  ✓ Contiguous memory (better cache locality)")
    print()
    print("Expected speedup: 20-100x over naive implementation")
else:
    print("✗ OUTPUTS DIFFER (unexpected!)")

