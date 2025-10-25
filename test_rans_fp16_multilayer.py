#!/usr/bin/env python3
"""
rANS Multi-Layer FP16 Test
Compress multiple MLP layers with pure FP16 (no quantization)
"""

import sys
from pathlib import Path
import numpy as np
import ctypes
import torch
import time

print("="*80)
print("rANS MULTI-LAYER FP16 TEST")
print("="*80)
print()

# Configuration
NUM_LAYERS_TO_COMPRESS = 5  # Start with 5 layers
print(f"Configuration: Compressing {NUM_LAYERS_TO_COMPRESS} MLP layers (gate_proj only)")
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

# Helper function to compress FP16 layer
def compress_fp16_layer(weight_np, tile_size=256):
    """Compress FP16 weight tensor, return compressed tiles"""
    rows, cols = weight_np.shape
    
    # Reinterpret FP16 as uint8 bytes (2 bytes per FP16)
    weight_bytes = weight_np.view(np.uint8).reshape(-1)
    weight_int8_view = weight_bytes.view(np.int8)
    
    # Split into tiles
    tile_elements = tile_size * tile_size
    num_tiles = (len(weight_int8_view) + tile_elements - 1) // tile_elements
    
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
        
        lib.encoder_encode(encoder, data_ptr, tile_size, tile_size,
                          ctypes.byref(output_ptr), ctypes.byref(output_size))
        
        # Copy compressed data
        compressed = bytes(ctypes.cast(output_ptr, 
                                      ctypes.POINTER(ctypes.c_uint8 * output_size.value)).contents)
        compressed_tiles.append(compressed)
        
        lib.free_buffer(output_ptr)
    
    lib.encoder_destroy(encoder)
    
    return compressed_tiles, len(weight_bytes)

# Helper function to decompress FP16 layer
def decompress_fp16_layer(compressed_tiles, rows, cols, tile_size=256):
    """Decompress tiles back to FP16 weight tensor"""
    weight_bytes_len = rows * cols * 2  # 2 bytes per FP16
    
    decoder = lib.decoder_create()
    all_data = []
    
    for compressed in compressed_tiles:
        # Decompress tile
        decoded = np.zeros((tile_size, tile_size), dtype=np.int8)
        decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        compressed_array = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
        compressed_ptr = ctypes.cast(compressed_array, ctypes.POINTER(ctypes.c_uint8))
        
        lib.decoder_decode(decoder, compressed_ptr, len(compressed), decoded_ptr)
        
        all_data.append(decoded.flatten())
    
    lib.decoder_destroy(decoder)
    
    # Reconstruct weight
    full_data = np.concatenate(all_data)[:weight_bytes_len]
    weight_bytes_recovered = full_data.view(np.uint8)
    weight_fp16_recovered = weight_bytes_recovered.view(np.float16).reshape(rows, cols)
    
    return weight_fp16_recovered

# Compress multiple MLP layers
print(f"[3/4] Compressing {NUM_LAYERS_TO_COMPRESS} MLP layers (gate_proj)...")

compressed_layers = []
total_original_bytes = 0
total_compressed_bytes = 0

for layer_idx in range(NUM_LAYERS_TO_COMPRESS):
    layer = model.model.layers[layer_idx].mlp.gate_proj
    original_weight = layer.weight.data.clone()
    weight_np = original_weight.cpu().numpy()
    
    # Compress
    compressed_tiles, original_bytes = compress_fp16_layer(weight_np)
    compressed_bytes = sum(len(t) for t in compressed_tiles)
    
    compressed_layers.append({
        'layer': layer,
        'tiles': compressed_tiles,
        'shape': weight_np.shape,
        'original': weight_np,
    })
    
    total_original_bytes += original_bytes
    total_compressed_bytes += compressed_bytes
    
    print(f"  Layer {layer_idx}: {original_bytes/1024**2:.1f} MB → {compressed_bytes/1024**2:.1f} MB ({original_bytes/compressed_bytes:.2f}x)")

print(f"\nTotal: {total_original_bytes/1024**2:.1f} MB → {total_compressed_bytes/1024**2:.1f} MB")
print(f"Ratio: {total_original_bytes/total_compressed_bytes:.2f}x")
print()

# Decompress and replace weights
print("[4/4] Decompressing and running inference...")

for layer_data in compressed_layers:
    rows, cols = layer_data['shape']
    weight_recovered = decompress_fp16_layer(layer_data['tiles'], rows, cols)
    
    # Verify bit-exact
    if not np.array_equal(layer_data['original'], weight_recovered):
        print("✗ COMPRESSION NOT LOSSLESS!")
        sys.exit(1)
    
    # Replace weight
    layer_data['layer'].weight.data = torch.from_numpy(weight_recovered).to("cuda")

print(f"✓ All {NUM_LAYERS_TO_COMPRESS} layers decompressed and replaced (bit-exact)")

# Run inference with compressed weights
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
    print(f"✓✓✓ OUTPUTS MATCH PERFECTLY ({NUM_LAYERS_TO_COMPRESS} layers compressed)! ✓✓✓")
    print()
    print("This PROVES:")
    print(f"  ✓ rANS FP16 compression works with {NUM_LAYERS_TO_COMPRESS} layers")
    print("  ✓ Completely lossless (bit-exact verification)")
    print("  ✓ Static weight loading works perfectly")
    print("  ✓ Zero accuracy loss with pure FP16")
    print()
    print(f"Compression ratio: {total_original_bytes/total_compressed_bytes:.2f}x")
    print("  (Poor ratio expected - FP16 has high entropy)")
    print()
    print("Key insight:")
    print("  - FP16 compression ratio is poor (~1.0x)")
    print("  - But outputs are PERFECT (no quality loss)")
    print("  - Trade-off: Low compression vs perfect accuracy")
    print()
    print("Next steps:")
    print("  1. Try even more layers (10, 20, 50)")
    print("  2. Test dynamic weight loading (the real blocker)")
    print("  3. Consider INT8 quantization-aware training")
else:
    print("✗ OUTPUTS DIFFER")
    print()
    print("This is UNEXPECTED and suggests:")
    print("  - Possible floating point non-determinism")
    print("  - PyTorch state issue")
    print("  - Need to investigate")

