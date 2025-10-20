#!/usr/bin/env python3
"""
Debug version of Zstd inference test - shows detailed memory usage
"""
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import numpy as np
import torch
import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder

def print_memory_stats(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"\n{'='*60}")
        print(f"GPU Memory Stats: {label}")
        print(f"{'='*60}")
        print(f"  Total VRAM:     {total:.2f} GB")
        print(f"  Allocated:      {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"  Reserved/Cached: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"  Peak Allocated:  {max_allocated:.2f} GB")
        print(f"  Free:           {total - reserved:.2f} GB")
        print(f"{'='*60}\n")

print("ZSTD MEMORY DEBUG TEST")
print("="*80)
print()

# Initial memory state
print_memory_stats("Initial (before loading model)")

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Loading model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print_memory_stats("After loading model")

# Find Linear layers
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append((name, module))

print(f"Found {len(linear_layers)} Linear layers")
print(f"First 5 layers:")
for i, (name, module) in enumerate(linear_layers[:5]):
    weight_size = module.weight.data.nbytes / 1024**2
    print(f"  {i}: {name} - {weight_size:.1f} MB")
print()

# Test baseline inference
print("Running baseline inference...")
prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id)

baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Output: '{baseline_text}'")

print_memory_stats("After baseline inference")

# Clear everything
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
gc.collect()

print_memory_stats("After clearing cache")

# Compress just 1 layer
print("Compressing 1 layer...")
encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()

name, module = linear_layers[0]
weight = module.weight.data.cpu().numpy()
w_min, w_max = weight.min(), weight.max()
scale = max(abs(w_min), abs(w_max)) / 127.0
weight_int8 = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)

print(f"Layer: {name}")
print(f"  Original size: {weight.nbytes / 1024**2:.1f} MB")
print(f"  Shape: {weight.shape}")

compressed, ratio = encoder.encode_layer(weight_int8)
print(f"  Compressed size: {len(compressed) / 1024**2:.1f} MB")
print(f"  Ratio: {ratio:.2f}x")

print_memory_stats("After compression (still in CPU RAM)")

# Test decompression
print("Testing decompression...")
decompressed = decoder.decode_layer(compressed)
print(f"  Decompressed shape: {decompressed.shape}")
print(f"  Match: {np.array_equal(weight_int8, decompressed)}")

print_memory_stats("After decompression test")

# Clear again
del decompressed
torch.cuda.empty_cache()
gc.collect()

print_memory_stats("After cleanup")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

if total_vram < 10:
    print(f"⚠️  WARNING: Only {total_vram:.1f} GB VRAM available!")
    print("   This seems like a low-end GPU or shared instance.")
    print("   Expected 32GB but got much less.")
elif reserved > total_vram * 0.9:
    print(f"⚠️  WARNING: {reserved:.1f} GB / {total_vram:.1f} GB reserved!")
    print("   PyTorch cache is holding onto memory.")
    print("   Try: torch.cuda.empty_cache() or restart process.")
else:
    print(f"✓ Plenty of VRAM available: {total_vram:.1f} GB total")
    print(f"  Reserved: {reserved:.1f} GB")
    print(f"  Free: {total_vram - reserved:.1f} GB")

print("\nDone!")

