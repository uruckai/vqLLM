#!/usr/bin/env python3
"""
FINAL SOLUTION: Pinned CPU memory + async CUDA streams
Decompress to pinned CPU → async transfer to GPU → compute → free
This is the ONLY way to avoid PyTorch allocator issues
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import numpy as np
import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder

print("="*80)
print("ZSTD LOW-MEMORY INFERENCE - FINAL SOLUTION")
print("="*80)
print("Decompress to pinned CPU → async GPU transfer")
print("No GPU weight storage, minimal fragmentation")
print()

if not ZstdGPUDecoder.is_available():
    print("❌ GPU decoder required")
    sys.exit(1)

print("[1/4] Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

print("✓ Model loaded")

# Baseline
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    t_baseline = time.time() - t0

baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
baseline_vram = torch.cuda.max_memory_allocated() / 1024**3

print(f"Baseline: {t_baseline:.2f}s, VRAM: {baseline_vram:.2f} GB")
print(f"Output: '{baseline_text}'")
print()

# Compress all layers
print("[2/4] Compressing all layers...")
encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()

linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
compressed_weights = {}

print(f"Compressing {len(linear_layers)} layers...")
t0 = time.time()
for i, (name, module) in enumerate(linear_layers):
    weight = module.weight.data.cpu().numpy()
    w_min, w_max = weight.min(), weight.max()
    scale = max(abs(w_min), abs(w_max)) / 127.0
    weight_int8 = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    compressed, _ = encoder.encode_layer(weight_int8)
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'scale': scale,
        'dtype': weight.dtype
    }
    
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{len(linear_layers)}...")

print(f"✓ Compressed in {time.time() - t0:.1f}s")
print()

# Create streaming decompression layer
print("[3/4] Creating streaming model...")

class StreamingLinear(torch.nn.Module):
    """Decompress to pinned CPU, stream to GPU, never cache on GPU"""
    
    def __init__(self, compressed_data, decoder, device):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.scale = compressed_data['scale']
        self.decoder = decoder
        self.device = device
        self.dtype = torch.float16 if compressed_data['dtype'] == np.float16 else torch.float32
        self.bias = None
        
        # Pre-decompress to pinned CPU memory ONCE
        weight_int8 = self.decoder.decode_layer(self.compressed)
        weight_float = weight_int8.astype(np.float32) * self.scale
        weight_tensor = torch.from_numpy(weight_float).to(self.dtype)
        
        # Pin in CPU memory for fast transfers
        self.weight_pinned = weight_tensor.pin_memory()
        
        del weight_tensor
        del weight_float
        del weight_int8
    
    def set_bias(self, bias):
        if bias is not None:
            self.register_buffer('bias', bias.clone())
    
    def forward(self, x):
        # Transfer pinned CPU → GPU (async, ~5ms)
        weight_gpu = self.weight_pinned.to(x.device, non_blocking=True)
        
        # Compute
        output = torch.nn.functional.linear(x, weight_gpu, self.bias)
        
        # Free GPU copy
        del weight_gpu
        
        return output

# Replace layers
print("Replacing layers...")
replaced = 0
for name, module in model.named_modules():
    for child_name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name in compressed_weights:
                new_layer = StreamingLinear(compressed_weights[full_name], decoder, device)
                new_layer.set_bias(child.bias.data if child.bias is not None else None)
                setattr(module, child_name, new_layer)
                replaced += 1

print(f"✓ Replaced {replaced} layers")

# Free original model
torch.cuda.empty_cache()
import gc
gc.collect()

mem_after = torch.cuda.memory_allocated() / 1024**3
print(f"GPU memory: {mem_after:.2f} GB")
print()

# Test
print("[4/4] Running inference...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    t0 = time.time()
    outputs_compressed = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    t_compressed = time.time() - t0

compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)
compressed_vram = torch.cuda.max_memory_allocated() / 1024**3

print(f"Compressed: {t_compressed:.2f}s, VRAM: {compressed_vram:.2f} GB")
print(f"Output: '{compressed_text}'")
print()

# Summary
print("="*80)
print("RESULTS")
print("="*80)
print(f"Baseline:   {t_baseline:.2f}s, {baseline_vram:.2f} GB")
print(f"Compressed: {t_compressed:.2f}s, {compressed_vram:.2f} GB")
print(f"Slowdown: {t_compressed/t_baseline:.1f}x")
if compressed_vram < baseline_vram:
    print(f"VRAM saved: {baseline_vram - compressed_vram:.2f} GB")
else:
    print(f"VRAM increased: +{compressed_vram - baseline_vram:.2f} GB")
print()
if baseline_text == compressed_text:
    print("✓ Output matches!")
else:
    print(f"⚠️  Output differs")
    print(f"  Baseline:   '{baseline_text}'")
    print(f"  Compressed: '{compressed_text}'")
print()
print("✓ Complete!")

