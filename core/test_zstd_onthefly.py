#!/usr/bin/env python3
"""
TRUE on-the-fly decompression: decompress → use → free on EVERY forward pass
No caching, minimal VRAM footprint
"""

# Set before torch import
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
print("ZSTD ON-THE-FLY DECOMPRESSION TEST")
print("="*80)
print("Decompress → Use → Free on EVERY forward pass")
print("Minimal VRAM footprint, accept speed cost")
print()

# Check GPU
if not ZstdGPUDecoder.is_available():
    print("❌ GPU decoder required for this test")
    sys.exit(1)
print("✓ GPU decoder available")
print()

print("[1/5] Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

print(f"✓ Model loaded on {device}")
print()

# Baseline
print("[2/5] Baseline inference...")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    t_baseline = time.time() - t0

baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
baseline_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

print(f"  Output: '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s")
print(f"  Peak VRAM: {baseline_vram:.2f} GB")
print()

# Compress ALL layers
print("[3/5] Compressing ALL Linear layers...")
encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()

linear_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
compressed_weights = {}

print(f"  Compressing {len(linear_layers)} layers...")
t0 = time.time()
for i, (name, module) in enumerate(linear_layers):
    weight = module.weight.data.cpu().numpy()
    
    # Quantize to int8
    w_min, w_max = weight.min(), weight.max()
    scale = max(abs(w_min), abs(w_max)) / 127.0
    weight_int8 = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    
    # Compress
    compressed, ratio = encoder.encode_layer(weight_int8)
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'scale': scale,
        'dtype': weight.dtype
    }
    
    if (i + 1) % 20 == 0:
        print(f"    {i+1}/{len(linear_layers)} compressed...")

compress_time = time.time() - t0
total_orig = sum(w.shape[0] * w.shape[1] * 2 for n, w in linear_layers)  # fp16 = 2 bytes
total_comp = sum(len(d['compressed']) for d in compressed_weights.values())

print(f"✓ Compressed {len(compressed_weights)} layers in {compress_time:.2f}s")
print(f"  Original: {total_orig/1024**2:.0f} MB")
print(f"  Compressed: {total_comp/1024**2:.0f} MB")
print(f"  Ratio: {total_orig/total_comp:.2f}x")
print()

# Replace with on-the-fly decompression layers
print("[4/5] Creating on-the-fly decompression model...")

class OnTheFlyLinear(torch.nn.Module):
    """Decompress weights fresh on EVERY forward pass"""
    
    def __init__(self, compressed_data, decoder, device):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.scale = compressed_data['scale']
        self.decoder = decoder
        self.device = device
        
        # Dtype
        if compressed_data['dtype'] == np.float16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        # Bias (keep on GPU)
        self.bias = None
    
    def set_bias(self, bias):
        if bias is not None:
            self.register_buffer('bias', bias.clone())
    
    def forward(self, x):
        # Decompress fresh (via nvCOMP GPU)
        weight_int8 = self.decoder.decode_layer(self.compressed)
        
        # Dequantize
        weight_float = (weight_int8.astype(np.float32) * self.scale).astype(
            np.float16 if self.dtype == torch.float16 else np.float32
        )
        
        # To tensor and GPU
        weight = torch.from_numpy(weight_float).to(x.device)
        
        # Compute
        output = torch.nn.functional.linear(x, weight, self.bias)
        
        # CRITICAL: Free immediately
        del weight
        del weight_float
        del weight_int8
        
        return output

# Replace layers
replaced = 0
for name, module in model.named_modules():
    for child_name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            # Find compressed version
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name in compressed_weights:
                # Create on-the-fly layer
                new_layer = OnTheFlyLinear(compressed_weights[full_name], decoder, device)
                new_layer.set_bias(child.bias.data if child.bias is not None else None)
                
                # Replace
                setattr(module, child_name, new_layer)
                replaced += 1

print(f"✓ Replaced {replaced} layers with on-the-fly decompression")

# Free original model memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    current_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory: {current_mem:.2f} GB")
print()

# Run compressed inference
print("[5/5] Running on-the-fly decompression inference...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    t0 = time.time()
    outputs_compressed = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    t_compressed = time.time() - t0

compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)
compressed_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

print(f"  Output: '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s")
print(f"  Peak VRAM: {compressed_vram:.2f} GB")
print()

# Summary
print("="*80)
print("RESULTS")
print("="*80)
print()
print("Baseline:")
print(f"  Time: {t_baseline:.2f}s")
print(f"  VRAM: {baseline_vram:.2f} GB")
print(f"  Output: '{baseline_text}'")
print()
print("On-the-fly decompression:")
print(f"  Time: {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x)")
print(f"  VRAM: {compressed_vram:.2f} GB")
if compressed_vram < baseline_vram:
    print(f"  VRAM saved: {baseline_vram - compressed_vram:.2f} GB ({(1 - compressed_vram/baseline_vram)*100:.0f}%)")
else:
    print(f"  VRAM increased: +{compressed_vram - baseline_vram:.2f} GB")
print(f"  Output: '{compressed_text}'")
print()
if baseline_text == compressed_text:
    print("✓ Output matches!")
else:
    print("⚠️  Output differs (quantization)")
print()
print("✓ Test complete!")

