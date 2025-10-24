#!/usr/bin/env python3
"""
Test: Compress ONLY MLP layers, skip all attention layers
Solution: Attention layers are too sensitive, compress feed-forward layers only
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("TEST: MLP-ONLY COMPRESSION (SKIP ATTENTION)")
print("="*80)
print()
print("Strategy: Compress only gate_proj, up_proj, down_proj (MLP layers)")
print("Expected: Perfect accuracy + fast cache + good compression")
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Baseline
print("[1/3] Baseline inference...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    t_baseline = time.time() - t0

baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
baseline_vram = torch.cuda.max_memory_allocated() / 1024**3
print(f"  Output: '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s, VRAM: {baseline_vram:.2f} GB")
print()

# Reload and compress ONLY MLP layers
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

all_linear = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

# MLP layers only: gate_proj, up_proj, down_proj
# SKIP: q_proj, k_proj, v_proj, o_proj, embed, lm_head
mlp_layers = [(n, m) for n, m in all_linear 
              if any(x in n for x in ['gate_proj', 'up_proj', 'down_proj'])]

print(f"[2/3] Compressing {len(mlp_layers)} MLP layers...")
print(f"  gate_proj: {len([n for n, _ in mlp_layers if 'gate_proj' in n])}")
print(f"  up_proj: {len([n for n, _ in mlp_layers if 'up_proj' in n])}")
print(f"  down_proj: {len([n for n, _ in mlp_layers if 'down_proj' in n])}")
print(f"  (Skipping all attention layers)")
print()

encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()
compressed_weights = {}
total_original = 0
total_compressed = 0

for name, module in mlp_layers:
    weight = module.weight.data.cpu().numpy()
    
    # INT8 quantization
    scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8).astype(np.float32)
    weight_int8 = np.clip(np.round(weight / scales), -127, 127).astype(np.int8)
    
    compressed, _ = encoder.encode_layer(weight_int8)
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'scale': scales.squeeze().copy(),
    }
    
    total_original += weight.nbytes
    total_compressed += len(compressed)

overall_ratio = total_original / total_compressed
print(f"  Compressed: {total_original/1024**2:.1f} MB → {total_compressed/1024**2:.1f} MB")
print(f"  Ratio: {overall_ratio:.2f}x")
print()

class CompressedLinear(torch.nn.Module):
    def __init__(self, original_module, compressed_data, decoder_handle):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.decoder = decoder_handle
        
        scale_tensor = torch.from_numpy(compressed_data['scale']).to(torch.float16).to(device)
        self.register_buffer('scale', scale_tensor)
        
        if original_module.bias is not None:
            self.register_buffer('bias', original_module.bias.data.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        import ctypes
        
        gpu_ptr, rows, cols, _ = self.decoder.decode_layer_to_gpu(self.compressed)
        weight_int8 = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
        
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaMemcpy(
            ctypes.c_void_p(weight_int8.data_ptr()),
            ctypes.c_void_p(gpu_ptr),
            ctypes.c_size_t(rows * cols),
            ctypes.c_int(1)
        )
        cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
        
        weight_fp = weight_int8.to(torch.float16) * self.scale.view(-1, 1)
        weight_fp = weight_fp.reshape(self.shape)
        
        output = torch.nn.functional.linear(x, weight_fp, self.bias)
        
        del weight_fp, weight_int8
        return output

def replace_linear_with_compressed(module, compressed_weights, decoder):
    for name, child in list(module.named_children()):
        full_name = name
        if hasattr(module, '__module_name__'):
            full_name = module.__module_name__ + '.' + name
        
        if isinstance(child, torch.nn.Linear):
            for compressed_name, compressed_data in compressed_weights.items():
                if compressed_name.endswith('.' + name) or compressed_name == name:
                    setattr(module, name, CompressedLinear(child, compressed_data, decoder))
                    break
        else:
            child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
            replace_linear_with_compressed(child, compressed_weights, decoder)

print("  Replacing MLP layers...")
model.__module_name__ = ''
replace_linear_with_compressed(model, compressed_weights, decoder)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(f"  ✓ Replaced {len(compressed_weights)} MLP layers")
print()

# Test with cache ENABLED (should work now!)
print("[3/3] Running inference with use_cache=True...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, 
                           pad_token_id=tokenizer.eos_token_id, use_cache=True)
    t_compressed = time.time() - t0

compressed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
compressed_vram = torch.cuda.max_memory_allocated() / 1024**3

print()
print("="*80)
print("RESULTS")
print("="*80)
print()
print(f"Baseline (uncompressed, cache=True):")
print(f"  '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s, VRAM: {baseline_vram:.2f} GB")
print()
print(f"MLP-only compressed ({len(mlp_layers)} layers, cache=True):")
print(f"  '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x)")
print(f"  VRAM: {compressed_vram:.2f} GB ({(1 - compressed_vram/baseline_vram)*100:.1f}% savings)")
print(f"  Compression: {overall_ratio:.2f}x")
print()

if baseline_text == compressed_text:
    print("✓✓✓ PERFECT MATCH! ✓✓✓")
    print()
    print("SUCCESS: MLP-only compression works!")
    print()
    print("What worked:")
    print("  ✓ Skip all attention layers (q/k/v/o_proj)")
    print("  ✓ Compress only MLP layers (gate/up/down_proj)")
    print("  ✓ Cache enabled (fast generation)")
    print("  ✓ Perfect accuracy maintained")
    print()
    print(f"Trade-offs:")
    print(f"  ✓ {overall_ratio:.2f}x compression on MLP weights")
    print(f"  ✓ ~{(1 - compressed_vram/baseline_vram)*100:.0f}% VRAM savings")
    print(f"  ✓ {t_compressed/t_baseline:.1f}x speed (cache working)")
    print(f"  - Lower ratio than all-layers (but actually works!)")
    print()
    print("Next steps:")
    print("  1. Test on longer sequences (100+ tokens)")
    print("  2. Profile performance bottlenecks")
    print("  3. Scale to larger models (7B+)")
    print("  4. Optimize decompression speed")
else:
    print("✗ STILL BROKEN")
    print()
    print(f"Baseline:   '{baseline_text}'")
    print(f"Compressed: '{compressed_text}'")
    print()
    print("Even MLP-only compression fails.")
    print("Need to investigate further.")

print()

