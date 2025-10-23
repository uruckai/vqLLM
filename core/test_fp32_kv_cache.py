#!/usr/bin/env python3
"""
Test: FP32 KV Cache with Compressed Attention Layers
Solution: Store K/V in FP32 while keeping compressed FP16 weights
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
print("TEST: FP32 KV CACHE WITH COMPRESSED ATTENTION")
print("="*80)
print()
print("Strategy: Compress attention weights, but store K/V cache in FP32")
print("Expected: Perfect accuracy with compressed weights + cache enabled")
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

# Reload model and compress attention layers
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

# Get all attention projections (Q, K, V, O)
all_linear = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
attention_layers = [(n, m) for n, m in all_linear 
                   if any(x in n for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])]

print(f"[2/3] Compressing {len(attention_layers)} attention projections...")
print(f"  Found: {len([n for n, _ in attention_layers if 'q_proj' in n])} Q projections")
print(f"  Found: {len([n for n, _ in attention_layers if 'k_proj' in n])} K projections")
print(f"  Found: {len([n for n, _ in attention_layers if 'v_proj' in n])} V projections")
print()

encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()
compressed_weights = {}
total_original = 0
total_compressed = 0

for name, module in attention_layers:
    weight = module.weight.data.cpu().numpy()
    
    # Quantize to INT8
    scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8).astype(np.float32)
    weight_int8 = np.clip(np.round(weight / scales), -127, 127).astype(np.int8)
    
    compressed, ratio = encoder.encode_layer(weight_int8)
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'scale': scales.squeeze().copy(),
        'dtype': weight.dtype,
    }
    
    total_original += weight.nbytes
    total_compressed += len(compressed)

overall_ratio = total_original / total_compressed
print(f"  ✓ Compressed {len(attention_layers)} attention layers")
print(f"  Ratio: {overall_ratio:.2f}x")
print()

# Compressed Linear with FP32 output option
class CompressedLinearFP32(torch.nn.Module):
    """Compressed linear that can output FP32 for KV cache stability"""
    def __init__(self, original_module, compressed_data, decoder_handle, output_fp32=False):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.decoder = decoder_handle
        self.output_fp32 = output_fp32  # True for K/V projections
        
        scale_tensor = torch.from_numpy(compressed_data['scale']).to(torch.float16).to(device)
        self.register_buffer('scale', scale_tensor)
        
        if original_module.bias is not None:
            self.register_buffer('bias', original_module.bias.data.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        import ctypes
        
        # Decompress
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
        
        # Dequantize
        weight_fp = weight_int8.to(torch.float16) * self.scale.view(-1, 1)
        weight_fp = weight_fp.reshape(self.shape)
        
        # Compute output
        if self.output_fp32:
            # For K/V projections: compute in FP32 for cache stability
            output = torch.nn.functional.linear(x.float(), weight_fp.float(), 
                                               self.bias.float() if self.bias is not None else None)
        else:
            # For Q/O projections: keep FP16
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
                    # Use FP32 output for K and V projections (they go into cache)
                    output_fp32 = ('k_proj' in compressed_name or 'v_proj' in compressed_name)
                    
                    setattr(module, name, CompressedLinearFP32(
                        child, compressed_data, decoder, 
                        output_fp32=output_fp32
                    ))
                    break
        else:
            child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
            replace_linear_with_compressed(child, compressed_weights, decoder)

print("  Replacing layers with FP32 KV cache...")
model.__module_name__ = ''
replace_linear_with_compressed(model, compressed_weights, decoder)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(f"  ✓ Replaced {len(compressed_weights)} layers")
print()

# Test inference with cache enabled
print("[3/3] Running inference (compressed + FP32 cache + cache enabled)...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs_compressed = model.generate(**inputs, max_new_tokens=10, do_sample=False, 
                                       pad_token_id=tokenizer.eos_token_id, use_cache=True)
    t_compressed = time.time() - t0

compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)
compressed_vram = torch.cuda.max_memory_allocated() / 1024**3

print()
print("="*80)
print("RESULTS")
print("="*80)
print()
print(f"Baseline (FP16, uncompressed, cache=True):")
print(f"  '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s, VRAM: {baseline_vram:.2f} GB")
print()
print(f"Compressed ({len(attention_layers)} attention layers, FP32 KV cache, cache=True):")
print(f"  '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s, VRAM: {compressed_vram:.2f} GB")
print(f"  Compression: {overall_ratio:.2f}x")
print()

# Quality check
if baseline_text == compressed_text:
    print("✓✓✓ PERFECT MATCH! ✓✓✓")
    print()
    print("SUCCESS: FP32 KV cache solves the problem!")
    print()
    print("What worked:")
    print("  ✓ Compressed ALL attention layers (Q/K/V/O)")
    print("  ✓ K/V projections output FP32 (stored in cache)")
    print("  ✓ Q/O projections stay FP16 (not cached)")
    print("  ✓ Cache enabled (fast generation)")
    print("  ✓ Perfect accuracy maintained")
    print()
    print("Memory breakdown:")
    print(f"  Attention weights: {total_original/1024**2:.1f} MB → {total_compressed/1024**2:.1f} MB ({overall_ratio:.2f}x)")
    print(f"  KV cache: ~{(compressed_vram - baseline_vram)*1024:.0f} MB extra (FP32 vs FP16)")
    print()
    print("Next steps:")
    print("  1. Add MLP compression for higher ratio")
    print("  2. Profile performance vs baseline")
    print("  3. Test on longer sequences")
    print("  4. Scale to larger models")
elif compressed_text[:30] == baseline_text[:30]:
    print("⚠ CLOSE MATCH (first 30 chars identical)")
    print("FP32 cache helps significantly but may need tuning.")
    print()
    print(f"Baseline:   '{baseline_text}'")
    print(f"Compressed: '{compressed_text}'")
else:
    print("✗ STILL HAS ISSUES")
    print()
    print(f"Baseline:   '{baseline_text}'")
    print(f"Compressed: '{compressed_text}'")
    print()
    print("FP32 cache didn't fully solve it. Possible reasons:")
    print("  - Need higher precision in other places")
    print("  - Implementation bug in FP32 path")
    print("  - Need to also handle Q projection differently")

print()

