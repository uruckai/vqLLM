#!/usr/bin/env python3
"""
Test: FP16 Compression + FP32 KV Cache (NO QUANTIZATION)
Solution: Skip INT8 quantization entirely, store cache in FP32
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
print("TEST: FP16 COMPRESSION + FP32 KV CACHE (NO INT8)")
print("="*80)
print()
print("Strategy: FP16 → Zstd → FP16, output as FP32 for K/V cache")
print("Expected: Perfect accuracy (no quantization loss + stable cache)")
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

print(f"[2/3] Compressing {len(attention_layers)} attention layers (FP16, no quantization)...")
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
    weight = module.weight.data.cpu().numpy().astype(np.float16)
    
    # NO QUANTIZATION - compress FP16 directly
    weight_bytes = weight.tobytes()
    weight_int8_view = np.frombuffer(weight_bytes, dtype=np.int8).reshape(weight.shape[0], weight.shape[1] * 2)
    
    compressed, ratio = encoder.encode_layer(weight_int8_view)
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'dtype': weight.dtype,
    }
    
    total_original += weight.nbytes
    total_compressed += len(compressed)

overall_ratio = total_original / total_compressed
print(f"  ✓ Compressed {len(attention_layers)} attention layers")
print(f"  Ratio: {overall_ratio:.2f}x")
print()

# Compressed Linear with FP32 output option (NO quantization)
class CompressedLinearFP16FP32(torch.nn.Module):
    """FP16 compressed linear that outputs FP32 for KV cache stability"""
    def __init__(self, original_module, compressed_data, decoder_handle, output_fp32=False):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.decoder = decoder_handle
        self.output_fp32 = output_fp32  # True for K/V projections
        
        if original_module.bias is not None:
            self.register_buffer('bias', original_module.bias.data.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        import ctypes
        
        # Decompress FP16 weights (stored as int8 bytes)
        gpu_ptr, rows, cols, _ = self.decoder.decode_layer_to_gpu(self.compressed)
        weight_bytes_gpu = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
        
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaMemcpy(
            ctypes.c_void_p(weight_bytes_gpu.data_ptr()),
            ctypes.c_void_p(gpu_ptr),
            ctypes.c_size_t(rows * cols),
            ctypes.c_int(1)
        )
        cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
        
        # Reinterpret bytes as FP16 (NO dequantization, just type cast)
        weight_as_bytes = weight_bytes_gpu.flatten()
        weight_as_int16 = weight_as_bytes.view(torch.int16)
        weight_fp16 = weight_as_int16.view(torch.float16).reshape(self.shape)
        
        # Compute output
        if self.output_fp32:
            # For Q/K/V projections: compute and output in FP32 for cache stability
            output = torch.nn.functional.linear(x.float(), weight_fp16.float(), 
                                               self.bias.float() if self.bias is not None else None)
        else:
            # For O projection: keep FP16 (faster, not cached)
            output = torch.nn.functional.linear(x.to(weight_fp16.dtype), weight_fp16, self.bias)
        
        del weight_fp16, weight_as_bytes, weight_as_int16, weight_bytes_gpu
        return output

def replace_linear_with_compressed(module, compressed_weights, decoder):
    for name, child in list(module.named_children()):
        full_name = name
        if hasattr(module, '__module_name__'):
            full_name = module.__module_name__ + '.' + name
        
        if isinstance(child, torch.nn.Linear):
            for compressed_name, compressed_data in compressed_weights.items():
                if compressed_name.endswith('.' + name) or compressed_name == name:
                    # Use FP32 output for Q/K/V projections (attention needs matching dtypes)
                    output_fp32 = any(proj in compressed_name for proj in ['q_proj', 'k_proj', 'v_proj'])
                    
                    setattr(module, name, CompressedLinearFP16FP32(
                        child, compressed_data, decoder, 
                        output_fp32=output_fp32
                    ))
                    break
        else:
            child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
            replace_linear_with_compressed(child, compressed_weights, decoder)

print("  Replacing layers with FP16+FP32 cache...")
model.__module_name__ = ''
replace_linear_with_compressed(model, compressed_weights, decoder)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(f"  ✓ Replaced {len(compressed_weights)} layers")
print()

# Test inference with cache enabled
print("[3/3] Running inference (FP16 compressed + FP32 cache + cache enabled)...")
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
print(f"Compressed (FP16→FP16, FP32 KV cache, cache=True):")
print(f"  '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s, VRAM: {compressed_vram:.2f} GB")
print(f"  Compression: {overall_ratio:.2f}x")
print()

# Quality check
if baseline_text == compressed_text:
    print("✓✓✓ PERFECT MATCH! ✓✓✓")
    print()
    print("SUCCESS: FP16 compression + FP32 cache works!")
    print()
    print("What worked:")
    print("  ✓ NO INT8 quantization (lossless FP16)")
    print("  ✓ Q/K/V projections output FP32 (cache stability)")
    print("  ✓ O projection stays FP16 (faster)")
    print("  ✓ Cache enabled (fast generation)")
    print("  ✓ Perfect accuracy maintained")
    print()
    print("Next steps:")
    print("  1. Add MLP compression for higher ratio")
    print("  2. Test on longer sequences")
    print("  3. Compare INT8 quantization savings")
elif compressed_text[:30] == baseline_text[:30]:
    print("⚠ CLOSE MATCH (first 30 chars identical)")
    print()
    print(f"Baseline:   '{baseline_text}'")
    print(f"Compressed: '{compressed_text}'")
else:
    print("✗ STILL HAS ISSUES")
    print()
    print(f"Baseline:   '{baseline_text}'")
    print(f"Compressed: '{compressed_text}'")
    print()
    print("Even FP16+FP32 doesn't work. Possible reasons:")
    print("  - Compression/decompression introduces errors")
    print("  - Byte reinterpretation is wrong")
    print("  - Need to skip attention layers entirely")

print()

