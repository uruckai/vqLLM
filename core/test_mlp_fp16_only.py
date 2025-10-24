#!/usr/bin/env python3
"""
Test: MLP layers with FP16 compression (NO INT8 quantization)
Goal: Test if removing quantization fixes MLP compression
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
print("TEST: MLP FP16 COMPRESSION (NO INT8 QUANTIZATION)")
print("="*80)
print()
print("Strategy: Compress FP16 weights directly, skip INT8 quantization")
print("Expected: Perfect accuracy with lossless FP16 compression")
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

# Reload and compress ONLY MLP layers with FP16 (no quantization)
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

all_linear = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
mlp_layers = [(n, m) for n, m in all_linear 
              if any(x in n for x in ['gate_proj', 'up_proj', 'down_proj'])]

print(f"[2/3] Compressing {len(mlp_layers)} MLP layers (FP16, no INT8)...")
print(f"  gate_proj: {len([n for n, _ in mlp_layers if 'gate_proj' in n])}")
print(f"  up_proj: {len([n for n, _ in mlp_layers if 'up_proj' in n])}")
print(f"  down_proj: {len([n for n, _ in mlp_layers if 'down_proj' in n])}")
print()

encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()
compressed_weights = {}
total_original = 0
total_compressed = 0

for name, module in mlp_layers:
    weight = module.weight.data.cpu().numpy().astype(np.float16)
    
    # NO INT8 QUANTIZATION - compress FP16 bytes directly
    weight_bytes = weight.tobytes()
    weight_int8_view = np.frombuffer(weight_bytes, dtype=np.int8).reshape(weight.shape[0], weight.shape[1] * 2)
    
    compressed, _ = encoder.encode_layer(weight_int8_view)
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'dtype': weight.dtype,
    }
    
    total_original += weight.nbytes
    total_compressed += len(compressed)

overall_ratio = total_original / total_compressed
print(f"  Compressed: {total_original/1024**2:.1f} MB → {total_compressed/1024**2:.1f} MB")
print(f"  Ratio: {overall_ratio:.2f}x")
print()

class CompressedLinearFP16(torch.nn.Module):
    """FP16 compressed linear (no quantization)"""
    def __init__(self, original_module, compressed_data, decoder_handle):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.decoder = decoder_handle
        
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
        
        output = torch.nn.functional.linear(x, weight_fp16, self.bias)
        
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
                    setattr(module, name, CompressedLinearFP16(child, compressed_data, decoder))
                    break
        else:
            child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
            replace_linear_with_compressed(child, compressed_weights, decoder)

print("  Replacing MLP layers (FP16 compression)...")
model.__module_name__ = ''
replace_linear_with_compressed(model, compressed_weights, decoder)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(f"  ✓ Replaced {len(compressed_weights)} MLP layers")
print()

# Test with cache enabled
print("[3/3] Running inference (FP16 MLP compression, cache=True)...")
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
print(f"Baseline (FP16, uncompressed):")
print(f"  '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s, VRAM: {baseline_vram:.2f} GB")
print()
print(f"MLP-only (FP16 compressed, NO INT8, {len(mlp_layers)} layers):")
print(f"  '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x)")
print(f"  VRAM: {compressed_vram:.2f} GB")
print(f"  Compression: {overall_ratio:.2f}x")
print()

if baseline_text == compressed_text:
    print("✓✓✓ PERFECT MATCH! ✓✓✓")
    print()
    print("SUCCESS: FP16 MLP compression works!")
    print()
    print("What worked:")
    print("  ✓ NO INT8 quantization (lossless FP16)")
    print("  ✓ Compress only MLP layers (gate/up/down_proj)")
    print("  ✓ Cache enabled (fast generation)")
    print("  ✓ Perfect accuracy maintained")
    print()
    print(f"Trade-offs:")
    print(f"  ✓ {overall_ratio:.2f}x compression ratio")
    print(f"  - Lower than INT8 (~1.3x vs ~2.3x)")
    print(f"  - But actually works!")
    print()
    print("Next steps:")
    print("  1. Optimize decompression speed (currently slow)")
    print("  2. Test on longer sequences")
    print("  3. Consider FP8 on Blackwell for better ratio")
else:
    print("✗ STILL BROKEN")
    print()
    print(f"Baseline:   '{baseline_text}'")
    print(f"Compressed: '{compressed_text}'")
    print()
    print("Even FP16 lossless compression fails.")
    print("The issue must be in:")
    print("  - Byte reinterpretation logic")
    print("  - GPU decode path")
    print("  - PyTorch module integration")

print()

