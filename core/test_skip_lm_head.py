#!/usr/bin/env python3
"""
Test: Compress all layers EXCEPT LM head
Hypothesis: LM head quantization breaks token prediction
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
print("TEST: ALL LAYERS COMPRESSED EXCEPT LM HEAD")
print("="*80)
print()
print("Hypothesis: LM head quantization causes invalid token predictions")
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "The capital of France is"

# Baseline
print("[1/3] Baseline inference...")
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    t_baseline = time.time() - t0

baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
baseline_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
print(f"  Output: '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s, VRAM: {baseline_vram:.2f} GB")
print()

# Reload and compress (SKIP LM HEAD)
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

all_linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

# FILTER OUT LM HEAD!
linear_layers = [(n, m) for n, m in all_linear_layers if 'lm_head' not in n.lower()]

print(f"[2/3] Compressing {len(linear_layers)}/{len(all_linear_layers)} layers (SKIPPING LM HEAD)...")

# Check what we're skipping
skipped = [n for n, _ in all_linear_layers if 'lm_head' in n.lower()]
print(f"  Skipping layers: {skipped}")
print()

encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()
compressed_weights = {}
total_original = 0
total_compressed = 0

for i, (name, module) in enumerate(linear_layers):
    weight = module.weight.data.cpu().numpy()
    
    scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8).astype(np.float32)
    weight_int8 = np.clip(np.round(weight / scales), -127, 127).astype(np.int8)
    
    compressed, ratio = encoder.encode_layer(weight_int8)
    scales_to_store = scales.squeeze().copy()
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'scale': scales_to_store,
        'scale_dtype': scales_to_store.dtype,
        'dtype': weight.dtype,
        'ratio': ratio
    }
    
    total_original += weight.nbytes
    total_compressed += len(compressed)
    
    if (i + 1) % 20 == 0 or i == 0 or i == len(linear_layers) - 1:
        print(f"  Progress: {i+1}/{len(linear_layers)}...")

overall_ratio = total_original / total_compressed
print(f"  ✓ Compressed {len(linear_layers)} layers (ratio: {overall_ratio:.2f}x)")
print()

# Create compressed model
class CompressedLinear(torch.nn.Module):
    def __init__(self, original_module, compressed_data, decoder_handle, target_device='cuda'):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.dtype = torch.float16 if compressed_data['dtype'] == np.float16 else torch.float32
        self.decoder = decoder_handle
        
        scale_np = compressed_data['scale']
        scale_tensor = torch.from_numpy(scale_np).to(self.dtype).to(target_device) if isinstance(scale_np, np.ndarray) else torch.tensor(scale_np, dtype=self.dtype, device=target_device)
        self.register_buffer('scale', scale_tensor)
        
        if original_module.bias is not None:
            self.register_buffer('bias', original_module.bias.data.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        import ctypes
        gpu_ptr, rows, cols, _ = self.decoder.decode_layer_to_gpu(self.compressed)
        weight_int8_gpu = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
        
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaMemcpy(ctypes.c_void_p(weight_int8_gpu.data_ptr()), ctypes.c_void_p(gpu_ptr), ctypes.c_size_t(rows * cols), ctypes.c_int(1))
        cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
        
        weight_fp = weight_int8_gpu.to(self.dtype)
        if self.scale.dim() == 1:
            weight_fp = weight_fp * self.scale.view(-1, 1)
        else:
            weight_fp = weight_fp * self.scale
        
        weight_fp = weight_fp.reshape(self.shape)
        output = torch.nn.functional.linear(x, weight_fp, self.bias)
        
        del weight_fp, weight_int8_gpu
        return output

def replace_linear_with_compressed(module, compressed_weights, decoder):
    for name, child in list(module.named_children()):
        full_name = name
        if hasattr(module, '__module_name__'):
            full_name = module.__module_name__ + '.' + name
        
        if isinstance(child, torch.nn.Linear):
            for compressed_name, compressed_data in compressed_weights.items():
                if compressed_name.endswith('.' + name) or compressed_name == name:
                    setattr(module, name, CompressedLinear(child, compressed_data, decoder, target_device=device))
                    break
        else:
            child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
            replace_linear_with_compressed(child, compressed_weights, decoder)

print("  Replacing layers...")
model.__module_name__ = ''
replace_linear_with_compressed(model, compressed_weights, decoder)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(f"  ✓ Replaced {len(compressed_weights)} layers (LM head left uncompressed)")
print()

# Inference
print("[3/3] Running inference...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs_compressed = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    t_compressed = time.time() - t0

compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)
compressed_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

print()
print("="*80)
print("RESULTS")
print("="*80)
print()
print(f"Baseline (all FP16):")
print(f"  '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s, VRAM: {baseline_vram:.2f} GB")
print()
print(f"154 layers compressed (SKIPPED LM HEAD):")
print(f"  '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x)")
print(f"  VRAM: {compressed_vram:.2f} GB")
print(f"  Ratio: {overall_ratio:.2f}x")
print()

# Quality check
if baseline_text == compressed_text:
    print("✓✓✓ PERFECT MATCH! ✓✓✓")
    print()
    print("HYPOTHESIS CONFIRMED!")
    print("The LM head quantization was breaking token prediction.")
    print("Keeping LM head in FP16 fixes the quality issue!")
    print()
    print("Next steps:")
    print("  1. This is now production-ready!")
    print("  2. Optimize performance (batch decompress, caching)")
    print("  3. Test on longer sequences")
elif compressed_text[:30] == baseline_text[:30]:
    print("⚠ MINOR DIFFERENCE (first 30 chars match)")
    print()
    print("PARTIAL SUCCESS!")
    print("Output is much better, but not perfect.")
    print("May need to skip more layers (embeddings?)")
else:
    print("✗ STILL HAS ISSUES")
    print()
    print("HYPOTHESIS REJECTED")
    print("The problem is not just the LM head.")
    print("Need to investigate:")
    print("  - Token embeddings")
    print("  - Position embeddings")  
    print("  - Layer normalization")
    
print()

