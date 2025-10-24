#!/usr/bin/env python3
"""
Test: Compressed layers with KV cache DISABLED
Hypothesis: KV cache is causing autoregressive error amplification
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
print("TEST: DISABLE KV CACHE")
print("="*80)
print()
print("Hypothesis: KV cache is accumulating numerical errors from compressed layers")
print("Solution: use_cache=False forces model to recompute attention each token")
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Test 1: Baseline with cache
print("[1/4] Baseline WITH cache...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, 
                           pad_token_id=tokenizer.eos_token_id, use_cache=True)
baseline_cached = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Output: '{baseline_cached}'")
print()

# Test 2: Baseline WITHOUT cache
print("[2/4] Baseline WITHOUT cache...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, 
                           pad_token_id=tokenizer.eos_token_id, use_cache=False)
baseline_no_cache = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Output: '{baseline_no_cache}'")
print()

if baseline_cached != baseline_no_cache:
    print("⚠ WARNING: Baseline outputs differ with/without cache!")
    print("This should not happen and indicates a model issue.")
    print()

# Test 3: Compress just 1 layer and test with cache
print("[3/4] Compressing 1 layer, testing WITH cache...")
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

all_linear = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
attention_layers = [(n, m) for n, m in all_linear if 'q_proj' in n]
layer_to_compress = attention_layers[0:1]  # Just first Q projection

print(f"  Compressing: {layer_to_compress[0][0]}")

encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()
compressed_weights = {}

for name, module in layer_to_compress:
    weight = module.weight.data.cpu().numpy()
    scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8).astype(np.float32)
    weight_int8 = np.clip(np.round(weight / scales), -127, 127).astype(np.int8)
    compressed, _ = encoder.encode_layer(weight_int8)
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'scale': scales.squeeze().copy(),
    }

class CompressedLinear(torch.nn.Module):
    def __init__(self, original_module, compressed_data, decoder_handle):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.decoder = decoder_handle
        
        scale_tensor = torch.from_numpy(compressed_data['scale']).to(torch.float16).to('cuda')
        self.register_buffer('scale', scale_tensor)
        self.bias = original_module.bias.data.clone() if original_module.bias is not None else None
    
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
        output = torch.nn.functional.linear(x, weight_fp, self.bias)
        
        del weight_fp, weight_int8
        return output

# Replace layer
for name, child in model.named_modules():
    if name == layer_to_compress[0][0]:
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = model
        for part in parent_name.split('.'):
            if part:
                parent = getattr(parent, part)
        setattr(parent, child_name, CompressedLinear(child, compressed_weights[name], decoder))
        break

torch.cuda.empty_cache()

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, 
                           pad_token_id=tokenizer.eos_token_id, use_cache=True)
compressed_cached = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Output: '{compressed_cached}'")
print()

# Test 4: Same compressed layer, but WITHOUT cache
print("[4/4] Same 1 compressed layer, testing WITHOUT cache...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, 
                           pad_token_id=tokenizer.eos_token_id, use_cache=False)
    t_no_cache = time.time() - t0
compressed_no_cache = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Output: '{compressed_no_cache}'")
print(f"  Time: {t_no_cache:.2f}s")
print()

# Analysis
print("="*80)
print("RESULTS")
print("="*80)
print()
print(f"Baseline (cache=True):       '{baseline_cached}'")
print(f"Baseline (cache=False):      '{baseline_no_cache}'")
print(f"Compressed (cache=True):     '{compressed_cached}'")
print(f"Compressed (cache=False):    '{compressed_no_cache}'")
print()

if compressed_no_cache == baseline_no_cache:
    print("✓✓✓ SUCCESS! ✓✓✓")
    print()
    print("CONCLUSION: KV cache was the problem!")
    print("  - Without cache: Perfect output")
    print("  - With cache: Corrupted output")
    print()
    print("ROOT CAUSE:")
    print("  - Compressed attention layers produce slightly different K/V tensors")
    print("  - These are cached and reused for subsequent tokens")
    print("  - Small numerical differences compound over autoregressive generation")
    print()
    print("SOLUTIONS:")
    print("  1. Disable KV cache (slower but accurate)")
    print("  2. Improve compression quality (better codecs)")
    print("  3. Don't compress attention layers (only compress MLP)")
    print("  4. Use higher precision (FP32 cache even with FP16 weights)")
elif compressed_no_cache[:30] == baseline_no_cache[:30]:
    print("⚠ PARTIAL IMPROVEMENT")
    print("KV cache helps but isn't the only issue.")
else:
    print("✗ STILL BROKEN")
    print("KV cache is not the primary problem.")
    print("Issue is deeper in the compression/decompression pipeline.")

print()

