#!/usr/bin/env python3
"""
Test: Compress just 1 MLP layer to isolate the issue
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
print("TEST: 1 MLP LAYER COMPRESSED")
print("="*80)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Baseline
print("[1/3] Baseline...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t_baseline_start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    t_baseline = time.time() - t_baseline_start

baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s")
print()

# Compress just 1 MLP layer
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

all_linear = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
mlp_layers = [(n, m) for n, m in all_linear
              if any(x in n for x in ['gate_proj', 'up_proj', 'down_proj'])]

print(f"[2/3] Compressing 1 MLP layer: {mlp_layers[0][0]}...")
print(f"  Total MLP layers available: {len(mlp_layers)}")
print()

encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()
compressed_weights = {}

# Just compress the first MLP layer
name, module = mlp_layers[0]
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

print(f"  Compressed: {weight.nbytes/1024**2:.1f} MB → {len(compressed)/1024**2:.1f} MB")
print(f"  Ratio: {weight.nbytes/len(compressed):.2f}x")
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

print("  Replacing 1 MLP layer...")
model.__module_name__ = ''
replace_linear_with_compressed(model, compressed_weights, decoder)
torch.cuda.empty_cache()
print(f"  ✓ Replaced 1 MLP layer")
print()

# Test
print("[3/3] Testing 1 compressed MLP layer...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id, use_cache=True)
    t_compressed = time.time() - t0

compressed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print()
print("="*80)
print("RESULTS")
print("="*80)
print()
print(f"Baseline: '{baseline_text}'")
print(f"1-MLP:    '{compressed_text}'")
print(f"Baseline time: {t_baseline:.2f}s")
print(f"Compressed time: {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x)")
print()

if baseline_text == compressed_text:
    print("✓✓✓ 1 MLP LAYER WORKS! ✓✓✓")
    print()
    print("The issue is cumulative errors from many layers.")
    print("Need to optimize compression quality or use fewer layers.")
else:
    print("✗ EVEN 1 MLP LAYER FAILS")
    print()
    print("The issue is fundamental to MLP compression.")
    print("Need to investigate the compression pipeline.")

print()

