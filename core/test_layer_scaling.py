#!/usr/bin/env python3
"""
Test: Find the breaking point - how many compressed layers before failure?
Run with cache=False to isolate layer count from KV cache issues
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
print("TEST: LAYER SCALING - FIND THE BREAKING POINT")
print("="*80)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Baseline
print("[Baseline] Uncompressed...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, 
                           pad_token_id=tokenizer.eos_token_id, use_cache=False)
baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Output: '{baseline_text}'")
print()

# Get all attention layers
all_linear = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
attention_layers = [(n, m) for n, m in all_linear 
                   if any(x in n for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])]

print(f"Total attention layers: {len(attention_layers)}")
print()

# Compressed Linear Layer
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
    replaced_count = 0
    for name, child in list(module.named_children()):
        full_name = name
        if hasattr(module, '__module_name__'):
            full_name = module.__module_name__ + '.' + name
        
        if isinstance(child, torch.nn.Linear):
            for compressed_name in compressed_weights.keys():
                if compressed_name.endswith('.' + name) or compressed_name == name:
                    setattr(module, name, CompressedLinear(
                        child, compressed_weights[compressed_name], decoder
                    ))
                    replaced_count += 1
                    break
        else:
            child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
            replaced_count += replace_linear_with_compressed(child, compressed_weights, decoder)
    return replaced_count

# Test with different layer counts
layer_counts = [1, 4, 11, 22, 44, 88]  # 1, first layer, first 3 layers, 1 full block, 2 blocks, all
results = []

encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()

for num_layers in layer_counts:
    print(f"[Test] Compressing {num_layers} layers (cache=False)...")
    
    # Reload model
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    
    # Compress first N layers
    layers_to_compress = attention_layers[:num_layers]
    compressed_weights = {}
    
    for name, module in layers_to_compress:
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
    
    # Replace layers
    model.__module_name__ = ''
    replaced = replace_linear_with_compressed(model, compressed_weights, decoder)
    
    # Run inference
    torch.cuda.empty_cache()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        t0 = time.time()
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, 
                               pad_token_id=tokenizer.eos_token_id, use_cache=False)
        t = time.time() - t0
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = "✓" if output_text == baseline_text else "✗"
    
    print(f"  Replaced: {replaced}/{num_layers}")
    print(f"  Output: '{output_text}'")
    print(f"  Match: {match}, Time: {t:.2f}s")
    print()
    
    results.append({
        'num_layers': num_layers,
        'replaced': replaced,
        'output': output_text,
        'match': match,
        'time': t
    })

# Summary
print("="*80)
print("SUMMARY: LAYER SCALING TEST")
print("="*80)
print()
print(f"Baseline: '{baseline_text}'")
print()
print("Layer Count | Replaced | Match | Time   | Output Preview")
print("-" * 80)
for r in results:
    preview = r['output'][:40] + "..." if len(r['output']) > 40 else r['output']
    print(f"{r['num_layers']:11d} | {r['replaced']:8d} | {r['match']:5s} | {r['time']:5.1f}s | {preview}")

print()
print("="*80)
print("ANALYSIS")
print("="*80)
print()

# Find breaking point
breaking_point = None
for i, r in enumerate(results):
    if r['match'] == '✗':
        breaking_point = r['num_layers']
        if i > 0:
            print(f"✓ Layers 1-{results[i-1]['num_layers']}: WORKING")
        print(f"✗ Layers {r['num_layers']}+: BROKEN")
        print()
        print("CONCLUSION:")
        print(f"  Error amplification occurs after ~{results[i-1]['num_layers'] if i > 0 else 0} compressed attention layers")
        print("  Even without KV cache, quantization errors compound through many layers")
        print()
        print("SOLUTION:")
        print(f"  Option 1: Compress only first {results[i-1]['num_layers'] if i > 0 else 0} layers")
        print("  Option 2: Skip attention entirely, compress only MLP layers")
        print("  Option 3: Use higher precision (FP16 instead of INT8)")
        break

if breaking_point is None:
    print("✓ ALL LAYER COUNTS WORKING!")
    print("No breaking point found - something else is wrong.")

print()

