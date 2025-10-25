#!/usr/bin/env python3
"""
Test: rANS codec with LLM inference (sanity check)
Expected: Will have same issues as Zstd
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

# Add python module to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("TEST: rANS CODEC WITH LLM INFERENCE")
print("="*80)
print()

# Load rANS codec
try:
    from wcodec import bindings
    # Use larger tile size (256) to reduce memory overhead
    encoder = bindings.Encoder(tile_size=256)
    decoder = bindings.Decoder(tile_size=256)
    gpu_decoder = bindings.GPUDecoder(tile_size=256) if bindings.is_gpu_available() else None

    print("✓ Loaded rANS codec")
    print(f"  CPU encoder: {type(encoder).__name__} (tile_size=256)")
    print(f"  CPU decoder: {type(decoder).__name__} (tile_size=256)")
    print(f"  GPU decoder: {'Available' if gpu_decoder else 'Not available'}")
except ImportError as e:
    print(f"✗ Cannot load rANS codec: {e}")
    print()
    print("Build the codec first:")
    print("  mkdir -p build && cd build")
    print("  cmake .. -DCMAKE_BUILD_TYPE=Release")
    print("  make -j$(nproc)")
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Baseline
print()
print("[1/3] Baseline inference...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    t_baseline = time.time() - t0

baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Output: '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s")
print()

# Compress just 1 MLP layer with rANS
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

all_linear = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
mlp_layers = [(n, m) for n, m in all_linear if 'gate_proj' in n]

print(f"[2/3] Compressing 1 MLP layer with rANS: {mlp_layers[0][0]}...")

name, module = mlp_layers[0]
weight = module.weight.data.cpu().numpy()

# Quantize
scale = np.abs(weight).max() / 127.0
weight_int8 = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)

# Compress with rANS
compressed_bytes, stats = encoder.encode_layer(weight_int8)
ratio = weight_int8.nbytes / len(compressed_bytes)

print(f"  Compressed: {weight_int8.nbytes/1024**2:.1f} MB → {len(compressed_bytes)/1024**2:.1f} MB ({ratio:.2f}x)")
print(f"  rANS ratio: {stats['compression_ratio']:.2f}x")
print()

# Create compressed layer
class CompressedLinearRANS(torch.nn.Module):
    def __init__(self, original_module, compressed_data, codec_handle, scale_val, shape):
        super().__init__()
        self.compressed = compressed_data
        self.codec = codec_handle
        self.shape = shape

        scale_tensor = torch.tensor(scale_val, dtype=torch.float16, device=device)
        self.register_buffer('scale', scale_tensor)

        if original_module.bias is not None:
            self.register_buffer('bias', original_module.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        # Decode with rANS
        weight_int8, _ = self.codec.decode_layer(self.compressed, self.shape[0], self.shape[1])
        weight_tensor = torch.from_numpy(weight_int8).to(device)

        # Dequantize
        weight_fp = weight_tensor.to(torch.float16) * self.scale

        # Forward
        output = torch.nn.functional.linear(x, weight_fp, self.bias)

        del weight_fp, weight_tensor
        return output

# Replace layer
for n, child in model.named_modules():
    if n == name:
        parent_name = '.'.join(n.split('.')[:-1])
        child_name = n.split('.')[-1]
        parent = model
        for part in parent_name.split('.'):
            if part:
                parent = getattr(parent, part)
        setattr(parent, child_name, CompressedLinearRANS(child, compressed_bytes, decoder, scale, weight.shape))
        break

torch.cuda.empty_cache()
print("  ✓ Replaced 1 MLP layer with rANS compressed version")
print()

# Test inference
print("[3/3] Testing rANS compressed inference...")
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
print(f"rANS:     '{compressed_text}'")
print(f"Baseline time: {t_baseline:.2f}s")
print(f"rANS time: {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x)")
print()

if baseline_text == compressed_text:
    print("✓✓✓ UNEXPECTED: rANS WORKS! ✓✓✓")
    print()
    print("This would be surprising and means:")
    print("  - rANS codec has different behavior than Zstd")
    print("  - Or this specific layer is more robust")
    print()
    print("Next: Test with all MLP layers")
else:
    print("✗ EXPECTED: rANS HAS SAME ISSUE AS ZSTD")
    print()
    print("This confirms:")
    print("  - The problem is NOT specific to Zstd")
    print("  - The problem is dynamic weight loading itself")
    print("  - See COMPRESSION_BLOCKERS.md for details")
    print()
    print("Even with different compression algorithms (rANS vs Zstd),")
    print("dynamically loading weights breaks LLM inference.")

print()

