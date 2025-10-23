#!/usr/bin/env python3
"""
All-layers compression test with MINIMAL logging
Only shows critical info and actionable errors
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import sys
import time
from pathlib import Path

# Suppress transformers warnings
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("ALL LAYERS COMPRESSED TEST (QUIET MODE)")
print("="*80)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print()

# Load model
print("[1/3] Loading model...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
prompt = "The capital of France is"

# Baseline
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
print(f"  Found {len(linear_layers)} Linear layers")

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    t_baseline = time.time() - t0

baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
baseline_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

print(f"  Baseline output: '{baseline_text[:50]}...'")
print(f"  Time: {t_baseline:.2f}s, VRAM: {baseline_vram:.2f} GB")
print()

# Reload for compression
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

# Compress ALL layers
print(f"[2/3] Compressing ALL {len(linear_layers)} layers...")
encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()

compressed_weights = {}
total_original = 0
total_compressed = 0

for i, (name, module) in enumerate(linear_layers):
    weight = module.weight.data.cpu().numpy()
    
    # Per-channel quantization
    scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8)
    scales = scales.astype(np.float32)
    weight_int8 = np.clip(np.round(weight / scales), -127, 127).astype(np.int8)
    
    # Compress (encoder prints are in C++ - can't suppress easily)
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
    
    # Progress every 20 layers (quieter)
    if (i + 1) % 20 == 0 or i == 0 or i == len(linear_layers) - 1:
        print(f"  Progress: {i+1}/{len(linear_layers)} layers...")

overall_ratio = total_original / total_compressed
print(f"  ✓ Compression complete!")
print(f"    Original: {total_original/1024**2:.1f} MB → Compressed: {total_compressed/1024**2:.1f} MB")
print(f"    Ratio: {overall_ratio:.2f}x")
print()

# Create compressed model (suppress debug prints)
print("  Creating compressed model...")

class CompressedLinear(torch.nn.Module):
    def __init__(self, original_module, compressed_data, decoder_handle, target_device='cuda', name=''):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.target_device = target_device
        self.name = name  # For error reporting
        
        numpy_dtype = compressed_data['dtype']
        torch_dtype = torch.float16 if numpy_dtype == np.float16 else torch.float32
        
        self.dtype = torch_dtype
        self.decoder = decoder_handle
        
        scale_np = compressed_data['scale']
        if isinstance(scale_np, np.ndarray):
            scale_tensor = torch.from_numpy(scale_np).to(torch_dtype).to(target_device)
            self.register_buffer('scale', scale_tensor)
        else:
            self.register_buffer('scale', torch.tensor(scale_np, dtype=torch_dtype, device=target_device))
        
        if original_module.bias is not None:
            self.register_buffer('bias', original_module.bias.data.clone())
        else:
            self.bias = None
        
        # Track if we've printed stats for this layer
        self._debug_printed = False
    
    def forward(self, x):
        import ctypes
        
        try:
            # GPU decode (decoder prints are in C++ - can't suppress)
            gpu_ptr, rows, cols, dtype = self.decoder.decode_layer_to_gpu(self.compressed)
            
            # Copy to PyTorch tensor
            weight_int8_gpu = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
            
            cudart = ctypes.CDLL('libcudart.so')
            cudart.cudaMemcpy(
                ctypes.c_void_p(weight_int8_gpu.data_ptr()),
                ctypes.c_void_p(gpu_ptr),
                ctypes.c_size_t(rows * cols),
                ctypes.c_int(1)
            )
            cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
            
            # Print stats ONCE per layer for first 3 layers only
            if not self._debug_printed and hasattr(self, '_layer_idx') and self._layer_idx < 3:
                print(f"    Layer {self._layer_idx} ({self.name}): scale range [{self.scale.min():.6f}, {self.scale.max():.6f}]")
                self._debug_printed = True
            
            # Dequantize
            weight_fp_unscaled = weight_int8_gpu.to(self.dtype)
            
            if self.scale.dim() == 1:
                scale_expanded = self.scale.view(-1, 1)
            else:
                scale_expanded = self.scale
            
            weight_fp = weight_fp_unscaled * scale_expanded
            weight_fp = weight_fp.reshape(self.shape)
            output = torch.nn.functional.linear(x, weight_fp, self.bias)
            
            del weight_fp, weight_fp_unscaled, weight_int8_gpu
            return output
            
        except Exception as e:
            print(f"\n❌ ERROR in layer '{self.name}': {e}")
            raise

# Replace layers
def replace_linear_with_compressed(module, compressed_weights, decoder):
    layer_idx = 0
    for name, child in list(module.named_children()):
        full_name = name
        if hasattr(module, '__module_name__'):
            full_name = module.__module_name__ + '.' + name
        
        if isinstance(child, torch.nn.Linear):
            for compressed_name, compressed_data in compressed_weights.items():
                if compressed_name.endswith('.' + name) or compressed_name == name:
                    compressed_layer = CompressedLinear(child, compressed_data, decoder, target_device=device, name=compressed_name)
                    compressed_layer._layer_idx = layer_idx
                    layer_idx += 1
                    setattr(module, name, compressed_layer)
                    break
        else:
            child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
            replace_linear_with_compressed(child, compressed_weights, decoder)

model.__module_name__ = ''
replace_linear_with_compressed(model, compressed_weights, decoder)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print(f"  ✓ Replaced {len(compressed_weights)} layers")
print()

# Run inference (decoder will still print - can't suppress C++ easily)
print(f"[3/3] Running compressed inference...")
print(f"  Note: Decoder messages below are from C++ (can't suppress)")
print(f"  Generating 10 tokens...")
print()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs_compressed = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    t_compressed = time.time() - t0

compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)
compressed_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

print()
print("="*80)
print("RESULTS")
print("="*80)
print()
print(f"Baseline:")
print(f"  Output: '{baseline_text}'")
print(f"  Time:   {t_baseline:.2f}s")
print(f"  VRAM:   {baseline_vram:.2f} GB")
print()
print(f"All {len(compressed_weights)} layers compressed:")
print(f"  Output: '{compressed_text}'")
print(f"  Time:   {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x slower)")
print(f"  VRAM:   {compressed_vram:.2f} GB")
print(f"  Ratio:  {overall_ratio:.2f}x")
print()

# Quality check
if baseline_text == compressed_text:
    print("✓ PERFECT - Outputs match!")
elif compressed_text[:25] == baseline_text[:25]:
    print("⚠ MINOR DIFF - First 25 chars match")
    print(f"  Expected: '{baseline_text}'")
    print(f"  Got:      '{compressed_text}'")
else:
    print("✗ MAJOR DIFF - Outputs diverge")
    print(f"  Expected: '{baseline_text}'")
    print(f"  Got:      '{compressed_text}'")
    
    # Show where they diverge
    for i, (c1, c2) in enumerate(zip(baseline_text, compressed_text)):
        if c1 != c2:
            print(f"  Diverges at position {i}: expected '{c1}' got '{c2}'")
            break

print()
print("="*80)
print("ACTIONABLE INFORMATION")
print("="*80)
print()

if "PERFECT" in locals():
    print("✓ Quality is perfect! Move to performance optimization.")
elif "MINOR" in locals():
    print("⚠ Minor quality issues detected:")
    print("  1. Run: python3 test_quantization_roundtrip.py")
    print("  2. Check if scales are reasonable")
    print("  3. Try skipping LM head layer")
elif "MAJOR" in locals():
    print("✗ Major quality issues detected:")
    print("  1. Run: python3 test_quantization_roundtrip.py")
    print("  2. Check scale precision (float16 vs float32)")
    print("  3. Verify dequantization math")
    print("  4. Try per-tensor quantization instead of per-channel")

print()

