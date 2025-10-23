#!/usr/bin/env python3
"""
Test: Compress FP16 weights directly (NO quantization to INT8)
Goal: Isolate whether quantization is the problem
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
print("TEST: FP16 COMPRESSION (NO QUANTIZATION)")
print("="*80)
print()
print("Strategy: Compress FP16 weights directly, skip INT8 quantization entirely")
print("If this works: Quantization was the problem")
print("If this fails: Something else is wrong")
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

# Reload and compress (SKIP QUANTIZATION)
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

all_linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

# Skip LM head AND embeddings (just to be safe)
linear_layers = [(n, m) for n, m in all_linear_layers 
                if 'lm_head' not in n.lower() and 'embed' not in n.lower()]

print(f"[2/3] Compressing {len(linear_layers)}/{len(all_linear_layers)} layers (FP16, no quantization)...")
skipped = [n for n, _ in all_linear_layers if 'lm_head' in n.lower() or 'embed' in n.lower()]
print(f"  Skipping: {len(skipped)} layers (embeddings + LM head)")
print()

encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()
compressed_weights = {}
total_original = 0
total_compressed = 0

for i, (name, module) in enumerate(linear_layers):
    weight = module.weight.data.cpu().numpy()
    
    # NO QUANTIZATION - Just convert FP16 to bytes and compress
    # Treat FP16 as 2-byte integers for compression
    weight_bytes = weight.astype(np.float16).tobytes()
    weight_as_int8 = np.frombuffer(weight_bytes, dtype=np.int8).reshape(weight.shape[0], weight.shape[1] * 2)
    
    # Compress the raw bytes
    compressed, ratio = encoder.encode_layer(weight_as_int8)
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'dtype': weight.dtype,
        'ratio': ratio,
        'is_fp16': True  # Flag to indicate no quantization
    }
    
    total_original += weight.nbytes
    total_compressed += len(compressed)
    
    if (i + 1) % 20 == 0 or i == 0 or i == len(linear_layers) - 1:
        print(f"  Progress: {i+1}/{len(linear_layers)}... (ratio so far: {total_original/total_compressed:.2f}x)")

overall_ratio = total_original / total_compressed
print(f"  ✓ Compressed {len(linear_layers)} layers")
print(f"  Original: {total_original/1024**2:.1f} MB → Compressed: {total_compressed/1024**2:.1f} MB")
print(f"  Ratio: {overall_ratio:.2f}x (FP16 compression, no quantization loss)")
print()

# Create compressed model
class CompressedLinearFP16(torch.nn.Module):
    """Compressed linear layer with NO quantization - direct FP16 compression"""
    def __init__(self, original_module, compressed_data, decoder_handle, target_device='cuda'):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.dtype = torch.float16
        self.decoder = decoder_handle
        
        if original_module.bias is not None:
            self.register_buffer('bias', original_module.bias.data.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        import ctypes
        
        # Decode compressed bytes
        gpu_ptr, rows, cols, _ = self.decoder.decode_layer_to_gpu(self.compressed)
        
        # The decoder returns INT8, but we stored FP16 as bytes
        # So we get (rows, cols*2) INT8 array
        weight_bytes_gpu = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
        
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaMemcpy(
            ctypes.c_void_p(weight_bytes_gpu.data_ptr()),
            ctypes.c_void_p(gpu_ptr),
            ctypes.c_size_t(rows * cols),
            ctypes.c_int(1)
        )
        cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
        
        # Convert bytes back to FP16 (pure GPU operation)
        # The INT8 array has shape (rows, cols*2) where cols*2 is the byte count  
        # Use view() to reinterpret INT8 bytes as FP16 without copying
        # First reshape to (rows * cols * 2,) then view as int16, then to float16
        weight_as_bytes = weight_bytes_gpu.flatten()
        # View pairs of int8 as int16, then cast to float16
        weight_as_int16 = weight_as_bytes.view(torch.int16)
        weight_fp16 = weight_as_int16.view(torch.float16).reshape(self.shape)
        
        output = torch.nn.functional.linear(x, weight_fp16, self.bias)
        
        del weight_bytes_gpu, weight_fp16
        return output

def replace_linear_with_compressed(module, compressed_weights, decoder):
    for name, child in list(module.named_children()):
        full_name = name
        if hasattr(module, '__module_name__'):
            full_name = module.__module_name__ + '.' + name
        
        if isinstance(child, torch.nn.Linear):
            for compressed_name, compressed_data in compressed_weights.items():
                if compressed_name.endswith('.' + name) or compressed_name == name:
                    if compressed_data.get('is_fp16', False):
                        setattr(module, name, CompressedLinearFP16(child, compressed_data, decoder, target_device=device))
                    break
        else:
            child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
            replace_linear_with_compressed(child, compressed_weights, decoder)

print("  Replacing layers...")
model.__module_name__ = ''
replace_linear_with_compressed(model, compressed_weights, decoder)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(f"  ✓ Replaced {len(compressed_weights)} layers (pure FP16 compression)")
print()

# Inference
print("[3/3] Running inference (FP16 compressed, no quantization)...")
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
print(f"Baseline (FP16, uncompressed):")
print(f"  '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s, VRAM: {baseline_vram:.2f} GB")
print()
print(f"FP16 compressed (NO quantization, {len(linear_layers)} layers):")
print(f"  '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x)")
print(f"  VRAM: {compressed_vram:.2f} GB")
print(f"  Ratio: {overall_ratio:.2f}x")
print()

# Quality check
if baseline_text == compressed_text:
    print("✓✓✓ PERFECT MATCH! ✓✓✓")
    print()
    print("CONCLUSION: Quantization was the problem!")
    print("  - FP16 compression works perfectly")
    print("  - INT8 quantization introduced errors")
    print("  - Can proceed with FP16/FP8 compression for production")
    print()
    print("Benefits of FP16 compression:")
    print(f"  ✓ Compression ratio: {overall_ratio:.2f}x")
    print(f"  ✓ Zero quality loss")
    print(f"  ✓ Simpler pipeline (no quantization math)")
    print()
    print("Next steps:")
    print("  1. Test FP8 compression (on Blackwell hardware)")
    print("  2. Optimize decompression performance")
    print("  3. Scale to full model")
elif compressed_text[:30] == baseline_text[:30]:
    print("⚠ MINOR DIFFERENCE")
    print("FP16 compression is better but not perfect.")
    print("May need different compression settings.")
else:
    print("✗ STILL BROKEN")
    print()
    print("CONCLUSION: Not quantization, something deeper")
    print("Possible issues:")
    print("  - Layer replacement logic wrong")
    print("  - Compression corrupting data somehow")
    print("  - GPU decode has subtle bugs")
    print("  - Need to skip more layers (all embeddings?)")
    
print()

