#!/usr/bin/env python3
"""
Test: Compress just 1-5 layers to isolate when corruption starts
Goal: Find if bug is immediate (layer replacement) or cumulative (error propagation)
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("MINIMAL LAYERS TEST - Find Where Corruption Starts")
print("="*80)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Baseline
print("Baseline:")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  '{baseline_text}'")
print()

# Test with increasing layer counts
for num_layers in [1, 3, 5, 10]:
    print(f"Testing with {num_layers} compressed layers...")
    
    # Reload model
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    
    all_linear = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    
    # Only compress ATTENTION layers (skip embeddings, LM head, MLP)
    # Start with just attention Q/K/V projections
    attention_layers = [(n, m) for n, m in all_linear if any(x in n for x in ['q_proj', 'k_proj', 'v_proj'])]
    layers_to_compress = attention_layers[:num_layers]
    
    print(f"  Compressing: {[n for n, _ in layers_to_compress]}")
    
    encoder = ZstdEncoder(compression_level=9)
    decoder = ZstdGPUDecoder()
    compressed_weights = {}
    
    for name, module in layers_to_compress:
        weight = module.weight.data.cpu().numpy()
        
        # Use INT8 quantization (simpler, known to work in isolation)
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
    
    # Replace layers
    class CompressedLinear(torch.nn.Module):
        def __init__(self, original_module, compressed_data, decoder_handle, target_device='cuda'):
            super().__init__()
            self.compressed = compressed_data['compressed']
            self.shape = compressed_data['shape']
            self.dtype = torch.float16
            self.decoder = decoder_handle
            
            scale_np = compressed_data['scale']
            scale_tensor = torch.from_numpy(scale_np).to(self.dtype).to(target_device)
            self.register_buffer('scale', scale_tensor)
            
            if original_module.bias is not None:
                self.register_buffer('bias', original_module.bias.data.clone())
            else:
                self.bias = None
        
        def forward(self, x):
            import ctypes
            
            # Decompress
            gpu_ptr, rows, cols, _ = self.decoder.decode_layer_to_gpu(self.compressed)
            
            # Verify dimensions match expectations
            if rows != self.shape[0] or cols != self.shape[1]:
                print(f"    ⚠ DIMENSION MISMATCH: expected {self.shape}, got ({rows}, {cols})")
            
            weight_int8_gpu = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
            
            cudart = ctypes.CDLL('libcudart.so')
            cudart.cudaMemcpy(
                ctypes.c_void_p(weight_int8_gpu.data_ptr()),
                ctypes.c_void_p(gpu_ptr),
                ctypes.c_size_t(rows * cols),
                ctypes.c_int(1)
            )
            cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
            
            # Verify data range
            if not hasattr(self, '_data_checked'):
                int8_min, int8_max = weight_int8_gpu.min().item(), weight_int8_gpu.max().item()
                print(f"    INT8 range: [{int8_min}, {int8_max}]")
                if int8_min == 0 and int8_max == 0:
                    print(f"    ⚠ ALL ZEROS - DECOMPRESSION FAILED!")
                self._data_checked = True
            
            # Dequantize
            weight_fp = weight_int8_gpu.to(self.dtype) * self.scale.view(-1, 1)
            weight_fp = weight_fp.reshape(self.shape)
            
            # Verify output range
            if not hasattr(self, '_fp_checked'):
                fp_min, fp_max = weight_fp.min().item(), weight_fp.max().item()
                print(f"    FP16 range: [{fp_min:.6f}, {fp_max:.6f}]")
                self._fp_checked = True
            
            output = torch.nn.functional.linear(x, weight_fp, self.bias)
            
            del weight_fp, weight_int8_gpu
            return output
    
    def replace_linear_with_compressed(module, compressed_weights, decoder):
        for name, child in list(module.named_children()):
            full_name = name
            if hasattr(module, '__module_name__'):
                full_name = module.__module_name__ + '.' + name
            
            if isinstance(child, torch.nn.Linear):
                for compressed_name in compressed_weights:
                    if compressed_name.endswith('.' + name) or compressed_name == name:
                        setattr(module, name, CompressedLinear(child, compressed_weights[compressed_name], decoder, target_device=device))
                        break
            else:
                child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
                replace_linear_with_compressed(child, compressed_weights, decoder)
    
    model.__module_name__ = ''
    replace_linear_with_compressed(model, compressed_weights, decoder)
    torch.cuda.empty_cache()
    
    # Test inference
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    
    compressed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Check quality
    if compressed_text == baseline_text:
        status = "✓ PERFECT"
    elif compressed_text[:20] == baseline_text[:20]:
        status = "⚠ MINOR"
    else:
        status = "✗ BROKEN"
    
    print(f"  Result: {status}")
    print(f"  Output: '{compressed_text}'")
    print()

print("="*80)
print("ANALYSIS")
print("="*80)
print()
print("If corruption appears at 1 layer:")
print("  → Bug in decompression or layer replacement")
print("  → Check dimension mismatches, data ranges")
print()
print("If corruption appears after N layers:")
print("  → Error amplification (even with no quantization!)")
print("  → May need ALL layers compressed (not hybrid)")
print()

