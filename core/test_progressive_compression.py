#!/usr/bin/env python3
"""
Progressive layer compression test
Tests 1, 5, 10, and 20 compressed layers to find sweet spot
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_with_n_layers(n_layers, model, tokenizer, linear_layers, prompt, baseline_text, device):
    """Test compression with N layers"""
    print("\n" + "="*80)
    print(f"TESTING WITH {n_layers} COMPRESSED LAYERS")
    print("="*80 + "\n")
    
    # Reset model to original state
    # (We'll reload to avoid state issues)
    
    # Compress layers
    encoder = ZstdEncoder(compression_level=9)
    decoder = ZstdGPUDecoder()
    
    compressed_weights = {}
    total_original = 0
    total_compressed = 0
    
    num_to_compress = min(n_layers, len(linear_layers))
    print(f"Compressing {num_to_compress} layers...")
    
    for i, (name, module) in enumerate(linear_layers[:num_to_compress]):
        weight = module.weight.data.cpu().numpy()
        
        # Per-channel quantization
        scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
        scales = np.maximum(scales, 1e-8)
        scales = scales.astype(np.float32)
        weight_int8 = np.clip(np.round(weight / scales), -127, 127).astype(np.int8)
        
        # Compress
        compressed, ratio = encoder.encode_layer(weight_int8)
        
        # Store with copy to avoid aliasing
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
    
    overall_ratio = total_original / total_compressed
    print(f"  Compression ratio: {overall_ratio:.2f}x")
    print(f"  Original: {total_original/1024**2:.1f} MB → Compressed: {total_compressed/1024**2:.1f} MB")
    
    # Create compressed linear module
    class CompressedLinear(torch.nn.Module):
        def __init__(self, original_module, compressed_data, decoder_handle, target_device='cuda'):
            super().__init__()
            self.compressed = compressed_data['compressed']
            self.shape = compressed_data['shape']
            self.target_device = target_device
            
            numpy_dtype = compressed_data['dtype']
            if numpy_dtype == np.float16:
                torch_dtype = torch.float16
            elif numpy_dtype == np.float32:
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float32
            
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
        
        def forward(self, x):
            import ctypes
            
            # GPU decode
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
    
    # Replace layers
    def replace_linear_with_compressed(module, compressed_weights, decoder):
        for name, child in list(module.named_children()):
            full_name = name
            if hasattr(module, '__module_name__'):
                full_name = module.__module_name__ + '.' + name
            
            if isinstance(child, torch.nn.Linear):
                for compressed_name, compressed_data in compressed_weights.items():
                    if compressed_name.endswith('.' + name) or compressed_name == name:
                        compressed_layer = CompressedLinear(child, compressed_data, decoder, target_device=device)
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
    
    # Run inference
    print("Running compressed inference...")
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
    
    if torch.cuda.is_available():
        compressed_vram = torch.cuda.max_memory_allocated() / 1024**3
    else:
        compressed_vram = 0
    
    # Results
    print(f"\n  Output: '{compressed_text}'")
    print(f"  Time: {t_compressed:.2f}s")
    print(f"  VRAM: {compressed_vram:.2f} GB")
    
    # Quality check
    if compressed_text == baseline_text:
        quality = "✓ PERFECT"
    elif compressed_text[:20] == baseline_text[:20]:
        quality = "⚠ MINOR DIFF"
    else:
        quality = "✗ MAJOR DIFF"
    
    print(f"  Quality: {quality}")
    
    return {
        'layers': n_layers,
        'text': compressed_text,
        'time': t_compressed,
        'vram': compressed_vram,
        'ratio': overall_ratio,
        'quality': quality
    }

def main():
    print("="*80)
    print("PROGRESSIVE COMPRESSION TEST")
    print("="*80 + "\n")
    
    # Check GPU
    if not ZstdGPUDecoder.is_available():
        print("⚠️  GPU decoder not available!")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load model once
    print("Loading model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    prompt = "The capital of France is"
    
    # Run baseline
    print("\nRunning baseline inference...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((name, module))
    
    print(f"Found {len(linear_layers)} Linear layers")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        t_baseline = time.time() - t0
    
    baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    baseline_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    print(f"  Output: '{baseline_text}'")
    print(f"  Time: {t_baseline:.2f}s")
    print(f"  VRAM: {baseline_vram:.2f} GB")
    
    # Test progressive layer counts
    results = []
    for n in [1, 5, 10, 20]:
        # Reload model for each test to avoid state issues
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append((name, module))
        
        result = test_with_n_layers(n, model, tokenizer, linear_layers, prompt, baseline_text, device)
        results.append(result)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80 + "\n")
    print(f"Baseline: '{baseline_text}' ({t_baseline:.2f}s, {baseline_vram:.2f} GB)")
    print()
    print("Compressed results:")
    print(f"{'Layers':<8} {'Time':<10} {'VRAM':<10} {'Ratio':<8} {'Quality':<15} Output")
    print("-" * 80)
    for r in results:
        time_ratio = f"{r['time']/t_baseline:.1f}x" if t_baseline > 0 else "N/A"
        print(f"{r['layers']:<8} {r['time']:.2f}s ({time_ratio:<6}) {r['vram']:.2f} GB  {r['ratio']:.2f}x   {r['quality']:<15} '{r['text'][:40]}...'")

if __name__ == "__main__":
    main()

