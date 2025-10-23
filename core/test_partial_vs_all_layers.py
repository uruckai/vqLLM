#!/usr/bin/env python3
"""
Direct comparison: Partial layer compression vs All layers compressed
Tests if hybrid compressed/uncompressed is causing error amplification
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

def compress_and_test(model, linear_layers, tokenizer, prompt, baseline_text, device, num_layers_to_compress, test_name):
    """Helper function to compress N layers and test"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}\n")
    
    encoder = ZstdEncoder(compression_level=9)
    decoder = ZstdGPUDecoder()
    
    compressed_weights = {}
    total_original = 0
    total_compressed = 0
    
    layers_to_compress = linear_layers[:num_layers_to_compress] if num_layers_to_compress < len(linear_layers) else linear_layers
    
    print(f"Compressing {len(layers_to_compress)}/{len(linear_layers)} layers...")
    
    for i, (name, module) in enumerate(layers_to_compress):
        weight = module.weight.data.cpu().numpy()
        
        # Per-channel quantization
        scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
        scales = np.maximum(scales, 1e-8)
        scales = scales.astype(np.float32)
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
        
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(layers_to_compress)}...")
    
    overall_ratio = total_original / total_compressed
    print(f"  Compression ratio: {overall_ratio:.2f}x")
    
    # Create compressed linear
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
            gpu_ptr, rows, cols, dtype = self.decoder.decode_layer_to_gpu(self.compressed)
            weight_int8_gpu = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
            
            cudart = ctypes.CDLL('libcudart.so')
            cudart.cudaMemcpy(
                ctypes.c_void_p(weight_int8_gpu.data_ptr()),
                ctypes.c_void_p(gpu_ptr),
                ctypes.c_size_t(rows * cols),
                ctypes.c_int(1)
            )
            cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
            
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
    print(f"Running inference...")
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
    
    # Quality
    if compressed_text == baseline_text:
        quality = "✓ PERFECT"
    elif compressed_text[:20] == baseline_text[:20]:
        quality = "⚠ MINOR"
    else:
        quality = "✗ MAJOR"
    
    print(f"  Output: '{compressed_text}'")
    print(f"  Quality: {quality}")
    print(f"  Time: {t_compressed:.2f}s")
    print(f"  VRAM: {compressed_vram:.2f} GB")
    
    return {
        'text': compressed_text,
        'time': t_compressed,
        'vram': compressed_vram,
        'ratio': overall_ratio,
        'quality': quality,
        'num_layers': len(layers_to_compress)
    }

def main():
    print("="*80)
    print("PARTIAL vs ALL LAYERS COMPRESSION TEST")
    print("="*80)
    print()
    print("Hypothesis: Compressing ALL layers gives better quality than partial")
    print("Reason: Hybrid compressed/uncompressed may amplify quantization errors")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "The capital of France is"
    
    # Baseline
    print("[1/4] Running baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((name, module))
    
    print(f"  Found {len(linear_layers)} Linear layers")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        t0 = time.time()
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        t_baseline = time.time() - t0
    
    baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    baseline_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    print(f"  Output: '{baseline_text}'")
    print(f"  Time: {t_baseline:.2f}s")
    print(f"  VRAM: {baseline_vram:.2f} GB")
    
    results = []
    
    # Test 1: 20 layers compressed (partial)
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    
    result = compress_and_test(model, linear_layers, tokenizer, prompt, baseline_text, device, 20, "20 layers compressed (PARTIAL)")
    results.append(result)
    
    # Test 2: ALL layers compressed
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    
    result = compress_and_test(model, linear_layers, tokenizer, prompt, baseline_text, device, len(linear_layers), f"ALL {len(linear_layers)} layers compressed")
    results.append(result)
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print()
    print(f"Baseline: '{baseline_text}' ({t_baseline:.2f}s)")
    print()
    print(f"{'Test':<40} {'Quality':<15} {'Time':<12} {'Output'}")
    print("-" * 90)
    
    for i, r in enumerate(results):
        test_name = "20 layers (partial)" if i == 0 else f"ALL {r['num_layers']} layers"
        time_str = f"{r['time']:.2f}s ({r['time']/t_baseline:.1f}x)"
        print(f"{test_name:<40} {r['quality']:<15} {time_str:<12} '{r['text'][:40]}...'")
    
    print()
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()
    
    partial_quality = results[0]['quality']
    all_quality = results[1]['quality']
    
    if "PERFECT" in all_quality or "MINOR" in all_quality:
        if "MAJOR" in partial_quality:
            print("✓ HYPOTHESIS CONFIRMED!")
            print()
            print("Compressing ALL layers gives BETTER quality than partial compression.")
            print()
            print("Why: The hybrid approach creates error amplification:")
            print("  - Quantized early layers introduce errors")
            print("  - Uncompressed later layers amplify those errors")
            print("  - With all layers quantized, errors are more uniform")
            print()
            print("Recommendation: Compress ALL layers for production!")
        else:
            print("✓ BOTH APPROACHES WORK!")
            print()
            print("Both partial and full compression give good quality.")
            print("This means quantization is working correctly in both cases.")
        print()
        print("Next steps:")
        print("  1. Optimize performance (batch decompress)")
        print("  2. Test on longer sequences")
        print("  3. Profile bottlenecks")
    else:
        print("⚠ HYPOTHESIS INCONCLUSIVE")
        print()
        print("Both approaches show quality issues. This suggests:")
        print("  - Quantization method needs tuning")
        print("  - Per-channel scales may need adjustment")
        print("  - Consider mixed precision or adaptive quantization")
        print()
        print("Run: python3 test_quantization_debug.py for detailed analysis")
    
    print()

if __name__ == "__main__":
    main()

