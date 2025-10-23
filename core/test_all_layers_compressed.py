#!/usr/bin/env python3
"""
Test with ALL layers compressed - might actually be better than partial!
Theory: Hybrid compressed/uncompressed creates error amplification
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

print("="*80)
print("ALL LAYERS COMPRESSED TEST")
print("="*80)
print()
print("Theory: Compressing ALL layers might give better quality than partial")
print("Reason: Uniform quantization error vs mixed precision error amplification")
print()

# Check GPU
if not ZstdGPUDecoder.is_available():
    print("⚠️  GPU decoder not available!")
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load model
print("Loading model...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "The capital of France is"

# Baseline inference
print("\n[1/3] Running baseline (uncompressed)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

# Find all Linear layers
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append((name, module))

print(f"  Found {len(linear_layers)} Linear layers")

# Get layer types
embedding_layers = [n for n, _ in linear_layers if 'embed' in n.lower()]
attention_layers = [n for n, _ in linear_layers if any(x in n for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])]
mlp_layers = [n for n, _ in linear_layers if any(x in n for x in ['gate_proj', 'up_proj', 'down_proj'])]
lm_head_layers = [n for n, _ in linear_layers if 'lm_head' in n]
other_layers = [n for n, _ in linear_layers if n not in embedding_layers + attention_layers + mlp_layers + lm_head_layers]

print(f"  - Embedding layers: {len(embedding_layers)}")
print(f"  - Attention layers: {len(attention_layers)}")
print(f"  - MLP layers: {len(mlp_layers)}")
print(f"  - LM head layers: {len(lm_head_layers)}")
print(f"  - Other layers: {len(other_layers)}")

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

# Reload model for compression test
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n[2/3] Compressing ALL layers...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

# Rebuild linear_layers list
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append((name, module))

encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()

compressed_weights = {}
total_original = 0
total_compressed = 0
compress_time = 0

# Compress ALL layers (or all except embeddings/lm_head for safety)
# Option 1: Compress everything
layers_to_compress = linear_layers

# Option 2: Skip embeddings/lm_head (uncomment if needed)
# layers_to_compress = [(n, m) for n, m in linear_layers if n not in embedding_layers + lm_head_layers]

print(f"  Compressing {len(layers_to_compress)}/{len(linear_layers)} layers...")
print(f"  (This may take a few minutes...)")

for i, (name, module) in enumerate(layers_to_compress):
    weight = module.weight.data.cpu().numpy()
    
    # Per-channel quantization
    scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8)
    scales = scales.astype(np.float32)
    weight_int8 = np.clip(np.round(weight / scales), -127, 127).astype(np.int8)
    
    # Compress
    t0 = time.time()
    compressed, ratio = encoder.encode_layer(weight_int8)
    compress_time += time.time() - t0
    
    # Store
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
    
    if (i + 1) % 10 == 0 or i == 0:
        print(f"    Progress: {i+1}/{len(layers_to_compress)} layers...")

overall_ratio = total_original / total_compressed
print(f"\n  ✓ Compression complete!")
print(f"  Original size:    {total_original/1024**2:.1f} MB")
print(f"  Compressed size:  {total_compressed/1024**2:.1f} MB")
print(f"  Compression ratio: {overall_ratio:.2f}x")
print(f"  Compression time: {compress_time:.2f}s")

# Create compressed model
print("\n  Creating compressed model...")

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

print(f"  ✓ Replaced {len(compressed_weights)} layers with compressed versions")

# Run inference
print("\n[3/3] Running compressed inference (all layers)...")
print("  This will be slower due to per-token decompression...")
print("  (Expected: ~1-2 minutes for 10 tokens)\n")

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

print(f"  Output: '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x slower)")
print(f"  VRAM: {compressed_vram:.2f} GB")

# Results
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)
print()
print("Baseline (uncompressed):")
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

# Quality assessment
if baseline_text == compressed_text:
    quality = "✓ PERFECT - Outputs match exactly!"
    print(f"Quality: {quality}")
elif compressed_text[:20] == baseline_text[:20]:
    quality = "⚠ MINOR DIFFERENCE - First 20 chars match"
    print(f"Quality: {quality}")
    print(f"  Expected: '{baseline_text}'")
    print(f"  Got:      '{compressed_text}'")
else:
    quality = "✗ MAJOR DIFFERENCE - Significant mismatch"
    print(f"Quality: {quality}")
    print(f"  Expected: '{baseline_text}'")
    print(f"  Got:      '{compressed_text}'")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
if "PERFECT" in quality or "MINOR" in quality:
    print("✓ Compressing all layers works!")
    print()
    print("This suggests the hybrid approach (some compressed, some not) may")
    print("have been causing error amplification. With uniform quantization")
    print("across all layers, the model handles it better.")
    print()
    print("Next steps:")
    print("  1. Compare this to partial compression (1, 5, 10, 20 layers)")
    print("  2. Optimize performance (batch decompress, fused kernels)")
    print("  3. Test on longer sequences")
else:
    print("⚠ Still seeing quality issues even with all layers compressed.")
    print()
    print("This suggests the quantization method itself may need tuning:")
    print("  - Try different scales (per-channel vs per-tensor)")
    print("  - Try symmetric vs asymmetric quantization")
    print("  - Consider mixed precision (FP8 for some layers)")
    print()
    print("Run: python3 test_quantization_debug.py for detailed analysis")

print()

