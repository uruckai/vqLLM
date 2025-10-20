#!/usr/bin/env python3
"""
Real-world LLM inference test with Zstd compression
Uses TinyLlama and GPU-accelerated Zstd decode via nvCOMP
"""
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import numpy as np
import torch
import sys
import time
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from bindings_zstd import ZstdEncoder, ZstdGPUDecoder

print("="*80)
print("ZSTD LOW-MEMORY INFERENCE TEST")
print("="*80)
print()

# Check GPU
if not ZstdGPUDecoder.is_available():
    print("⚠️  GPU decoder not available - will use CPU fallback")
else:
    print("✓ GPU decoder available")
print()

print("[1/6] Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"  Model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

print("✓ Model loaded")
print()

# Find Linear layers
print("[2/6] Finding Linear layers...")
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append((name, module))
print(f"✓ Found {len(linear_layers)} Linear layers")
print()

# Baseline inference
print("[3/6] Running baseline inference...")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(f"  Prompt: '{prompt}'")

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
print(f"  Output: '{baseline_text}'")
print(f"  Time: {t_baseline:.2f}s")

# Check VRAM
if torch.cuda.is_available():
    baseline_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak VRAM: {baseline_vram:.2f} GB")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
else:
    baseline_vram = 0
print()

# Compress layers
print("[4/6] Compressing Linear layers...")
encoder = ZstdEncoder(compression_level=9)  # Balanced compression
decoder = ZstdGPUDecoder()

compressed_weights = {}
total_original = 0
total_compressed = 0
compress_time = 0

num_to_compress = min(20, len(linear_layers))  # Compress 20 layers - we have 29GB free!
print(f"  Compressing {num_to_compress} layers...")

for i, (name, module) in enumerate(linear_layers[:num_to_compress]):
    # Get weight
    weight = module.weight.data.cpu().numpy()
    
    # Quantize to int8
    w_min, w_max = weight.min(), weight.max()
    scale = max(abs(w_min), abs(w_max)) / 127.0
    weight_int8 = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    
    # Compress
    t0 = time.time()
    compressed, ratio = encoder.encode_layer(weight_int8)
    compress_time += time.time() - t0
    
    # Store
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'scale': scale,
        'dtype': weight.dtype,
        'ratio': ratio
    }
    
    total_original += weight.nbytes
    total_compressed += len(compressed)
    
    if (i + 1) % 5 == 0:
        print(f"    {i+1}/{num_to_compress} layers compressed...")

overall_ratio = total_original / total_compressed
print(f"✓ Compressed {num_to_compress} layers")
print(f"  Original size:    {total_original/1024**2:.1f} MB")
print(f"  Compressed size:  {total_compressed/1024**2:.1f} MB")
print(f"  Compression ratio: {overall_ratio:.2f}x")
print(f"  Compression time: {compress_time:.2f}s")
print()

# Replace layers with compressed versions
print("[5/6] Creating compressed model...")

class CompressedLinear(torch.nn.Module):
    """Linear layer that decompresses weights once and keeps them on GPU"""
    
    def __init__(self, original_module, compressed_data, decoder_handle, target_device='cuda'):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.scale = compressed_data['scale']
        self.target_device = target_device
        
        # Convert numpy dtype to torch dtype
        numpy_dtype = compressed_data['dtype']
        if numpy_dtype == np.float16:
            torch_dtype = torch.float16
        elif numpy_dtype == np.float32:
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float32
        
        self.dtype = torch_dtype
        self.decoder = decoder_handle
        
        # Keep bias if present
        if original_module.bias is not None:
            self.register_buffer('bias', original_module.bias.data.clone())
        else:
            self.bias = None
        
        # Cache decompressed weight on GPU (not CPU!)
        self._cached_weight_gpu = None
    
    def decompress_to_gpu(self):
        """Decompress once and store on GPU"""
        if self._cached_weight_gpu is None:
            # Decompress on CPU/GPU (via nvCOMP)
            weight_int8 = self.decoder.decode_layer(self.compressed)
            weight_float = weight_int8.astype(np.float32) * self.scale
            
            # Convert to tensor on CPU first
            weight_tensor = torch.from_numpy(weight_float).to(self.dtype)
            
            # Free NumPy arrays immediately before GPU transfer
            del weight_float
            del weight_int8
            
            # Move to GPU with explicit error handling
            try:
                self._cached_weight_gpu = weight_tensor.to(self.target_device)
            except RuntimeError as e:
                # Print debug info if OOM
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"\n  OOM during decompress!")
                    print(f"    Weight size: {weight_tensor.nbytes / 1024**2:.1f} MB")
                    print(f"    Allocated: {allocated:.2f} GB")
                    print(f"    Reserved: {reserved:.2f} GB")
                raise
            finally:
                # Always free CPU tensor
                del weight_tensor
                
            # Aggressively free memory after each layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def forward(self, x):
        # Ensure weight is decompressed and on GPU
        if self._cached_weight_gpu is None:
            self.decompress_to_gpu()
        
        # Use cached GPU weight directly (no transfer!)
        output = torch.nn.functional.linear(x, self._cached_weight_gpu, self.bias)
        
        return output

# Replace layers
def replace_linear_with_compressed(module, compressed_weights, decoder):
    """Recursively replace Linear layers with CompressedLinear"""
    for name, child in list(module.named_children()):
        full_name = name
        # Build full qualified name
        if hasattr(module, '__module_name__'):
            full_name = module.__module_name__ + '.' + name
        
        if isinstance(child, torch.nn.Linear):
            # Find matching compressed weight
            for compressed_name, compressed_data in compressed_weights.items():
                if compressed_name.endswith('.' + name) or compressed_name == name:
                    # Replace with compressed version (will decompress to GPU later)
                    compressed_layer = CompressedLinear(child, compressed_data, decoder, target_device=device)
                    setattr(module, name, compressed_layer)
                    break
        else:
            # Recursive replacement
            child.__module_name__ = full_name if not hasattr(module, '__module_name__') else module.__module_name__ + '.' + name
            replace_linear_with_compressed(child, compressed_weights, decoder)

# Track which layers got replaced
replaced_count = 0
for name, module in model.named_modules():
    if name in compressed_weights:
        replaced_count += 1

print(f"  Will replace {len(compressed_weights)} layers")

# Do replacement
model.__module_name__ = ''
replace_linear_with_compressed(model, compressed_weights, decoder)

print(f"✓ Model ready with compressed layers")
print()

# Pre-warm the cache: decompress all layers to GPU
print("[5.5/6] Decompressing layers to GPU...")
print("  This happens once - then weights stay on GPU")

# Show initial memory state
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  Starting GPU memory: {initial_mem:.2f} GB")

warmup_count = 0
decompress_time = 0
import gc

for name, module in model.named_modules():
    if isinstance(module, CompressedLinear):
        # Show memory before
        if torch.cuda.is_available() and warmup_count == 0:
            mem_before = torch.cuda.memory_allocated() / 1024**3
            print(f"  Memory before first decompress: {mem_before:.2f} GB")
        
        t0 = time.time()
        module.decompress_to_gpu()
        decompress_time += time.time() - t0
        warmup_count += 1
        
        # Force garbage collection after each layer
        gc.collect()
        
        if warmup_count % 5 == 0:
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / 1024**3
                print(f"    {warmup_count}/{num_to_compress} layers decompressed (GPU: {current_mem:.2f} GB)")
            else:
                print(f"    {warmup_count}/{num_to_compress} layers decompressed")

print(f"✓ {warmup_count} layers decompressed to GPU in {decompress_time:.2f}s")
if torch.cuda.is_available():
    final_mem = torch.cuda.memory_allocated() / 1024**3
    added_mem = final_mem - initial_mem
    print(f"  GPU memory now: {final_mem:.2f} GB (+{added_mem:.2f} GB for {warmup_count} layers)")
    print(f"  Average per layer: {added_mem * 1024 / warmup_count:.1f} MB")
    torch.cuda.empty_cache()  # Clean up any temporary allocations
print()

# Run compressed inference
print("[6/6] Running compressed inference...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Aggressively clear memory before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    import gc
    gc.collect()

# Monitor memory during generation
print("  Generating tokens...")
with torch.no_grad():
    t0 = time.time()
    
    # Print memory before
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"  Memory before generation: {mem_before:.2f} GB")
    
    try:
        outputs_compressed = model.generate(
            **inputs,
            max_new_tokens=10,  # Back to 10 now that we know we have memory
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True  # Re-enable cache since we have plenty of VRAM
        )
        t_compressed = time.time() - t0
        
        # Print memory after
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Memory after generation: {mem_after:.2f} GB")
            print(f"  Memory peak: {mem_peak:.2f} GB")
            print(f"  Memory delta: +{mem_after - mem_before:.2f} GB")
    except RuntimeError as e:
        print(f"\n  ❌ Error during generation: {e}")
        if torch.cuda.is_available():
            mem_error = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Memory at error: {mem_error:.2f} GB")
            print(f"  Peak memory: {mem_peak:.2f} GB")
        raise

compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)
print(f"  Output: '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s")

# Check VRAM
if torch.cuda.is_available():
    compressed_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak VRAM: {compressed_vram:.2f} GB")
else:
    compressed_vram = 0

print()
print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print()
print(f"Baseline:")
print(f"  Time: {t_baseline:.2f}s")
print(f"  VRAM: {baseline_vram:.2f} GB")
print(f"  Output: '{baseline_text}'")
print()
print(f"Compressed (Zstd):")
print(f"  Time: {t_compressed:.2f}s ({t_compressed/t_baseline:.1f}x slower)")
print(f"  VRAM: {compressed_vram:.2f} GB ({baseline_vram/compressed_vram:.2f}x reduction)")
print(f"  Compression: {overall_ratio:.2f}x")
print(f"  Compressed layers: {num_to_compress}/{len(linear_layers)}")
print(f"  Output: '{compressed_text}'")
print()

# Check output quality
if baseline_text == compressed_text:
    print("✓ Output matches baseline (perfect reconstruction)")
else:
    print("⚠️  Output differs from baseline (quantization artifacts)")
    print(f"  Baseline:   '{baseline_text}'")
    print(f"  Compressed: '{compressed_text}'")

print()
print("✓ Test complete!")

