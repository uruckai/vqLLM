#!/usr/bin/env python3
"""
Real-world LLM inference test with batched compression
Uses TinyLlama and batched layer-level decode
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

from bindings_batched import BatchedEncoder, BatchedGPUDecoder

print("="*80)
print("BATCHED LOW-MEMORY INFERENCE TEST")
print("="*80)
print()

# Check GPU
if not BatchedGPUDecoder.is_available():
    print("❌ GPU decoder not available")
    sys.exit(1)

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
else:
    baseline_vram = 0
print()

# Compress layers
print("[4/6] Compressing Linear layers...")
encoder = BatchedEncoder(tile_size=256)
decoder = BatchedGPUDecoder()

compressed_weights = {}
total_original = 0
total_compressed = 0
compress_time = 0

num_to_compress = min(50, len(linear_layers))  # Compress 50 layers (cache will hold 10)
print(f"  Compressing {num_to_compress} layers (cache will hold 10)...")

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
    
    if (i + 1) % 10 == 0:
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

from collections import OrderedDict

class CompressedLinear(torch.nn.Module):
    """Linear layer that decompresses weights on-the-fly with LRU caching"""
    
    # Class-level cache shared across all CompressedLinear instances
    _weight_cache = OrderedDict()
    _cache_size = 10  # Cache 10 hot layers (out of 50 compressed)
    _cache_hits = 0
    _cache_misses = 0
    _layer_id_counter = 0
    
    def __init__(self, original_module, compressed_data, decoder_handle):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.scale = compressed_data['scale']
        
        # Convert numpy dtype to torch dtype
        numpy_dtype = compressed_data['dtype']
        if numpy_dtype == np.float16:
            self.torch_dtype = torch.float16
        elif numpy_dtype == np.float32:
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.float32  # fallback
        
        self.decoder = decoder_handle
        
        # Unique ID for this layer (for cache key)
        self.layer_id = CompressedLinear._layer_id_counter
        CompressedLinear._layer_id_counter += 1
        
        # Keep bias uncompressed
        if original_module.bias is not None:
            self.bias = original_module.bias
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Check cache first
        if self.layer_id in self._weight_cache:
            # Cache hit! Move to end (most recently used)
            self._weight_cache.move_to_end(self.layer_id)
            weight = self._weight_cache[self.layer_id].to(x.device)
            CompressedLinear._cache_hits += 1
        else:
            # Cache miss - decompress
            CompressedLinear._cache_misses += 1
            
            rows, cols = self.shape
            decoded_int8, decode_time = self.decoder.decode_layer(
                self.compressed, rows, cols
            )
            
            # Dequantize
            weight_fp = torch.from_numpy(decoded_int8.astype(np.float32) * self.scale)
            weight = weight_fp.to(dtype=self.torch_dtype, device=x.device)
            
            # Add to cache
            self._weight_cache[self.layer_id] = weight.cpu()  # Store on CPU to save VRAM
            
            # Evict oldest if cache is full
            if len(self._weight_cache) > self._cache_size:
                oldest_id = next(iter(self._weight_cache))
                evicted = self._weight_cache.pop(oldest_id)
                del evicted
        
        # Linear operation
        output = torch.nn.functional.linear(x, weight, self.bias)
        
        return output
    
    @classmethod
    def get_cache_stats(cls):
        """Return cache statistics"""
        total = cls._cache_hits + cls._cache_misses
        hit_rate = cls._cache_hits / total if total > 0 else 0
        return {
            'hits': cls._cache_hits,
            'misses': cls._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(cls._weight_cache),
            'max_cache_size': cls._cache_size
        }
    
    @classmethod
    def reset_cache_stats(cls):
        """Reset cache statistics"""
        cls._cache_hits = 0
        cls._cache_misses = 0

# Replace compressed layers
for name, compressed_data in compressed_weights.items():
    # Find module
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    # Get original
    original = getattr(parent, parts[-1])
    
    # Replace
    compressed_layer = CompressedLinear(original, compressed_data, decoder)
    setattr(parent, parts[-1], compressed_layer)

print(f"✓ Replaced {len(compressed_weights)} layers with compressed versions")
print()

# Compressed inference
print("[6/6] Running compressed inference...")
print(f"  Prompt: '{prompt}'")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    t0 = time.time()
    outputs_compressed = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    t_compressed = time.time() - t0

compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)
print(f"  Output: '{compressed_text}'")
print(f"  Time: {t_compressed:.2f}s")

if torch.cuda.is_available():
    compressed_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak VRAM: {compressed_vram:.2f} GB")
else:
    compressed_vram = 0

print()

# Get cache stats
cache_stats = CompressedLinear.get_cache_stats()

print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print()
print(f"Compression:")
print(f"  Layers compressed:    {len(compressed_weights)}/{len(linear_layers)}")
print(f"  Compression ratio:    {overall_ratio:.2f}x")
print(f"  Memory saved:         {(total_original - total_compressed)/1024**2:.1f} MB")
print()
print(f"Cache Performance:")
print(f"  Cache size:           {cache_stats['cache_size']}/{cache_stats['max_cache_size']} layers")
print(f"  Cache hits:           {cache_stats['hits']}")
print(f"  Cache misses:         {cache_stats['misses']}")
print(f"  Hit rate:             {cache_stats['hit_rate']*100:.1f}%")
print(f"  Decodes avoided:      {cache_stats['hits']} (saved ~{cache_stats['hits']*0.1:.1f}s)")
print()
print(f"Inference:")
print(f"  Baseline time:        {t_baseline:.2f}s")
print(f"  Compressed time:      {t_compressed:.2f}s")
print(f"  Slowdown:             {t_compressed/t_baseline:.2f}x")
print()
if baseline_vram > 0:
    print(f"VRAM:")
    print(f"  Baseline:             {baseline_vram:.2f} GB")
    print(f"  Compressed:           {compressed_vram:.2f} GB")
    print(f"  VRAM saved:           {baseline_vram - compressed_vram:.2f} GB")
    print()
print(f"Output match:")
print(f"  Baseline:  '{baseline_text}'")
print(f"  Compressed: '{compressed_text}'")
print(f"  Match: {baseline_text == compressed_text}")
print()
print("="*80)
print("✓ Test complete!")
print("="*80)

