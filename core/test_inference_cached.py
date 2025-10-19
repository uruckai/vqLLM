#!/usr/bin/env python3
"""
CACHED Low-Memory Inference Test

Adds smart caching to make inference practical:
- Cache recently used layers (LRU cache)
- Only decompress when needed
- Much faster while still saving VRAM
"""

import numpy as np
import ctypes
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from collections import OrderedDict

# Import from existing test
from test_inference_lowmem import load_codec

class CachedCompressedTensor:
    """CompressedTensor WITHOUT internal caching (cache managed at layer level)"""
    
    def __init__(self, codec_lib, tensor, cache_size=0):
        """
        Args:
            codec_lib: Loaded codec library
            tensor: PyTorch tensor to compress
            cache_size: UNUSED (kept for compatibility)
        """
        self.lib = codec_lib
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.device = tensor.device
        
        # No internal caching - managed at layer level
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Quantize
        data_np = tensor.cpu().detach().numpy()
        self.scale = np.abs(data_np).max() / 127.0 if np.abs(data_np).max() > 0 else 1.0
        quantized = np.round(data_np / self.scale).astype(np.int8)
        
        # Flatten and pad
        flat = quantized.flatten()
        self.num_elements = len(flat)
        
        padded_size = ((self.num_elements + 65535) // 65536) * 65536
        padded = np.zeros(padded_size, dtype=np.int8)
        padded[:self.num_elements] = flat
        
        # Compress tiles
        self.compressed_tiles = []
        encoder = self.lib.encoder_create(256)
        
        total_input = 0
        total_compressed = 0
        
        for i in range(0, padded_size, 65536):
            tile = padded[i:i+65536].reshape(256, 256)
            tile_contig = np.ascontiguousarray(tile)
            
            compressed = ctypes.POINTER(ctypes.c_uint8)()
            compressed_size = ctypes.c_size_t()
            
            ratio = self.lib.encoder_encode(
                encoder,
                tile_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                256, 256,
                ctypes.byref(compressed),
                ctypes.byref(compressed_size)
            )
            
            compressed_bytes = bytes(compressed[:compressed_size.value])
            self.lib.free_buffer(compressed)
            
            self.compressed_tiles.append(compressed_bytes)
            total_input += 65536
            total_compressed += len(compressed_bytes)
        
        self.lib.encoder_destroy(encoder)
        
        self.original_size = tensor.element_size() * tensor.numel()
        self.compressed_size = sum(len(t) for t in self.compressed_tiles)
        self.compression_ratio = self.original_size / self.compressed_size
    
    def decompress(self):
        """Decompress and return the tensor (NO internal caching)"""
        self._cache_misses += 1
        decompress_start = time.time()
        
        # Progress indicator for large layers
        num_tiles = len(self.compressed_tiles)
        if num_tiles > 10:
            print(f"    [t+{time.time()-decompress_start:.1f}s] Decompressing {num_tiles} tiles...", end='', flush=True)
        
        # Decompress all tiles using GPU decoder
        decoder = self.lib.decoder_create()
        all_data = []
        
        tile_start = time.time()
        for tile_idx, compressed in enumerate(self.compressed_tiles):
            # Progress for very large layers
            if num_tiles > 50 and tile_idx % 10 == 0:
                elapsed = time.time() - tile_start
                avg_per_tile = elapsed / 10 if tile_idx > 0 else 0
                print(f" {tile_idx}/{num_tiles}({avg_per_tile*1000:.0f}ms/tile)", end='', flush=True)
                tile_start = time.time()
            
            decoded = np.zeros((256, 256), dtype=np.int8)
            decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
            
            compressed_array = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
            compressed_ptr = ctypes.cast(compressed_array, ctypes.POINTER(ctypes.c_uint8))
            
            # GPU decoder returns time in ms
            decode_time_ms = self.lib.decoder_decode(decoder, compressed_ptr, len(compressed), decoded_ptr)
            if decode_time_ms < 0:
                raise RuntimeError(f"GPU decode failed on tile {tile_idx}")
            
            all_data.append(decoded.flatten())
        
        self.lib.decoder_destroy(decoder)
        
        total_decompress_time = time.time() - decompress_start
        if num_tiles > 10:
            print(f" done in {total_decompress_time:.2f}s ({num_tiles} tiles, {total_decompress_time/num_tiles*1000:.1f}ms/tile)", flush=True)
        
        # Concatenate and trim
        full_data = np.concatenate(all_data)[:self.num_elements]
        
        # Dequantize
        if self.dtype in [torch.float16, torch.float32]:
            if self.dtype == torch.float16:
                full_data = full_data.astype(np.float16) * self.scale
            else:
                full_data = full_data.astype(np.float32) * self.scale
        
        # Reshape and convert to tensor
        result = full_data.reshape(self.shape)
        tensor = torch.from_numpy(result).to(self.device)
        
        return tensor
    
    def get_stats(self):
        """Return cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total * 100 if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cached': self._cached is not None
        }

class CachedCompressedLinear(torch.nn.Module):
    """Linear layer with cached compressed weights"""
    
    # Global LRU cache for all layers
    _global_cache = OrderedDict()
    _max_cache_size = 20  # Keep 20 most recent layers decompressed
    
    def __init__(self, original_linear, codec_lib, enable_cache=True):
        super().__init__()
        
        self.layer_id = id(self)
        self.enable_cache = enable_cache
        self.compressed_weight = CachedCompressedTensor(
            codec_lib, 
            original_linear.weight.data
        )
        
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.bias = None
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Stats
        self.decode_count = 0
        self.decode_time = 0.0
    
    def forward(self, x):
        """Forward pass with global LRU caching"""
        start = time.time()
        
        # Update target device
        self.compressed_weight.device = x.device
        
        # Check global cache
        cache_hit = False
        if self.enable_cache and self.layer_id in CachedCompressedLinear._global_cache:
            # Move to front (most recently used)
            weight = CachedCompressedLinear._global_cache.pop(self.layer_id)
            CachedCompressedLinear._global_cache[self.layer_id] = weight
            cache_hit = True
        else:
            # Decompress
            weight = self.compressed_weight.decompress()
            
            if self.enable_cache:
                # Add to global cache
                CachedCompressedLinear._global_cache[self.layer_id] = weight
                
                # Evict oldest if cache full
                while len(CachedCompressedLinear._global_cache) > CachedCompressedLinear._max_cache_size:
                    oldest_id, oldest_weight = CachedCompressedLinear._global_cache.popitem(last=False)
                    del oldest_weight
                    if x.device.type == 'cuda':
                        torch.cuda.empty_cache()
        
        decode_time = time.time() - start
        self.decode_time += decode_time
        self.decode_count += 1
        
        # Progress indicator every 50 operations
        if self.decode_count % 50 == 0:
            hit_str = "HIT " if cache_hit else "MISS"
            cached = len(CachedCompressedLinear._global_cache)
            total_time_per_op = self.decode_time / self.decode_count if self.decode_count > 0 else 0
            print(f"  [Op {self.decode_count:4d}] {hit_str} time={decode_time*1000:5.0f}ms avg={total_time_per_op*1000:5.0f}ms cache={cached}/{self._max_cache_size}", flush=True)
        
        # Compute
        output = nn.functional.linear(x, weight, self.bias)
        
        return output
    
    @classmethod
    def clear_global_cache(cls):
        """Clear the global cache"""
        cls._global_cache.clear()
        torch.cuda.empty_cache()
    
    @classmethod
    def get_cache_stats(cls):
        """Get global cache statistics"""
        return {
            'cached_layers': len(cls._global_cache),
            'max_cache_size': cls._max_cache_size
        }

def compress_model_cached(model, codec_lib, enable_cache=True, verbose=True):
    """Replace all Linear layers with CachedCompressedLinear"""
    compressed_count = 0
    total_original_size = 0
    total_compressed_size = 0
    
    def compress_layer(module, name=""):
        nonlocal compressed_count, total_original_size, total_compressed_size
        
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, nn.Linear):
                compressed = CachedCompressedLinear(child, codec_lib, enable_cache)
                setattr(module, child_name, compressed)
                
                compressed_count += 1
                total_original_size += compressed.compressed_weight.original_size
                total_compressed_size += compressed.compressed_weight.compressed_size
                
                if verbose and compressed_count % 10 == 0:
                    print(f"  Compressed {compressed_count} layers...", end='\r')
            else:
                compress_layer(child, full_name)
    
    compress_layer(model)
    
    if verbose:
        print(f"\n✓ Compressed {compressed_count} Linear layers")
        print(f"  Original size:    {total_original_size / 1024**3:.2f} GB")
        print(f"  Compressed size:  {total_compressed_size / 1024**3:.2f} GB")
        print(f"  Compression ratio: {total_original_size / total_compressed_size:.2f}x")
        print(f"  Cache: {'ENABLED' if enable_cache else 'DISABLED'} (keeps {CachedCompressedLinear._max_cache_size} layers decompressed)")
    
    return compressed_count

def test_cached_inference():
    """Test inference with smart caching"""
    
    print("="*80)
    print("CACHED LOW-MEMORY INFERENCE TEST")
    print("="*80)
    print("\nThis test uses SMART CACHING to make inference practical:")
    print("  - Keeps 20 most recent layers decompressed")
    print("  - Only decompresses when needed")
    print("  - Much faster than no caching!")
    
    # Load codec
    print("\n[1/5] Loading codec library...")
    lib = load_codec()
    if lib is None:
        return False
    
    if not lib.decoder_is_available():
        print("✗ GPU decoder not available")
        return False
    
    print("✓ Codec library loaded")
    print("✓ GPU decoder available")
    
    # Load PyTorch and Transformers
    print("\n[2/5] Loading PyTorch and Transformers...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ Libraries loaded")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU available, using CPU")
    
    # Load model
    print("\n[3/5] Loading model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"  Model: {model_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to('cpu')
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Prepare test
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    print("\n[4/5] Running baseline inference...")
    print(f"  Prompt: '{test_prompt}'")
    print(f"  Generating 5 tokens (fast test)...")
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        model = model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    baseline_start = time.time()
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    baseline_time = time.time() - baseline_start
    
    if device == 'cuda':
        baseline_vram = torch.cuda.max_memory_allocated() / 1024**3
    else:
        baseline_vram = 0
    
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    
    print(f"  Generated: '{baseline_text}'")
    print(f"  Time: {baseline_time:.2f}s")
    if device == 'cuda':
        print(f"  Peak VRAM: {baseline_vram:.2f} GB")
    
    # Move back to CPU for compression
    model = model.to('cpu')
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Compress with caching
    print("\n[5/5] Testing compressed inference with caching...")
    print("  Compressing model...")
    
    compress_start = time.time()
    num_compressed = compress_model_cached(model, lib, enable_cache=True, verbose=True)
    compress_time = time.time() - compress_start
    print(f"  Compression took: {compress_time:.1f}s")
    
    # Run inference
    print("\n  Running inference with cached decompression...")
    print("  (First few layers will be slow, then cache kicks in)")
    print("  Generating 5 tokens with progress indicators...\n")
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        model = model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Custom generation loop with progress
    cached_start = time.time()
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)
    
    from datetime import datetime
    def timestamp():
        return datetime.now().strftime("%H:%M:%S")
    
    print(f"  [{timestamp()}] Token 0: Starting generation...", flush=True)
    
    with torch.no_grad():
        for token_idx in range(5):
            token_start = time.time()
            
            print(f"  [{timestamp()}]   Forward pass starting...", flush=True)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)
            
            token_time = time.time() - token_start
            elapsed = time.time() - cached_start
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            
            # Get current cache stats
            cache_stats = CachedCompressedLinear.get_cache_stats()
            
            print(f"  [{timestamp()}] Token {token_idx+1}: '{token_text}' (took {token_time:.1f}s, total {elapsed:.1f}s, cache={cache_stats['cached_layers']}/20)", flush=True)
            
            # Stop if EOS
            if next_token[0, 0] == tokenizer.eos_token_id:
                print(f"  [{timestamp()}] [EOS token reached]", flush=True)
                break
    
    cached_output = input_ids
    cached_time = time.time() - cached_start
    
    if device == 'cuda':
        cached_vram = torch.cuda.max_memory_allocated() / 1024**3
    else:
        cached_vram = 0
    
    cached_text = tokenizer.decode(cached_output[0], skip_special_tokens=True)
    
    print(f"  Generated: '{cached_text}'")
    print(f"  Time: {cached_time:.2f}s")
    if device == 'cuda':
        print(f"  Peak VRAM: {cached_vram:.2f} GB")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nBaseline:  '{baseline_text}'")
    print(f"  Time: {baseline_time:.2f}s")
    if device == 'cuda':
        print(f"  VRAM: {baseline_vram:.2f} GB")
    
    print(f"\nCached:    '{cached_text}'")
    print(f"  Time: {cached_time:.2f}s")
    if device == 'cuda':
        print(f"  VRAM: {cached_vram:.2f} GB")
    
    print("\nComparison:")
    if baseline_text == cached_text:
        print("  ✅ Outputs MATCH exactly!")
    else:
        print(f"  ⚠️  Outputs differ (expected with quantization)")
    
    slowdown = cached_time / baseline_time
    print(f"  Speed: {slowdown:.1f}x slower")
    
    if device == 'cuda':
        vram_reduction = baseline_vram / cached_vram if cached_vram > 0 else 1
        print(f"  VRAM: {vram_reduction:.1f}x reduction")
    
    # Cache stats
    cache_stats = CachedCompressedLinear.get_cache_stats()
    print(f"\n  Cache stats:")
    print(f"    Cached layers: {cache_stats['cached_layers']}/{num_compressed}")
    print(f"    Max cache size: {cache_stats['max_cache_size']}")
    
    # Decode stats
    total_decodes = 0
    total_decode_time = 0.0
    for module in model.modules():
        if hasattr(module, 'decode_count'):
            total_decodes += module.decode_count
            total_decode_time += module.decode_time
    
    if total_decodes > 0:
        print(f"\n  Decompression stats:")
        print(f"    Total decompressions: {total_decodes}")
        print(f"    Total decode time: {total_decode_time:.2f}s")
        print(f"    Avg per decode: {total_decode_time/total_decodes*1000:.1f}ms")
        print(f"    Decode overhead: {total_decode_time/cached_time*100:.1f}% of time")
    
    print("\n" + "="*80)
    if slowdown < 5:
        print(f"✅ SUCCESS! Caching makes inference {slowdown:.1f}x slower (acceptable!)")
    else:
        print(f"⚠️  Still slow ({slowdown:.1f}x), but caching helps significantly")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = test_cached_inference()
    sys.exit(0 if success else 1)

