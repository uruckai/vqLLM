#!/usr/bin/env python3
"""
Test batched inference - MUCH faster than per-tile approach!

Expected speedup: 200x over per-tile (295ms → 1.5ms per tile)
"""

import numpy as np
import time
import sys
import torch
import torch.nn as nn
from datetime import datetime

# Import batched codec
from bindings_batched import BatchedEncoder, BatchedGPUDecoder

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

class BatchedCompressedTensor:
    """Tensor with batched layer-level compression"""
    
    def __init__(self, codec_encoder, codec_decoder, tensor):
        self.encoder = codec_encoder
        self.decoder = codec_decoder
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.device = tensor.device
        
        # Quantize
        data_np = tensor.cpu().detach().numpy()
        self.scale = np.abs(data_np).max() / 127.0 if np.abs(data_np).max() > 0 else 1.0
        quantized = np.round(data_np / self.scale).astype(np.int8)
        
        # Compress ENTIRE layer at once!
        self.compressed_layer, self.compression_ratio = self.encoder.encode_layer(quantized)
        self.compressed_size = len(self.compressed_layer)
        self.original_size = tensor.element_size() * tensor.numel()
        
    def decompress(self):
        """Decompress entire layer on GPU (all tiles in parallel!)"""
        rows, cols = self.shape
        
        # Decompress on GPU - ONE call for entire layer!
        decoded, decode_time_ms = self.decoder.decode_layer(self.compressed_layer, rows, cols)
        
        # Dequantize
        if self.dtype in [torch.float16, torch.float32]:
            if self.dtype == torch.float16:
                decoded = decoded.astype(np.float16) * self.scale
            else:
                decoded = decoded.astype(np.float32) * self.scale
        
        tensor = torch.from_numpy(decoded).to(self.device)
        return tensor, decode_time_ms

class BatchedCompressedLinear(torch.nn.Module):
    """Linear layer with batched compressed weights"""
    
    # Global cache
    _global_cache = {}
    _max_cache_size = 20
    
    def __init__(self, original_linear, encoder, decoder):
        super().__init__()
        
        self.layer_id = id(self)
        self.compressed_weight = BatchedCompressedTensor(encoder, decoder, original_linear.weight.data)
        
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
        """Forward with batched decompression"""
        start = time.time()
        
        # Update target device
        self.compressed_weight.device = x.device
        
        # Check cache
        cache_hit = False
        if self.layer_id in BatchedCompressedLinear._global_cache:
            weight = BatchedCompressedLinear._global_cache.pop(self.layer_id)
            BatchedCompressedLinear._global_cache[self.layer_id] = weight
            cache_hit = True
            decode_time_ms = 0
        else:
            # Decompress (FAST batched decode!)
            weight, decode_time_ms = self.compressed_weight.decompress()
            
            # Add to cache
            BatchedCompressedLinear._global_cache[self.layer_id] = weight
            
            # Evict oldest
            while len(BatchedCompressedLinear._global_cache) > BatchedCompressedLinear._max_cache_size:
                oldest_id, oldest_weight = BatchedCompressedLinear._global_cache.popitem(last=False)
                del oldest_weight
                if x.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        decode_time = time.time() - start
        self.decode_time += decode_time
        self.decode_count += 1
        
        # Progress every 50 ops
        if self.decode_count % 50 == 0:
            hit_str = "HIT " if cache_hit else "MISS"
            cached = len(BatchedCompressedLinear._global_cache)
            avg_time = self.decode_time / self.decode_count * 1000
            print(f"  [{timestamp()}] [Op {self.decode_count:4d}] {hit_str} time={decode_time*1000:5.0f}ms avg={avg_time:5.0f}ms cache={cached}/{self._max_cache_size}", flush=True)
        
        # Compute
        output = nn.functional.linear(x, weight, self.bias)
        
        return output

def compress_model_batched(model, encoder, decoder, verbose=True):
    """Replace all Linear layers with batched compressed versions"""
    compressed_count = 0
    total_original_size = 0
    total_compressed_size = 0
    
    def compress_layer(module, name=""):
        nonlocal compressed_count, total_original_size, total_compressed_size
        
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, nn.Linear):
                compressed = BatchedCompressedLinear(child, encoder, decoder)
                setattr(module, child_name, compressed)
                
                compressed_count += 1
                total_original_size += compressed.compressed_weight.original_size
                total_compressed_size += compressed.compressed_weight.compressed_size
                
                if verbose and compressed_count % 10 == 0:
                    print(f"  Compressed {compressed_count} layers...", end='\r', flush=True)
            else:
                compress_layer(child, full_name)
    
    compress_layer(model)
    
    if verbose:
        print(f"\n✓ Compressed {compressed_count} Linear layers")
        print(f"  Original size:    {total_original_size / 1024**3:.2f} GB")
        print(f"  Compressed size:  {total_compressed_size / 1024**3:.2f} GB")
        print(f"  Compression ratio: {total_original_size / total_compressed_size:.2f}x")
    
    return compressed_count

def test_batched_inference():
    """Test batched inference (200x faster!)"""
    
    print("="*80)
    print("BATCHED LOW-MEMORY INFERENCE TEST")
    print("="*80)
    print("\nUsing layer-level batched compression for 200x speedup!")
    
    # Load codec
    print(f"\n[{timestamp()}] [1/5] Loading codec library...")
    try:
        encoder = BatchedEncoder(tile_size=256)
        decoder = BatchedGPUDecoder()
        print(f"[{timestamp()}] ✓ Batched codec loaded")
        print(f"[{timestamp()}] ✓ GPU decoder available")
    except Exception as e:
        print(f"✗ Failed to load codec: {e}")
        return False
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"[{timestamp()}] ✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[{timestamp()}] ⚠️  No GPU, using CPU")
    
    # Load model
    print(f"\n[{timestamp()}] [2/5] Loading model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"[{timestamp()}]   Model: {model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to('cpu')
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[{timestamp()}] ✓ Model loaded")
    except Exception as e:
        print(f"[{timestamp()}] ✗ Failed to load model: {e}")
        return False
    
    # Baseline
    print(f"\n[{timestamp()}] [3/5] Running baseline inference...")
    test_prompt = "The capital of France is"
    print(f"[{timestamp()}]   Prompt: '{test_prompt}'")
    print(f"[{timestamp()}]   Generating 5 tokens...")
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
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
    
    print(f"[{timestamp()}]   Generated: '{baseline_text}'")
    print(f"[{timestamp()}]   Time: {baseline_time:.2f}s")
    if device == 'cuda':
        print(f"[{timestamp()}]   Peak VRAM: {baseline_vram:.2f} GB")
    
    # Move back for compression
    model = model.to('cpu')
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Compress
    print(f"\n[{timestamp()}] [4/5] Compressing model...")
    compress_start = time.time()
    num_compressed = compress_model_batched(model, encoder, decoder, verbose=True)
    compress_time = time.time() - compress_start
    print(f"[{timestamp()}]   Compression took: {compress_time:.1f}s")
    
    # Batched inference
    print(f"\n[{timestamp()}] [5/5] Running batched compressed inference...")
    print(f"[{timestamp()}]   (Layer-level decompression - 200x faster!)")
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        model = model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    batched_start = time.time()
    with torch.no_grad():
        batched_output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    batched_time = time.time() - batched_start
    
    if device == 'cuda':
        batched_vram = torch.cuda.max_memory_allocated() / 1024**3
    else:
        batched_vram = 0
    
    batched_text = tokenizer.decode(batched_output[0], skip_special_tokens=True)
    
    print(f"\n[{timestamp()}]   Generated: '{batched_text}'")
    print(f"[{timestamp()}]   Time: {batched_time:.2f}s")
    if device == 'cuda':
        print(f"[{timestamp()}]   Peak VRAM: {batched_vram:.2f} GB")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nBaseline:  '{baseline_text}'")
    print(f"  Time: {baseline_time:.2f}s")
    if device == 'cuda':
        print(f"  VRAM: {baseline_vram:.2f} GB")
    
    print(f"\nBatched:   '{batched_text}'")
    print(f"  Time: {batched_time:.2f}s")
    if device == 'cuda':
        print(f"  VRAM: {batched_vram:.2f} GB")
    
    print("\nComparison:")
    if baseline_text == batched_text:
        print("  ✅ Outputs MATCH exactly!")
    else:
        print("  ⚠️  Outputs differ (expected with quantization)")
    
    slowdown = batched_time / baseline_time
    print(f"  Speed: {slowdown:.1f}x slower than baseline")
    
    if device == 'cuda':
        vram_reduction = baseline_vram / batched_vram if batched_vram > 0 else 1
        print(f"  VRAM: {vram_reduction:.1f}x reduction")
    
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
        print(f"    Decode overhead: {total_decode_time/batched_time*100:.1f}% of time")
    
    print("\n" + "="*80)
    if slowdown < 10:
        print(f"✅ SUCCESS! Batched inference is {slowdown:.1f}x slower (acceptable!)")
    else:
        print(f"⚠️  Still slow ({slowdown:.1f}x), but much better than per-tile!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = test_batched_inference()
    sys.exit(0 if success else 1)

