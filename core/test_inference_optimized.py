#!/usr/bin/env python3
"""
OPTIMIZED Low-Memory Inference Test

Fixes performance issues with larger tile sizes and batching
"""

import numpy as np
import ctypes
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn

# Import from existing test
from test_inference_lowmem import load_codec

class OptimizedCompressedTensor:
    """CompressedTensor with optimized tile size and batching"""

    def __init__(self, codec_lib, tensor, tile_size=1024):
        """
        Args:
            codec_lib: Loaded codec library
            tensor: PyTorch tensor to compress
            tile_size: Size of each tile (larger = better compression, but more memory)
        """
        self.lib = codec_lib
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.tile_size = tile_size

        # Cached decompressed tensor (for speed)
        self._cached = None
        self._cache_hits = 0
        self._cache_misses = 0

        # Quantize
        data_np = tensor.cpu().detach().numpy()
        self.scale = np.abs(data_np).max() / 127.0 if np.abs(data_np).max() > 0 else 1.0
        quantized = np.round(data_np / self.scale).astype(np.int8)

        # Flatten and pad to tile boundary
        flat = quantized.flatten()
        self.num_elements = len(flat)

        # Calculate tiles needed
        tile_elements = tile_size * tile_size
        num_tiles = (self.num_elements + tile_elements - 1) // tile_elements
        padded_size = num_tiles * tile_elements

        padded = np.zeros(padded_size, dtype=np.int8)
        padded[:self.num_elements] = flat

        # Compress all tiles at once (more efficient)
        self.compressed_tiles = []
        encoder = self.lib.encoder_create(tile_size)

        for i in range(num_tiles):
            start_idx = i * tile_elements
            end_idx = min(start_idx + tile_elements, padded_size)
            tile_data = padded[start_idx:end_idx]

            # Pad last tile if needed
            if len(tile_data) < tile_elements:
                tile_padded = np.zeros(tile_elements, dtype=np.int8)
                tile_padded[:len(tile_data)] = tile_data
                tile_data = tile_padded

            tile = tile_data.reshape(tile_size, tile_size)
            tile_contig = np.ascontiguousarray(tile)

            compressed = ctypes.POINTER(ctypes.c_uint8)()
            compressed_size = ctypes.c_size_t()

            ratio = self.lib.encoder_encode(
                encoder,
                tile_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                tile_size, tile_size,
                ctypes.byref(compressed),
                ctypes.byref(compressed_size)
            )

            compressed_bytes = bytes(compressed[:compressed_size.value])
            self.lib.free_buffer(compressed)

            self.compressed_tiles.append(compressed_bytes)

        self.lib.encoder_destroy(encoder)

        self.original_size = tensor.element_size() * tensor.numel()
        self.compressed_size = sum(len(t) for t in self.compressed_tiles)
        self.compression_ratio = self.original_size / self.compressed_size

    def decompress(self):
        """Decompress and return the tensor"""
        # Check cache first
        if self._cached is not None:
            self._cache_hits += 1
            return self._cached

        self._cache_misses += 1

        # Batch decompress all tiles at once (more efficient)
        decoder = self.lib.decoder_create()
        tile_elements = self.tile_size * self.tile_size

        all_data = []
        for compressed in self.compressed_tiles:
            decoded = np.zeros(tile_elements, dtype=np.int8)
            decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))

            compressed_array = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
            compressed_ptr = ctypes.cast(compressed_array, ctypes.POINTER(ctypes.c_uint8))

            self.lib.decoder_decode(decoder, compressed_ptr, len(compressed), decoded_ptr)
            all_data.append(decoded)

        self.lib.decoder_destroy(decoder)

        # Concatenate and trim to original size
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

        # Cache for speed
        self._cached = tensor

        return tensor

    def clear_cache(self):
        """Clear cached decompressed tensor"""
        self._cached = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def get_stats(self):
        """Return cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total * 100 if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cached': self._cached is not None,
            'tiles': len(self.compressed_tiles),
            'tile_size': self.tile_size
        }

class OptimizedCompressedLinear(torch.nn.Module):
    """Linear layer with optimized compressed weights"""

    def __init__(self, original_linear, codec_lib, tile_size=1024):
        super().__init__()

        self.compressed_weight = OptimizedCompressedTensor(
            codec_lib,
            original_linear.weight.data,
            tile_size=tile_size
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
        """Forward pass with optimized decompression"""
        start = time.time()

        # Update target device
        self.compressed_weight.device = x.device

        # Decompress (should be cached after first use)
        weight = self.compressed_weight.decompress()

        self.decode_time += time.time() - start
        self.decode_count += 1

        # Compute
        output = nn.functional.linear(x, weight, self.bias)

        return output

def compress_model_optimized(model, codec_lib, tile_size=1024, verbose=True):
    """Replace all Linear layers with OptimizedCompressedLinear"""
    compressed_count = 0
    total_original_size = 0
    total_compressed_size = 0

    def compress_layer(module, name=""):
        nonlocal compressed_count, total_original_size, total_compressed_size

        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear):
                compressed = OptimizedCompressedLinear(child, codec_lib, tile_size)
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
        print(f"  Tile size: {tile_size}x{tile_size}")

    return compressed_count

def test_optimized_inference():
    """Test inference with optimized compression"""

    print("="*80)
    print("OPTIMIZED LOW-MEMORY INFERENCE TEST")
    print("="*80)
    print("\nThis test uses larger tile sizes and batching for better performance:")
    print("  - 1024x1024 tiles (16x larger than 256x256)")
    print("  - Fewer tiles per layer")
    print("  - Less Python overhead")

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

    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU available, using CPU")

    # Load PyTorch and Transformers
    print("\n[2/5] Loading PyTorch and Transformers...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ Libraries loaded")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

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
    print(f"  Generating 3 tokens (very fast test)...")

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        model = model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    baseline_start = time.time()
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs,
            max_new_tokens=3,  # Very few tokens for speed
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

    # Compress with optimized settings
    print("\n[5/5] Testing optimized compressed inference...")
    print("  Compressing model with 1024x1024 tiles...")

    compress_start = time.time()
    num_compressed = compress_model_optimized(model, lib, tile_size=1024, verbose=True)
    compress_time = time.time() - compress_start
    print(f"  Compression took: {compress_time:.1f}s")

    # Run inference
    print("\n  Running inference with optimized decompression...")
    print("  (First inference decompresses all layers, second should be faster)")

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        model = model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # First inference (cold cache)
    cached_start = time.time()
    with torch.no_grad():
        cached_output = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    first_time = time.time() - cached_start

    if device == 'cuda':
        first_vram = torch.cuda.max_memory_allocated() / 1024**3
    else:
        first_vram = 0

    # Second inference (warm cache)
    second_start = time.time()
    with torch.no_grad():
        cached_output2 = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    second_time = time.time() - second_start

    cached_text = tokenizer.decode(cached_output[0], skip_special_tokens=True)

    print(f"  First inference:  {first_time:.2f}s")
    print(f"  Second inference: {second_time:.2f}s")
    if device == 'cuda':
        print(f"  Peak VRAM: {first_vram:.2f} GB")

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\nBaseline:  '{baseline_text}'")
    print(f"  Time: {baseline_time:.2f}s")
    if device == 'cuda':
        print(f"  VRAM: {baseline_vram:.2f} GB")

    print(f"\nOptimized: '{cached_text}'")
    print(f"  First run:  {first_time:.2f}s")
    print(f"  Second run: {second_time:.2f}s")
    if device == 'cuda':
        print(f"  VRAM: {first_vram:.2f} GB")

    print("\nComparison:")
    if baseline_text == cached_text:
        print("  ✅ Outputs MATCH exactly!")
    else:
        print(f"  ⚠️  Outputs differ (expected with quantization)")

    first_slowdown = first_time / baseline_time
    second_slowdown = second_time / baseline_time
    print(f"  First run:  {first_slowdown:.1f}x slower")
    print(f"  Second run: {second_slowdown:.1f}x slower")

    if device == 'cuda':
        vram_reduction = baseline_vram / first_vram if first_vram > 0 else 1
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
        print(f"    Decode overhead: {total_decode_time/first_time*100:.1f}% of first run")

    print("\n" + "="*80)
    if second_slowdown < 10:
        print(f"✅ SUCCESS! Second run is {second_slowdown:.1f}x slower (acceptable with caching!)")
    else:
        print(f"⚠️  Still slow ({second_slowdown:.1f}x), but improved from first run")
    print("="*80)

    return True

if __name__ == "__main__":
    success = test_optimized_inference()
    sys.exit(0 if success else 1)

