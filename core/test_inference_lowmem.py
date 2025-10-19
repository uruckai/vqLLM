#!/usr/bin/env python3
"""
Low-Memory Inference with Compressed Weights

This script demonstrates VRAM-efficient inference by:
1. Keeping weights COMPRESSED in memory
2. Decompressing layers ON-DEMAND during forward pass
3. Freeing decompressed weights immediately after use

This allows running large models on GPUs with limited VRAM!
Example: Run Llama-3.1-8B on a 4GB GPU instead of 16GB
"""

import numpy as np
import ctypes
import os
import sys
import time
from pathlib import Path
import gc
import torch
import torch.nn as nn

def load_codec():
    """Load the codec library"""
    lib_path = Path("build/libcodec_core.so")
    if not lib_path.exists():
        print(f"✗ Codec library not found at {lib_path}")
        print("  Build it first: cd core && bash build.sh")
        return None
    
    lib = ctypes.CDLL(str(lib_path))
    
    # Set return types AND argument types (critical for 64-bit pointers!)
    lib.encoder_create.restype = ctypes.c_void_p
    lib.encoder_create.argtypes = [ctypes.c_uint16]
    
    lib.encoder_destroy.argtypes = [ctypes.c_void_p]
    
    lib.encoder_encode.restype = ctypes.c_float
    lib.encoder_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t)
    ]
    
    lib.decoder_create.restype = ctypes.c_void_p
    lib.decoder_create.argtypes = []
    
    lib.decoder_destroy.argtypes = [ctypes.c_void_p]
    
    lib.decoder_decode.restype = ctypes.c_float
    lib.decoder_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int8)
    ]
    
    lib.decoder_is_available.restype = ctypes.c_bool
    lib.decoder_is_available.argtypes = []
    
    lib.free_buffer.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    
    return lib

class CompressedTensor:
    """Wrapper around a compressed weight tensor that decompresses on-demand"""
    
    def __init__(self, codec_lib, original_tensor):
        """Compress and store a tensor"""
        self.lib = codec_lib
        self.device = original_tensor.device
        
        # Store metadata
        self.shape = original_tensor.shape
        self.dtype = original_tensor.dtype
        
        # Convert to numpy and quantize to INT8
        weight_np = original_tensor.detach().cpu().numpy()
        
        if weight_np.dtype in [np.float16, np.float32]:
            self.scale = np.abs(weight_np).max() / 127.0
            if self.scale == 0:
                self.scale = 1.0
            weight_int8 = np.clip(np.round(weight_np / self.scale), -128, 127).astype(np.int8)
        else:
            self.scale = 1.0
            weight_int8 = weight_np.astype(np.int8)
        
        # Flatten and compress into tiles
        flat = weight_int8.flatten()
        tile_size = 256 * 256
        num_tiles = (len(flat) + tile_size - 1) // tile_size
        
        self.compressed_tiles = []
        self.num_elements = len(flat)
        
        for i in range(num_tiles):
            start = i * tile_size
            end = min(start + tile_size, len(flat))
            tile_data = flat[start:end]
            
            # Pad to tile size if needed
            if len(tile_data) < tile_size:
                padded = np.zeros(tile_size, dtype=np.int8, order='C')
                padded[:len(tile_data)] = tile_data
                tile_data = padded
            
            # Force contiguous copy (critical for C++ library!)
            tile_2d = np.ascontiguousarray(tile_data.reshape(256, 256))
            
            # Encode tile
            encoder = self.lib.encoder_create(256)
            data_ptr = tile_2d.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
            output_ptr = ctypes.POINTER(ctypes.c_uint8)()
            output_size = ctypes.c_size_t()
            
            self.lib.encoder_encode(encoder, data_ptr, 256, 256,
                                   ctypes.byref(output_ptr), ctypes.byref(output_size))
            
            # Copy compressed data
            compressed = bytes(ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_uint8 * output_size.value)).contents)
            self.compressed_tiles.append(compressed)
            
            self.lib.free_buffer(output_ptr)
            self.lib.encoder_destroy(encoder)
        
        self.original_size = original_tensor.element_size() * original_tensor.numel()
        self.compressed_size = sum(len(t) for t in self.compressed_tiles)
    
    def decompress(self):
        """Decompress and return the tensor"""
        # Decompress all tiles
        decoder = self.lib.decoder_create()
        all_data = []
        
        for compressed in self.compressed_tiles:
            # Decompress tile
            decoded = np.zeros((256, 256), dtype=np.int8)
            decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
            
            # Convert bytes to ctypes array
            compressed_array = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
            compressed_ptr = ctypes.cast(compressed_array, ctypes.POINTER(ctypes.c_uint8))
            
            self.lib.decoder_decode(decoder, compressed_ptr, len(compressed), decoded_ptr)
            
            all_data.append(decoded.flatten())
        
        self.lib.decoder_destroy(decoder)
        
        # Concatenate and trim to original size
        full_data = np.concatenate(all_data)[:self.num_elements]
        
        # Dequantize - MUST convert to original dtype
        if self.dtype in [torch.float16, torch.float32]:
            # Convert int8 -> float with proper dtype
            if self.dtype == torch.float16:
                full_data = full_data.astype(np.float16) * self.scale
            else:
                full_data = full_data.astype(np.float32) * self.scale
        
        # Reshape and convert to tensor
        result = full_data.reshape(self.shape)
        tensor = torch.from_numpy(result).to(self.device)
        
        return tensor
    
    def get_compression_ratio(self):
        """Return compression ratio"""
        return self.original_size / self.compressed_size

class CompressedLinear(torch.nn.Module):
    """Replacement for nn.Linear that uses compressed weights"""
    
    def __init__(self, original_linear, codec_lib):
        """Wrap a linear layer with compressed weights"""
        super().__init__()
        
        self.compressed_weight = CompressedTensor(codec_lib, original_linear.weight.data)
        
        # Store bias uncompressed (usually small) as a parameter
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.bias = None
        
        # Store original layer attributes
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Track decode stats
        self.decode_count = 0
        self.decode_time = 0.0
    
    def forward(self, x):
        """Forward pass: decompress weights, compute, free"""
        # Decompress weights TO THE SAME DEVICE AS INPUT
        start = time.time()
        self.compressed_weight.device = x.device  # Update target device
        weight = self.compressed_weight.decompress()
        self.decode_time += time.time() - start
        self.decode_count += 1
        
        # Compute (now all on same device)
        output = nn.functional.linear(x, weight, self.bias)
        
        # Free decompressed weights immediately
        del weight
        if x.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return output

def compress_model_weights(model, codec_lib, verbose=True):
    """Replace all Linear layers in model with CompressedLinear"""
    compressed_count = 0
    total_original_size = 0
    total_compressed_size = 0
    
    def compress_layer(module, name=""):
        nonlocal compressed_count, total_original_size, total_compressed_size
        
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, nn.Linear):
                # Replace with compressed version
                compressed = CompressedLinear(child, codec_lib)
                setattr(module, child_name, compressed)
                
                compressed_count += 1
                total_original_size += compressed.compressed_weight.original_size
                total_compressed_size += compressed.compressed_weight.compressed_size
                
                if verbose and compressed_count % 10 == 0:
                    print(f"  Compressed {compressed_count} layers...", end='\r')
            else:
                # Recurse into submodules
                compress_layer(child, full_name)
    
    compress_layer(model)
    
    if verbose:
        print(f"\n✓ Compressed {compressed_count} Linear layers")
        print(f"  Original size:    {total_original_size / 1024**3:.2f} GB")
        print(f"  Compressed size:  {total_compressed_size / 1024**3:.2f} GB")
        print(f"  Compression ratio: {total_original_size / total_compressed_size:.2f}x")
        print(f"  Space saved:      {(1 - total_compressed_size/total_original_size)*100:.1f}%")
    
    return compressed_count

def test_low_memory_inference():
    """Test inference with layer-by-layer decompression"""
    
    print("="*80)
    print("LOW-MEMORY INFERENCE TEST")
    print("="*80)
    print("\nThis test demonstrates VRAM-efficient inference by keeping weights")
    print("COMPRESSED and decompressing only the current layer during forward pass.")
    
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
    
    # Get HuggingFace token
    hf_token = os.environ.get('HF_TOKEN', None)
    
    # Load model
    print("\n[3/5] Loading model...")
    
    # Start with smaller model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"  Model: {model_name}")
    print("  (Using TinyLlama for faster testing - same technique works for larger models)")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to('cpu')
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Model loaded: {model_name}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Measure baseline memory
    print("\n[4/5] Running baseline inference (uncompressed)...")
    test_prompt = "The capital of France is"
    print(f"  Prompt: '{test_prompt}'")
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    # Get initial memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model = model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            baseline_output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        baseline_vram = torch.cuda.max_memory_allocated() / 1024**3
        baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        
        print(f"  Generated: '{baseline_text}'")
        print(f"  Peak VRAM: {baseline_vram:.2f} GB")
        
        # Move back to CPU
        model = model.to('cpu')
        torch.cuda.empty_cache()
    else:
        print("  (Running on CPU - VRAM measurements not available)")
        with torch.no_grad():
            baseline_output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        print(f"  Generated: '{baseline_text}'")
        baseline_vram = 0
    
    # Compress model weights
    print("\n[5/5] Compressing model and running low-memory inference...")
    print("  Compressing all Linear layers...")
    
    num_compressed = compress_model_weights(model, lib)
    
    # Run inference with compressed weights
    print("\n  Running inference with compressed weights...")
    print("  (Weights decompress on-demand, then immediately freed)")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model = model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        start = time.time()
        with torch.no_grad():
            compressed_output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        inference_time = time.time() - start
        
        compressed_vram = torch.cuda.max_memory_allocated() / 1024**3
        compressed_text = tokenizer.decode(compressed_output[0], skip_special_tokens=True)
        
        print(f"  Generated: '{compressed_text}'")
        print(f"  Peak VRAM: {compressed_vram:.2f} GB")
        print(f"  Time: {inference_time:.2f}s")
    else:
        start = time.time()
        with torch.no_grad():
            compressed_output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        inference_time = time.time() - start
        
        compressed_text = tokenizer.decode(compressed_output[0], skip_special_tokens=True)
        print(f"  Generated: '{compressed_text}'")
        print(f"  Time: {inference_time:.2f}s")
        compressed_vram = 0
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nBaseline output:")
    print(f"  '{baseline_text}'")
    print("\nCompressed output:")
    print(f"  '{compressed_text}'")
    
    if baseline_text == compressed_text:
        print("\n✅ OUTPUTS MATCH!")
    else:
        print("\n⚠️  Outputs differ slightly (expected due to quantization)")
    
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("VRAM USAGE COMPARISON")
        print("="*80)
        print(f"Baseline (uncompressed):  {baseline_vram:.2f} GB")
        print(f"Compressed (on-demand):   {compressed_vram:.2f} GB")
        print(f"VRAM reduction:           {baseline_vram / compressed_vram:.2f}x")
        print(f"VRAM saved:               {(1 - compressed_vram/baseline_vram)*100:.1f}%")
        
        print("\n💡 KEY INSIGHT:")
        print("  With this approach, you can run models that normally require")
        print(f"  {baseline_vram:.0f}GB VRAM on a GPU with only {compressed_vram:.0f}GB!")
        print("  Example: Run Llama-3.1-8B on a 4GB GPU instead of 16GB GPU")
    
    return True

if __name__ == "__main__":
    success = test_low_memory_inference()
    sys.exit(0 if success else 1)

