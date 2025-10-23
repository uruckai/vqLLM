#!/usr/bin/env python3
"""
Real-world LLM inference test with Zstd compression
Uses TinyLlama and GPU-accelerated Zstd decode via nvCOMP
"""

# CRITICAL: Set environment variables BEFORE importing torch!
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch

# Verify the allocator config was applied
print(f"CUDA allocator config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print()
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
    
    # PER-CHANNEL QUANTIZATION (critical for LLMs!)
    # Each output channel (row) gets its own scale
    # This prevents garbage output from poorly-scaled channels
    scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8)  # Avoid division by zero
    # CRITICAL: Cast scales to float32 explicitly (weight might be float16, but scales must be float32 for precision)
    scales = scales.astype(np.float32)
    weight_int8 = np.clip(np.round(weight / scales), -127, 127).astype(np.int8)
    
    # Compress
    t0 = time.time()
    compressed, ratio = encoder.encode_layer(weight_int8)
    compress_time += time.time() - t0
    
    # Store (scales is now a vector, one per output channel)
    scales_to_store = scales.squeeze()  # Store as 1D array
    
    # DEBUG: Verify what we're storing (AFTER compression succeeds)
    if i == 0:
        print(f"\n[DEBUG COMPRESS] Layer '{name}':")
        print(f"  Weight shape: {weight.shape}, dtype: {weight.dtype}")
        print(f"  Scales shape BEFORE squeeze: {scales.shape}, dtype: {scales.dtype}")
        print(f"  Scales AFTER squeeze: shape={scales_to_store.shape}, dtype={scales_to_store.dtype}")
        print(f"  Scales range: [{scales_to_store.min():.6f}, {scales_to_store.max():.6f}]")
        print(f"  First 5 scale values: {scales_to_store[:5]}")
    
    compressed_weights[name] = {
        'compressed': compressed,
        'shape': weight.shape,
        'scale': scales_to_store,
        'scale_dtype': scales_to_store.dtype,  # Store scale dtype separately
        'dtype': weight.dtype,  # Original weight dtype
        'ratio': ratio
    }
    
    # DEBUG: Verify what's in the dictionary immediately after storing
    if i == 0:
        stored_scale = compressed_weights[name]['scale']
        print(f"  IMMEDIATELY after storing in dict:")
        print(f"    Type: {type(stored_scale)}")
        print(f"    Shape: {stored_scale.shape}")
        print(f"    Dtype: {stored_scale.dtype}")
        print(f"    Range: [{stored_scale.min():.6f}, {stored_scale.max():.6f}]")
        print(f"    First 5: {stored_scale[:5]}\n")
    
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

# DEBUG: Check the first layer's scales RIGHT BEFORE creating the model
first_layer_name = list(compressed_weights.keys())[0]
test_scale = compressed_weights[first_layer_name]['scale']
print(f"\n[DEBUG END OF COMPRESSION] Checking first layer '{first_layer_name}' scales:")
print(f"  Dtype: {test_scale.dtype}")
print(f"  Range: [{test_scale.min():.6f}, {test_scale.max():.6f}]")
print(f"  First 5: {test_scale[:5]}")

print()

# Replace layers with compressed versions
print("[5/6] Creating compressed model...")

class CompressedLinear(torch.nn.Module):
    """Linear layer that decompresses weights on-the-fly (no caching!)"""
    
    def __init__(self, original_module, compressed_data, decoder_handle, target_device='cuda'):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
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
        
        # Convert scale to torch tensor on GPU (per-channel or per-tensor)
        scale_np = compressed_data['scale']
        if isinstance(scale_np, np.ndarray):
            # DEBUG: Check what we received
            print(f"[DEBUG LOAD] Received scale_np:")
            print(f"  Type: {type(scale_np)}")
            print(f"  Shape: {scale_np.shape}")
            print(f"  Dtype: {scale_np.dtype}")
            print(f"  Range: [{scale_np.min():.6f}, {scale_np.max():.6f}]")
            print(f"  First 5 values: {scale_np[:5]}")
            
            # Per-channel quantization (1D array)
            scale_tensor = torch.from_numpy(scale_np).to(torch_dtype).to(target_device)
            # DEBUG: Verify scales are reasonable
            if scale_tensor.numel() < 10:
                print(f"[DEBUG] After torch conversion: {scale_tensor.cpu().numpy()}")
            else:
                print(f"[DEBUG] After torch conversion: shape={scale_tensor.shape}, range=[{scale_tensor.min():.6f}, {scale_tensor.max():.6f}]")
            self.register_buffer('scale', scale_tensor)
        else:
            # Per-tensor quantization (scalar) - legacy support
            print(f"[DEBUG] Using per-tensor scale: {scale_np}")
            self.register_buffer('scale', torch.tensor(scale_np, dtype=torch_dtype, device=target_device))
        
        # Keep bias if present
        if original_module.bias is not None:
            self.register_buffer('bias', original_module.bias.data.clone())
        else:
            self.bias = None
        
        # NO CACHING - decompress fresh every time!
    
    def forward(self, x):
        """Decompress on-the-fly with GPU-direct decode (no CPU roundtrip!)"""
        # GPU-DIRECT DECODE: decompress directly to GPU memory
        gpu_ptr, rows, cols, dtype = self.decoder.decode_layer_to_gpu(self.compressed)
        
        # Copy GPU memory to PyTorch tensor
        # Create empty tensor on GPU
        weight_int8_gpu = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
        
        # Copy from nvCOMP's GPU buffer to PyTorch's GPU buffer
        import ctypes
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaMemcpy(
            ctypes.c_void_p(weight_int8_gpu.data_ptr()),
            ctypes.c_void_p(gpu_ptr),
            ctypes.c_size_t(rows * cols),
            ctypes.c_int(1)  # cudaMemcpyDeviceToDevice
        )
        
        # Free nvCOMP's GPU buffer
        cudart.cudaFree(ctypes.c_void_p(gpu_ptr))
        
        # Dequantize on GPU (no CPU involved!)
        # Per-channel: scale is a vector (one per output channel/row)
        # Convert to float first, THEN multiply by scale
        weight_fp_unscaled = weight_int8_gpu.to(self.dtype)
        
        # Broadcast scale correctly: (rows,) -> (rows, 1) for broadcasting with (rows, cols)
        if self.scale.dim() == 1:
            # Per-channel: expand scale to (rows, 1)
            scale_expanded = self.scale.view(-1, 1)
        else:
            # Per-tensor: scalar
            scale_expanded = self.scale
        
        weight_fp = weight_fp_unscaled * scale_expanded
        
        # Reshape and use
        weight_fp = weight_fp.reshape(self.shape)
        output = torch.nn.functional.linear(x, weight_fp, self.bias)
        
        # FREE GPU memory
        del weight_fp
        del weight_fp_unscaled
        del weight_int8_gpu
        
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
                    # DEBUG: Check what we're passing to CompressedLinear
                    if 'q_proj' in compressed_name and 'layers.0' in compressed_name:
                        print(f"\n[DEBUG BEFORE PASSING] Passing to CompressedLinear for '{compressed_name}':")
                        scale_data = compressed_data['scale']
                        print(f"  Type: {type(scale_data)}")
                        print(f"  Shape: {scale_data.shape}")
                        print(f"  Dtype: {scale_data.dtype}")
                        print(f"  Range: [{scale_data.min():.6f}, {scale_data.max():.6f}]")
                        print(f"  First 5: {scale_data[:5]}\n")
                    
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

# CRITICAL: Delete original uncompressed weights to free VRAM!
print("  Freeing uncompressed weights...")
deleted_count = 0
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and not isinstance(module, CompressedLinear):
        # Check if this layer has a weight parameter (not compressed)
        if hasattr(module, 'weight') and module.weight is not None:
            # This is an uncompressed layer we can't compress - leave it
            pass
    elif isinstance(module, CompressedLinear):
        # For compressed layers, the original weight in the parent module is gone already
        deleted_count += 1

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    freed_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory after freeing: {freed_mem:.2f} GB")

print(f"✓ Model ready with {deleted_count} compressed layers")
print()

# Pre-warm the cache: decompress all layers to GPU
# Run compressed inference (on-the-fly decompression happens in forward()!)
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

