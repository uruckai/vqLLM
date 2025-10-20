#!/usr/bin/env python3
"""
GPU-DIRECT SOLUTION: Decompress straight to GPU, no CPU roundtrip
- Compressed weights stay in RAM (tiny footprint)
- Decode directly to GPU memory via nvCOMP
- Create PyTorch tensor from GPU pointer
- Use and free immediately
- Sequentialized MLP to prevent concurrent allocations
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import numpy as np
import torch
import torch.nn as nn
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder

print("="*80)
print("ZSTD GPU-DIRECT SOLUTION")
print("="*80)
print("Compressed RAM → GPU decode (nvCOMP) → PyTorch tensor → Use → Free")
print("NO CPU MEMORY INVOLVED IN DECODE PATH")
print()

if not ZstdGPUDecoder.is_available():
    print("❌ GPU decoder required")
    sys.exit(1)

print("[1/5] Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

print("✓ Model loaded")

# Baseline
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    t_baseline = time.time() - t0

baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
baseline_vram = torch.cuda.max_memory_allocated() / 1024**3

print(f"Baseline: {t_baseline:.2f}s, {baseline_vram:.2f} GB VRAM")
print(f"Output: '{baseline_text}'")
print()

# Compress all layers
print("[2/5] Compressing layers (kept in RAM)...")
encoder = ZstdEncoder(compression_level=9)
decoder = ZstdGPUDecoder()

linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
compressed_weights = {}

print(f"Compressing {len(linear_layers)} layers...")
t0 = time.time()
for i, (name, module) in enumerate(linear_layers):
    weight = module.weight.data.cpu().numpy()
    w_min, w_max = weight.min(), weight.max()
    scale = max(abs(w_min), abs(w_max)) / 127.0
    weight_int8 = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    compressed, _ = encoder.encode_layer(weight_int8)
    
    compressed_weights[name] = {
        'compressed': compressed,  # Stays in RAM
        'shape': weight.shape,
        'scale': scale,
        'dtype': weight.dtype
    }
    
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{len(linear_layers)}...")

compress_time = time.time() - t0
total_comp_size = sum(len(d['compressed']) for d in compressed_weights.values()) / 1024**2

print(f"✓ Compressed in {compress_time:.1f}s")
print(f"  Compressed size in RAM: {total_comp_size:.0f} MB")
print()

# GPU-direct layer
print("[3/5] Creating GPU-direct decompression model...")

class GPUDirectLinear(nn.Module):
    """Decompress directly to GPU on EVERY forward pass"""
    
    def __init__(self, compressed_data, decoder, device):
        super().__init__()
        self.compressed = compressed_data['compressed']  # Stays in RAM
        self.shape = compressed_data['shape']
        self.scale = compressed_data['scale']
        self.decoder = decoder
        self.device = device
        self.dtype = torch.float16 if compressed_data['dtype'] == np.float16 else torch.float32
        self.bias = None
    
    def set_bias(self, bias):
        if bias is not None:
            self.register_buffer('bias', bias.clone())
    
    def forward(self, x):
        # Decompress DIRECTLY TO GPU (nvCOMP does this internally)
        gpu_ptr, rows, cols, np_dtype = self.decoder.decode_layer_to_gpu(self.compressed)
        
        # Create PyTorch tensor from GPU pointer (NO COPY)
        # The data is already on GPU from nvCOMP
        import ctypes
        from torch.utils.dlpack import from_dlpack
        
        # We need to wrap the GPU pointer in a way PyTorch understands
        # Option 1: Use torch.as_tensor with a NumPy array that references GPU memory
        # Option 2: Use DLPack (more complex but cleaner)
        # Option 3: Use torch.from_numpy + .cuda() -- NO, this copies!
        
        # For now, use torch's direct GPU memory interface
        # Create a CUDA tensor from raw pointer
        numel = rows * cols
        weight_int8_gpu = torch.cuda.ByteTensor(
            torch.cuda.ByteStorage.from_buffer(
                ctypes.c_void_p(gpu_ptr),
                numel,
                dtype=torch.int8
            )
        ).view(torch.int8).view(rows, cols)
        
        # Dequantize on GPU
        weight_gpu = weight_int8_gpu.to(self.dtype) * self.scale
        
        # Compute
        output = torch.nn.functional.linear(x, weight_gpu, self.bias)
        
        # Free GPU memory via CUDA
        import cuda_helper  # We'll need a tiny C extension for this
        cuda_helper.free(gpu_ptr)
        
        del weight_gpu
        del weight_int8_gpu
        
        return output

# Wait, torch.cuda.ByteStorage.from_buffer doesn't work like that
# Let me use a simpler approach: cudaMemcpy the GPU pointer into a PyTorch tensor

class GPUDirectLinear(nn.Module):
    """Decompress directly to GPU, wrap in PyTorch tensor"""
    
    def __init__(self, compressed_data, decoder, device):
        super().__init__()
        self.compressed = compressed_data['compressed']
        self.shape = compressed_data['shape']
        self.scale = compressed_data['scale']
        self.decoder = decoder
        self.device = device
        self.dtype = torch.float16 if compressed_data['dtype'] == np.float16 else torch.float32
        self.bias = None
    
    def set_bias(self, bias):
        if bias is not None:
            self.register_buffer('bias', bias.clone())
    
    def forward(self, x):
        import ctypes
        
        # Decompress to GPU pointer
        gpu_ptr, rows, cols, np_dtype = self.decoder.decode_layer_to_gpu(self.compressed)
        
        # Create PyTorch tensor from GPU pointer
        # Allocate a PyTorch CUDA tensor
        weight_int8_torch = torch.empty((rows, cols), dtype=torch.int8, device=self.device)
        
        # Copy from our GPU buffer to PyTorch's GPU tensor
        # This is still a GPU-to-GPU copy (fast), NOT host-to-device
        torch.cuda.synchronize()
        import ctypes
        from torch.utils._pytree import tree_map
        
        # Use ctypes to call cudaMemcpy
        # Actually, simpler: use torch's copy_() which accepts device pointers
        # But we need to wrap the raw pointer...
        
        # Let's use pycuda or cupy? No, too heavy.
        # Use torch.cuda.external_stream()?
        
        # Simplest: Create a CuPy array from pointer, then convert to PyTorch
        # But that requires cupy...
        
        # OK, let's add a C helper function to copy GPU->GPU into a PyTorch tensor
        # For now, do it via ctypes + cudaMemcpy
        
        # Get PyTorch tensor's data pointer
        dst_ptr = weight_int8_torch.data_ptr()
        
        # Call cudaMemcpy (GPU to GPU)
        import ctypes
        cuda = ctypes.CDLL('libcudart.so')  # or cudart64_*.dll on Windows
        cuda.cudaMemcpy(
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(gpu_ptr),
            ctypes.c_size_t(rows * cols),
            ctypes.c_int(3)  # cudaMemcpyDeviceToDevice = 3
        )
        
        # Free the original GPU buffer
        cuda.cudaFree(ctypes.c_void_p(gpu_ptr))
        
        # Dequantize
        weight_gpu = weight_int8_torch.to(self.dtype) * self.scale
        del weight_int8_torch
        
        # Compute
        output = torch.nn.functional.linear(x, weight_gpu, self.bias)
        
        # Free
        del weight_gpu
        torch.cuda.empty_cache()
        
        return output

# Sequentialized MLP
class SequentialMLP(nn.Module):
    """MLP that calls gate/up/down sequentially"""
    
    def __init__(self, original_mlp, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = original_mlp.act_fn
    
    def forward(self, x):
        gate_output = self.gate_proj(x)
        activated = self.act_fn(gate_output)
        del gate_output
        
        up_output = self.up_proj(x)
        multiplied = activated * up_output
        del activated, up_output
        
        down_output = self.down_proj(multiplied)
        del multiplied
        
        return down_output

# Replace layers
print("Replacing layers...")
replaced_linear = 0
replaced_mlp = 0

for name, module in model.named_modules():
    if isinstance(module, LlamaMLP):
        gate_name = f"{name}.gate_proj" if name else "gate_proj"
        up_name = f"{name}.up_proj" if name else "up_proj"
        down_name = f"{name}.down_proj" if name else "down_proj"
        
        if gate_name in compressed_weights and up_name in compressed_weights and down_name in compressed_weights:
            gate_proj = GPUDirectLinear(compressed_weights[gate_name], decoder, device)
            gate_proj.set_bias(module.gate_proj.bias.data if module.gate_proj.bias is not None else None)
            
            up_proj = GPUDirectLinear(compressed_weights[up_name], decoder, device)
            up_proj.set_bias(module.up_proj.bias.data if module.up_proj.bias is not None else None)
            
            down_proj = GPUDirectLinear(compressed_weights[down_name], decoder, device)
            down_proj.set_bias(module.down_proj.bias.data if module.down_proj.bias is not None else None)
            
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            mlp_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, mlp_name, SequentialMLP(module, gate_proj, up_proj, down_proj))
            else:
                setattr(model, mlp_name, SequentialMLP(module, gate_proj, up_proj, down_proj))
            
            replaced_mlp += 1
            replaced_linear += 3
    else:
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                
                if any(full_name.endswith(proj) for proj in ['.gate_proj', '.up_proj', '.down_proj']):
                    continue
                
                if full_name in compressed_weights:
                    new_layer = GPUDirectLinear(compressed_weights[full_name], decoder, device)
                    new_layer.set_bias(child.bias.data if child.bias is not None else None)
                    setattr(module, child_name, new_layer)
                    replaced_linear += 1

print(f"✓ Replaced {replaced_linear} linear layers")
print(f"✓ Replaced {replaced_mlp} MLP modules")

torch.cuda.empty_cache()
import gc
gc.collect()

print()

# Test
print("[4/5] Running inference...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
torch.cuda.reset_peak_memory_stats()

try:
    with torch.no_grad():
        t0 = time.time()
        outputs_compressed = model.generate(
            **inputs,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        t_compressed = time.time() - t0
        
        compressed_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)
        compressed_vram = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"✓ Success!")
        print(f"  Time: {t_compressed:.2f}s")
        print(f"  Peak VRAM: {compressed_vram:.2f} GB")
        print(f"  Output: '{compressed_text}'")
        
except RuntimeError as e:
    print(f"❌ Error: {e}")
    if torch.cuda.is_available():
        print(f"GPU mem at error: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    raise

print()

# Summary
print("="*80)
print("RESULTS")
print("="*80)
print(f"Baseline:   {t_baseline:.2f}s, {baseline_vram:.2f} GB VRAM")
print(f"Compressed: {t_compressed:.2f}s, {compressed_vram:.2f} GB VRAM")
print(f"Slowdown: {t_compressed/t_baseline:.1f}x")
if compressed_vram < baseline_vram:
    print(f"✓ VRAM saved: {baseline_vram - compressed_vram:.2f} GB")
else:
    print(f"⚠️  VRAM increased: +{compressed_vram - baseline_vram:.2f} GB")
print()
print(f"Compressed size in RAM: {total_comp_size:.0f} MB")
if baseline_text == compressed_text:
    print("✓ Output matches baseline!")
else:
    print("⚠️  Output differs")
print()
print("✓ Complete - TRUE GPU-DIRECT DECODE!")

