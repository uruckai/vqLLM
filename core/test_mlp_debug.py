#!/usr/bin/env python3
"""
Debug: Test MLP compression step by step to find where errors start
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from bindings_zstd import ZstdEncoder, ZstdGPUDecoder
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("DEBUG: MLP COMPRESSION STEP BY STEP")
print("="*80)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Test 1: Single token baseline
print("=== TEST 1: Single token generation ===")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    # Generate just 1 more token
    outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
baseline_1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Baseline (1 token): '{baseline_1}'")

# Generate 5 more tokens
outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
baseline_5 = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Baseline (5 tokens): '{baseline_5}'")

# Generate 10 more tokens
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
baseline_10 = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Baseline (10 tokens): '{baseline_10}'")
print()

# Test 2: Check MLP layer outputs directly
print("=== TEST 2: MLP layer output comparison ===")
del model
torch.cuda.empty_cache()

# Get one MLP layer to test
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

# Find first MLP layer (gate_proj)
mlp_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and 'gate_proj' in name:
        mlp_layers.append((name, module))
        if len(mlp_layers) >= 1:
            break

if mlp_layers:
    layer_name, layer_module = mlp_layers[0]
    print(f"Testing MLP layer: {layer_name}")

    # Get some input to this layer (from first decoder layer)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # Run through embeddings and first few layers to get realistic input
        hidden_states = model.model.embed_tokens(inputs['input_ids'])
        position_ids = torch.arange(0, inputs['input_ids'].size(1), dtype=torch.long, device=device).unsqueeze(0)
        hidden_states = model.model.layers[0].input_layernorm(hidden_states)

        # Get the original output
        original_output = layer_module(hidden_states)

        # Now compress this layer and compare
        weight = layer_module.weight.data.cpu().numpy()

        encoder = ZstdEncoder(compression_level=9)
        scales = np.abs(weight).max(axis=1, keepdims=True) / 127.0
        scales = np.maximum(scales, 1e-8).astype(np.float32)
        weight_int8 = np.clip(np.round(weight / scales), -127, 127).astype(np.int8)
        compressed, _ = encoder.encode_layer(weight_int8)

        class CompressedMLP(torch.nn.Module):
            def __init__(self, original_module, compressed_data, decoder_handle):
                super().__init__()
                self.compressed = compressed_data['compressed']
                self.shape = compressed_data['shape']
                self.decoder = decoder_handle

                scale_tensor = torch.from_numpy(compressed_data['scale']).to(torch.float16).to(device)
                self.register_buffer('scale', scale_tensor)

                if original_module.bias is not None:
                    self.register_buffer('bias', original_module.bias.data.clone())
                else:
                    self.bias = None

            def forward(self, x):
                import ctypes

                gpu_ptr, rows, cols, _ = self.decoder.decode_layer_to_gpu(self.compressed)
                weight_int8 = torch.empty((rows, cols), dtype=torch.int8, device=x.device)

                cudart = ctypes.CDLL('libcudart.so')
                cudart.cudaMemcpy(
                    ctypes.c_void_p(weight_int8.data_ptr()),
                    ctypes.c_void_p(gpu_ptr),
                    ctypes.c_size_t(rows * cols),
                    ctypes.c_int(1)
                )
                cudart.cudaFree(ctypes.c_void_p(gpu_ptr))

                weight_fp = weight_int8.to(torch.float16) * self.scale.view(-1, 1)
                weight_fp = weight_fp.reshape(self.shape)

                output = torch.nn.functional.linear(x, weight_fp, self.bias)

                del weight_fp, weight_int8
                return output

        decoder = ZstdGPUDecoder()
        compressed_layer = CompressedMLP(layer_module, {
            'compressed': compressed,
            'shape': weight.shape,
            'scale': scales.squeeze().copy(),
        }, decoder)

        # Compare outputs
        with torch.no_grad():
            compressed_output = compressed_layer(hidden_states)

        print(f"Original output shape: {original_output.shape}")
        print(f"Compressed output shape: {compressed_output.shape}")

        diff = torch.abs(original_output - compressed_output).max().item()
        rel_diff = (torch.abs(original_output - compressed_output) / (torch.abs(original_output) + 1e-8)).max().item()

        print(f"Max absolute difference: {diff}")
        print(f"Max relative difference: {rel_diff}")

        if diff < 1e-3:
            print("✓ MLP layer compression is numerically accurate")
        else:
            print("✗ MLP layer compression has significant errors")

        # Check if outputs are close
        close = torch.allclose(original_output, compressed_output, rtol=1e-3, atol=1e-3)
        print(f"torch.allclose(): {close}")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()

if 'close' in locals() and close:
    print("MLP compression itself is working correctly.")
    print("The issue must be in the layer replacement or integration.")
    print()
    print("Possible issues:")
    print("  1. Layer replacement affecting gradient flow or caching")
    print("  2. Dtype mismatches in the forward pass")
    print("  3. Bias terms not handled correctly")
    print("  4. Module hooks or other PyTorch internals affected")
else:
    print("MLP compression has numerical errors.")
    print("Need to fix the compression/decompression pipeline.")

