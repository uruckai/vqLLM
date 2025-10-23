#!/usr/bin/env python3
"""
Debug quantization to see what's going wrong
"""

import numpy as np
import torch

print("Testing quantization schemes...")
print("=" * 80)

# Simulate a typical LLM weight matrix
np.random.seed(42)
rows, cols = 2048, 2048

# Create weight with varying channel magnitudes (like real LLM)
weight = np.random.randn(rows, cols).astype(np.float32)
# Make some channels have much larger values
weight[:100, :] *= 0.01  # Small values
weight[100:200, :] *= 5.0  # Large values

print(f"Weight shape: {weight.shape}")
print(f"Channel 0 range: [{weight[0].min():.6f}, {weight[0].max():.6f}]")
print(f"Channel 100 range: [{weight[100].min():.6f}, {weight[100].max():.6f}]")
print(f"Overall range: [{weight.min():.6f}, {weight.max():.6f}]")
print()

# Per-tensor quantization
print("-" * 80)
print("PER-TENSOR QUANTIZATION:")
scale_tensor = max(abs(weight.min()), abs(weight.max())) / 127.0
weight_int8_tensor = np.clip(np.round(weight / scale_tensor), -127, 127).astype(np.int8)
weight_dequant_tensor = weight_int8_tensor.astype(np.float32) * scale_tensor

error_tensor = np.abs(weight - weight_dequant_tensor)
print(f"  Scale: {scale_tensor:.6f}")
print(f"  Channel 0 error: {error_tensor[0].mean():.6f} (avg)")
print(f"  Channel 100 error: {error_tensor[100].mean():.6f} (avg)")
print(f"  Overall error: {error_tensor.mean():.6f} (avg)")
print(f"  Max error: {error_tensor.max():.6f}")
print()

# Per-channel quantization
print("-" * 80)
print("PER-CHANNEL QUANTIZATION:")
scales_channel = np.abs(weight).max(axis=1, keepdims=True) / 127.0
scales_channel = np.maximum(scales_channel, 1e-8)
weight_int8_channel = np.clip(np.round(weight / scales_channel), -127, 127).astype(np.int8)
weight_dequant_channel = weight_int8_channel.astype(np.float32) * scales_channel

error_channel = np.abs(weight - weight_dequant_channel)
print(f"  Scales shape: {scales_channel.shape}")
print(f"  Scale range: [{scales_channel.min():.6f}, {scales_channel.max():.6f}]")
print(f"  Channel 0 scale: {scales_channel[0, 0]:.6f}")
print(f"  Channel 100 scale: {scales_channel[100, 0]:.6f}")
print(f"  Channel 0 error: {error_channel[0].mean():.6f} (avg)")
print(f"  Channel 100 error: {error_channel[100].mean():.6f} (avg)")
print(f"  Overall error: {error_channel.mean():.6f} (avg)")
print(f"  Max error: {error_channel.max():.6f}")
print()

# Test PyTorch broadcasting
print("-" * 80)
print("PYTORCH DEQUANTIZATION TEST:")
print()

# Simulate what happens in our code
weight_int8_torch = torch.from_numpy(weight_int8_channel).to(torch.float16)
scales_torch = torch.from_numpy(scales_channel.squeeze()).to(torch.float16)  # 1D vector

print(f"weight_int8_torch.shape: {weight_int8_torch.shape}")
print(f"scales_torch.shape: {scales_torch.shape}")
print()

# Method 1: unsqueeze(1) - this is what we have
print("Method 1: scales.unsqueeze(1)")
scales_method1 = scales_torch.unsqueeze(1)
print(f"  scales_method1.shape: {scales_method1.shape}")
weight_dequant_method1 = weight_int8_torch * scales_method1
print(f"  Result shape: {weight_dequant_method1.shape}")
error_method1 = torch.abs(torch.from_numpy(weight).to(torch.float16) - weight_dequant_method1.to(torch.float32))
print(f"  Error: {error_method1.mean():.6f} (avg)")
print()

# Method 2: reshape to (rows, 1)
print("Method 2: scales.view(-1, 1)")
scales_method2 = scales_torch.view(-1, 1)
print(f"  scales_method2.shape: {scales_method2.shape}")
weight_dequant_method2 = weight_int8_torch * scales_method2
print(f"  Result shape: {weight_dequant_method2.shape}")
error_method2 = torch.abs(torch.from_numpy(weight).to(torch.float16) - weight_dequant_method2.to(torch.float32))
print(f"  Error: {error_method2.mean():.6f} (avg)")
print()

# Check if methods are identical
print(f"Methods identical: {torch.allclose(weight_dequant_method1, weight_dequant_method2)}")
print()

# Check compression ratios
print("-" * 80)
print("COMPRESSION RATIOS:")
unique_tensor = len(np.unique(weight_int8_tensor))
unique_channel = len(np.unique(weight_int8_channel))
print(f"  Per-tensor unique values: {unique_tensor} / 255 = {unique_tensor/255*100:.1f}%")
print(f"  Per-channel unique values: {unique_channel} / 255 = {unique_channel/255*100:.1f}%")
print()

print("=" * 80)
print("DIAGNOSIS:")
print()
if error_tensor.mean() > error_channel.mean() * 10:
    print("✓ Per-channel is MUCH better than per-tensor (expected)")
else:
    print("⚠️  Per-channel is not significantly better - something wrong?")

if unique_channel > unique_tensor:
    print("✓ Per-channel uses more unique values (better utilization)")
else:
    print("⚠️  Per-channel doesn't use more values - compression issue?")

print()
print("Expected: Per-channel should have ~10x lower error and better compression")

