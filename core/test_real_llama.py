#!/usr/bin/env python3
"""
Test codec on REAL Llama model weights
Downloads Llama-3.2-1B and compresses its weight matrices
"""

import numpy as np
import ctypes
import os
import sys

def download_and_test_llama():
    """Download real Llama model and test compression on its weights"""
    
    print("="*70)
    print("Real Llama Weight Compression Test")
    print("="*70)
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("✗ PyTorch not available. Install with: pip install torch")
        return False
    
    try:
        from transformers import AutoModelForCausalLM
        print("✓ Transformers available")
    except ImportError:
        print("✗ Transformers not available. Install with: pip install transformers")
        return False
    
    # HuggingFace token for gated models (from environment or default)
    hf_token = os.environ.get('HF_TOKEN', None)
    if not hf_token:
        print("Warning: No HF_TOKEN environment variable set.")
        print("To download Llama models, set your token:")
        print("  export HF_TOKEN=your_token_here")
        print("Attempting download anyway...")
    
    # Download Llama model (using 3.1-8B which user has access to)
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    print(f"\nDownloading {model_name}...")
    print("(This may take several minutes on first run, ~16GB download)")
    print("Note: We only load weights, not running inference, so memory usage is low")
    
    try:
        # Try to load model with HF token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="cpu",  # Keep on CPU to avoid GPU memory issues
            low_cpu_mem_usage=True
        )
        print(f"✓ Model loaded: {model_name}")
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        print("\nNote: Llama models require HuggingFace token.")
        print("Trying alternative open model: TinyLlama-1.1B...")
        
        try:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            print(f"✓ Model loaded: {model_name}")
        except Exception as e2:
            print(f"✗ Failed to download alternative: {e2}")
            return False
    
    # Extract weight matrices for testing
    print("\nExtracting weight matrices...")
    weight_tensors = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # Focus on 2D weight matrices (not embeddings or 1D biases)
            weight_tensors.append(param)
            layer_names.append(name)
    
    print(f"Found {len(weight_tensors)} weight matrices")
    
    # Test on several different layer types
    test_layers = [
        ("Embedding", 0),      # First layer (often embedding)
        ("Attention Q", len(weight_tensors)//4),  # Early attention
        ("Attention K", len(weight_tensors)//3),  # Mid attention
        ("MLP", len(weight_tensors)//2),          # MLP layer
        ("Output", -1),        # Last layer
    ]
    
    results = []
    
    for layer_desc, idx in test_layers:
        if idx >= len(weight_tensors):
            continue
            
        tensor = weight_tensors[idx]
        name = layer_names[idx]
        
        print(f"\n{'='*70}")
        print(f"Testing: {layer_desc}")
        print(f"Layer: {name}")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"{'='*70}")
        
        # Convert to numpy and quantize to int8 (like actual deployment)
        weight_np = tensor.detach().cpu().numpy()
        
        # Show original statistics
        print(f"\nOriginal (FP16) stats:")
        print(f"  Value range: [{weight_np.min():.4f}, {weight_np.max():.4f}]")
        print(f"  Mean: {weight_np.mean():.4f}, Std: {weight_np.std():.4f}")
        
        # Quantize to INT8 (simulate real deployment scenario)
        # Standard symmetric quantization: scale = max(abs(weight)) / 127
        scale = np.abs(weight_np).max() / 127.0
        weight_int8 = np.clip(np.round(weight_np / scale), -128, 127).astype(np.int8)
        
        print(f"\nQuantized (INT8) stats:")
        print(f"  Value range: [{weight_int8.min()}, {weight_int8.max()}]")
        print(f"  Mean: {weight_int8.mean():.2f}, Std: {weight_int8.std():.2f}")
        print(f"  Quantization scale: {scale:.6f}")
        
        # Analyze distribution
        unique, counts = np.unique(weight_int8, return_counts=True)
        top_5 = sorted(zip(counts, unique), reverse=True)[:5]
        print(f"  Most common values:")
        for count, val in top_5:
            freq = count / weight_int8.size * 100
            print(f"    {val:4d}: {freq:5.2f}% ({count} occurrences)")
        
        # Reshape to 256x256 tiles for testing
        # Take a representative 256x256 patch
        flat = weight_int8.flatten()
        if len(flat) < 256*256:
            # Pad if too small
            test_data = np.zeros((256, 256), dtype=np.int8)
            test_data.flat[:len(flat)] = flat
        else:
            # Take first 256x256
            test_data = flat[:256*256].reshape(256, 256)
        
        # Test compression
        result = test_compression_on_weights(test_data, f"{layer_desc} ({name})")
        results.append((layer_desc, name, result))
    
    # Summary
    print("\n" + "="*70)
    print("COMPRESSION SUMMARY")
    print("="*70)
    
    total_input = 0
    total_compressed = 0
    
    for layer_desc, name, result in results:
        if result:
            input_size, compressed_size, ratio = result
            total_input += input_size
            total_compressed += compressed_size
            print(f"{layer_desc:20s}: {ratio:.3f}x ({(1-1/ratio)*100:.1f}% saved)")
    
    if total_input > 0:
        overall_ratio = total_input / total_compressed
        print(f"\n{'Overall Average':20s}: {overall_ratio:.3f}x ({(1-1/overall_ratio)*100:.1f}% saved)")
    
    return True

def test_compression_on_weights(weights, description):
    """Test codec on weight matrix"""
    
    try:
        lib = ctypes.CDLL("build/libcodec_core.so")
        
        # Create encoder
        encoder = lib.encoder_create(256)
        
        # Encode
        data_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        output_ptr = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_size_t()
        
        ratio = lib.encoder_encode(encoder, data_ptr, 256, 256,
                                  ctypes.byref(output_ptr), ctypes.byref(output_size))
        
        input_size = weights.nbytes
        compressed_size = output_size.value
        
        print(f"\n--- Compression Results ---")
        print(f"Input size:      {input_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Compression ratio: {input_size / compressed_size:.3f}x")
        print(f"Space saved: {(1 - compressed_size/input_size)*100:.1f}%")
        
        # Decode and verify
        if lib.decoder_is_available():
            decoder = lib.decoder_create()
            decoded = np.zeros((256, 256), dtype=np.int8)
            decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
            
            lib.decoder_decode(decoder, output_ptr, output_size.value, decoded_ptr)
            
            # Check reconstruction
            matches = np.array_equal(weights, decoded)
            if matches:
                print(f"✓ Bit-exact reconstruction!")
            else:
                errors = np.abs(weights.astype(np.int32) - decoded.astype(np.int32))
                print(f"✗ Reconstruction error:")
                print(f"  Max error: {errors.max()}")
                print(f"  Mean error: {errors.mean():.2f}")
            
            lib.decoder_destroy(decoder)
        
        lib.free_buffer(output_ptr)
        lib.encoder_destroy(encoder)
        
        return (input_size, compressed_size, input_size / compressed_size)
        
    except Exception as e:
        print(f"Error during compression: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    success = download_and_test_llama()
    sys.exit(0 if success else 1)

