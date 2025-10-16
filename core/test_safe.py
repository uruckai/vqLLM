#!/usr/bin/env python3
"""
Safe test script with better error handling
"""

import sys
import traceback

def test_basic_codec():
    """Test basic codec functionality"""
    print("\n" + "="*80)
    print("TEST 1: Basic Codec Test")
    print("="*80)
    
    try:
        import numpy as np
        import ctypes
        from pathlib import Path
        
        # Load library
        lib_path = Path("build/libcodec_core.so")
        if not lib_path.exists():
            print("✗ Library not found")
            return False
        
        lib = ctypes.CDLL(str(lib_path))
        lib.encoder_create.restype = ctypes.c_void_p
        lib.decoder_create.restype = ctypes.c_void_p
        lib.decoder_is_available.restype = ctypes.c_bool
        
        print("✓ Library loaded")
        
        if not lib.decoder_is_available():
            print("✗ GPU decoder not available")
            return False
        
        print("✓ GPU decoder available")
        
        # Test encode/decode
        print("\nTesting encode/decode on 256x256 data...")
        data = np.random.randint(-128, 127, (256, 256), dtype=np.int8)
        
        encoder = lib.encoder_create(256)
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        output_ptr = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_size_t()
        
        lib.encoder_encode(encoder, data_ptr, 256, 256,
                          ctypes.byref(output_ptr), ctypes.byref(output_size))
        
        print(f"  Input: {data.nbytes} bytes")
        print(f"  Compressed: {output_size.value} bytes")
        print(f"  Ratio: {data.nbytes / output_size.value:.2f}x")
        
        # Decode
        decoder = lib.decoder_create()
        decoded = np.zeros((256, 256), dtype=np.int8)
        decoded_ptr = decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        lib.decoder_decode(decoder, output_ptr, output_size.value, decoded_ptr)
        
        # Verify
        if np.array_equal(data, decoded):
            print("✓ Bit-exact reconstruction!")
        else:
            print("✗ Reconstruction failed")
            return False
        
        # Cleanup
        lib.free_buffer(output_ptr)
        lib.encoder_destroy(encoder)
        lib.decoder_destroy(decoder)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_transformers_import():
    """Test if transformers can be imported"""
    print("\n" + "="*80)
    print("TEST 2: Transformers Import")
    print("="*80)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ Transformers imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_model_download():
    """Test model download (but don't run inference)"""
    print("\n" + "="*80)
    print("TEST 3: Model Download (No Inference)")
    print("="*80)
    
    try:
        import torch
        from transformers import AutoTokenizer
        import os
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Downloading tokenizer for {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded")
        
        # Don't load model yet - just test tokenizer
        test_text = "Hello world"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✓ Tokenization works: '{test_text}' → {tokens['input_ids'].shape[1]} tokens")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("SAFE DIAGNOSTIC TESTS")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Codec Test", test_basic_codec()))
    results.append(("Transformers Import", test_transformers_import()))
    results.append(("Model Download", test_model_download()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ All tests passed! System is working.")
        print("\nThe segfault is likely in the model inference code.")
        print("This might be a PyTorch/CUDA compatibility issue.")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

