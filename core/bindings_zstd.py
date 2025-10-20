#!/usr/bin/env python3
"""
Python bindings for Zstd-based compression/decompression

This is an alternative to the rANS-based bindings, using Zstd compression
for faster GPU decode via nvCOMP library.
"""

import ctypes
import numpy as np
from pathlib import Path

# Load shared library
lib_path = Path(__file__).parent / "build" / "libcodec_core.so"
if not lib_path.exists():
    # Try alternative locations
    lib_path = Path(__file__).parent / "build" / "libcodec_core.dll"
    if not lib_path.exists():
        lib_path = Path(__file__).parent / "build" / "Release" / "codec_core.dll"

if not lib_path.exists():
    raise RuntimeError(f"Cannot find codec library at {lib_path}")

lib = ctypes.CDLL(str(lib_path))

# ============================================================================
# Encoder API
# ============================================================================

lib.zstd_encoder_create.argtypes = [ctypes.c_int]
lib.zstd_encoder_create.restype = ctypes.c_void_p

lib.zstd_encoder_destroy.argtypes = [ctypes.c_void_p]
lib.zstd_encoder_destroy.restype = None

lib.zstd_encoder_encode_layer.argtypes = [
    ctypes.c_void_p,  # encoder
    ctypes.POINTER(ctypes.c_int8),  # data
    ctypes.c_uint32,  # rows
    ctypes.c_uint32,  # cols
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # output
    ctypes.POINTER(ctypes.c_size_t)  # output_size
]
lib.zstd_encoder_encode_layer.restype = ctypes.c_float

# ============================================================================
# Decoder API
# ============================================================================

lib.zstd_decoder_create.argtypes = []
lib.zstd_decoder_create.restype = ctypes.c_void_p

lib.zstd_decoder_destroy.argtypes = [ctypes.c_void_p]
lib.zstd_decoder_destroy.restype = None

lib.zstd_decoder_is_available.argtypes = []
lib.zstd_decoder_is_available.restype = ctypes.c_int

lib.zstd_decoder_decode_layer.argtypes = [
    ctypes.c_void_p,  # decoder
    ctypes.POINTER(ctypes.c_uint8),  # compressed_data
    ctypes.c_size_t,  # compressed_size
    ctypes.POINTER(ctypes.c_int8),  # output
    ctypes.POINTER(ctypes.c_uint32),  # rows
    ctypes.POINTER(ctypes.c_uint32)  # cols
]
lib.zstd_decoder_decode_layer.restype = ctypes.c_int

lib.zstd_decoder_parse_header.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # compressed_data
    ctypes.c_size_t,  # compressed_size
    ctypes.POINTER(ctypes.c_uint32),  # rows
    ctypes.POINTER(ctypes.c_uint32),  # cols
    ctypes.POINTER(ctypes.c_uint32)  # uncompressed_size
]
lib.zstd_decoder_parse_header.restype = ctypes.c_int

lib.zstd_decoder_decode_layer_to_gpu.argtypes = [
    ctypes.c_void_p,  # decoder
    ctypes.POINTER(ctypes.c_uint8),  # compressed_data
    ctypes.c_size_t,  # compressed_size
    ctypes.POINTER(ctypes.c_uint32),  # rows
    ctypes.POINTER(ctypes.c_uint32)  # cols
]
lib.zstd_decoder_decode_layer_to_gpu.restype = ctypes.c_void_p

# ============================================================================
# Python wrapper classes
# ============================================================================

class ZstdEncoder:
    """Zstd encoder for layer compression"""
    
    def __init__(self, compression_level=9):
        """
        Args:
            compression_level: Zstd compression level (1-22)
                             1 = fastest, lower compression
                             9 = balanced (recommended)
                             22 = slowest, best compression
        """
        self.handle = lib.zstd_encoder_create(compression_level)
        if not self.handle:
            raise RuntimeError("Failed to create Zstd encoder")
    
    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            lib.zstd_encoder_destroy(self.handle)
    
    def encode_layer(self, data):
        """
        Encode a layer using Zstd compression
        
        Args:
            data: numpy array (int8, 2D)
        
        Returns:
            (compressed_bytes, compression_ratio)
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be numpy array")
        
        if data.dtype != np.int8:
            raise TypeError("Input must be int8")
        
        if data.ndim != 2:
            raise ValueError("Input must be 2D")
        
        rows, cols = data.shape
        
        # Call C function
        output_ptr = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_size_t()
        
        ratio = lib.zstd_encoder_encode_layer(
            self.handle,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            rows,
            cols,
            ctypes.byref(output_ptr),
            ctypes.byref(output_size)
        )
        
        if ratio < 0:
            raise RuntimeError("Zstd encoding failed")
        
        # Copy to Python bytes
        compressed = bytes(output_ptr[:output_size.value])
        
        # Free C-allocated memory
        ctypes.CDLL(None).free(output_ptr)
        
        return compressed, ratio


class ZstdGPUDecoder:
    """GPU-accelerated Zstd decoder using nvCOMP"""
    
    def __init__(self):
        self.handle = lib.zstd_decoder_create()
        if not self.handle:
            raise RuntimeError("Failed to create Zstd decoder")
    
    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            lib.zstd_decoder_destroy(self.handle)
    
    @staticmethod
    def is_available():
        """Check if GPU decoder is available"""
        return bool(lib.zstd_decoder_is_available())
    
    def decode_layer(self, compressed_data):
        """
        Decode a compressed layer to CPU memory
        
        Args:
            compressed_data: bytes or numpy array (uint8)
        
        Returns:
            numpy array (int8, 2D)
        """
        if isinstance(compressed_data, np.ndarray):
            compressed_bytes = compressed_data.tobytes()
        else:
            compressed_bytes = bytes(compressed_data)
        
        # Parse header first
        rows = ctypes.c_uint32()
        cols = ctypes.c_uint32()
        uncompressed_size = ctypes.c_uint32()
        
        success = lib.zstd_decoder_parse_header(
            (ctypes.c_uint8 * len(compressed_bytes)).from_buffer_copy(compressed_bytes),
            len(compressed_bytes),
            ctypes.byref(rows),
            ctypes.byref(cols),
            ctypes.byref(uncompressed_size)
        )
        
        if not success:
            raise RuntimeError("Failed to parse Zstd header")
        
        # Allocate output
        output = np.zeros(uncompressed_size.value, dtype=np.int8)
        
        # Decode
        rows_out = ctypes.c_uint32()
        cols_out = ctypes.c_uint32()
        
        success = lib.zstd_decoder_decode_layer(
            self.handle,
            (ctypes.c_uint8 * len(compressed_bytes)).from_buffer_copy(compressed_bytes),
            len(compressed_bytes),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ctypes.byref(rows_out),
            ctypes.byref(cols_out)
        )
        
        if not success:
            raise RuntimeError("Zstd decoding failed")
        
        # Reshape
        return output.reshape(rows_out.value, cols_out.value)
    
    def decode_layer_to_gpu(self, compressed_data):
        """
        Decode a compressed layer directly to GPU memory (no CPU copy)
        
        Args:
            compressed_data: bytes or numpy array (uint8)
        
        Returns:
            (gpu_ptr, rows, cols, dtype) tuple
            - gpu_ptr: CUDA device pointer (int)
            - rows, cols: dimensions
            - dtype: numpy dtype (int8)
            
            NOTE: Caller MUST call cudaFree on the gpu_ptr when done!
        """
        if isinstance(compressed_data, np.ndarray):
            compressed_bytes = compressed_data.tobytes()
        else:
            compressed_bytes = bytes(compressed_data)
        
        rows = ctypes.c_uint32()
        cols = ctypes.c_uint32()
        
        gpu_ptr = lib.zstd_decoder_decode_layer_to_gpu(
            self.handle,
            (ctypes.c_uint8 * len(compressed_bytes)).from_buffer_copy(compressed_bytes),
            len(compressed_bytes),
            ctypes.byref(rows),
            ctypes.byref(cols)
        )
        
        if not gpu_ptr:
            raise RuntimeError("GPU decoding failed")
        
        return (gpu_ptr, rows.value, cols.value, np.int8)


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("Testing Zstd compression/decompression...")
    print()
    
    # Check GPU availability
    print(f"GPU decoder available: {ZstdGPUDecoder.is_available()}")
    print()
    
    # Test with random data
    print("Test 1: Random data (256x256)")
    data = np.random.randint(-127, 127, size=(256, 256), dtype=np.int8)
    
    encoder = ZstdEncoder(compression_level=9)
    compressed, ratio = encoder.encode_layer(data)
    
    print(f"  Original size: {data.nbytes} bytes")
    print(f"  Compressed size: {len(compressed)} bytes")
    print(f"  Compression ratio: {ratio:.2f}x")
    
    decoder = ZstdGPUDecoder()
    decompressed = decoder.decode_layer(compressed)
    
    if np.array_equal(data, decompressed):
        print(f"  ✓ Bit-exact reconstruction")
    else:
        errors = np.sum(data != decompressed)
        print(f"  ✗ Reconstruction failed: {errors} errors")
    print()
    
    # Test with correlated data (more like NN weights)
    print("Test 2: Correlated data (simulating NN weights)")
    base = np.random.randn(256, 256).astype(np.float32) * 0.02
    data_correlated = np.clip(np.round(base * 127 / 0.1), -127, 127).astype(np.int8)
    
    compressed, ratio = encoder.encode_layer(data_correlated)
    
    print(f"  Original size: {data_correlated.nbytes} bytes")
    print(f"  Compressed size: {len(compressed)} bytes")
    print(f"  Compression ratio: {ratio:.2f}x")
    
    decompressed = decoder.decode_layer(compressed)
    
    if np.array_equal(data_correlated, decompressed):
        print(f"  ✓ Bit-exact reconstruction")
    else:
        errors = np.sum(data_correlated != decompressed)
        print(f"  ✗ Reconstruction failed: {errors} errors")
    
    print()
    print("All tests complete!")

