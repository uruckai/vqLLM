"""
ctypes bindings for Weight Codec C API
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def _find_library():
    """Find the wcodec shared library"""
    search_paths = [
        Path(__file__).parent.parent.parent / "build" / "libwcodec.so",  # Linux
        Path(__file__).parent.parent.parent / "build" / "libwcodec.dylib",  # macOS
        Path(__file__).parent.parent.parent / "build" / "Release" / "wcodec.dll",  # Windows
        Path(__file__).parent.parent.parent / "build" / "Debug" / "wcodec.dll",
    ]
    
    for path in search_paths:
        if path.exists():
            return str(path)
    return None


# Load library
_lib_path = _find_library()
if _lib_path is None:
    raise RuntimeError("libwcodec not found. Please build the C++ library first.")

_lib = ctypes.CDLL(_lib_path)

# Define C structures
class _Encoder(ctypes.Structure):
    pass

class _Decoder(ctypes.Structure):
    pass

# Define function signatures
_lib.wcodec_encoder_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_lib.wcodec_encoder_create.restype = ctypes.POINTER(_Encoder)

_lib.wcodec_encoder_destroy.argtypes = [ctypes.POINTER(_Encoder)]
_lib.wcodec_encoder_destroy.restype = None

_lib.wcodec_encode_layer.argtypes = [
    ctypes.POINTER(_Encoder),
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
]
_lib.wcodec_encode_layer.restype = ctypes.c_int

_lib.wcodec_decoder_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_lib.wcodec_decoder_create.restype = ctypes.POINTER(_Decoder)

_lib.wcodec_decoder_destroy.argtypes = [ctypes.POINTER(_Decoder)]
_lib.wcodec_decoder_destroy.restype = None

_lib.wcodec_decode_layer.argtypes = [
    ctypes.POINTER(_Decoder),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_double)
]
_lib.wcodec_decode_layer.restype = ctypes.c_int

_lib.wcodec_free_buffer.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
_lib.wcodec_free_buffer.restype = None

# GPU decoder functions
class _GPUDecoder(ctypes.Structure):
    pass

_lib.wcodec_gpu_decoder_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_lib.wcodec_gpu_decoder_create.restype = ctypes.POINTER(_GPUDecoder)

_lib.wcodec_gpu_decoder_destroy.argtypes = [ctypes.POINTER(_GPUDecoder)]
_lib.wcodec_gpu_decoder_destroy.restype = None

_lib.wcodec_gpu_decode_layer.argtypes = [
    ctypes.POINTER(_GPUDecoder),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int)
]
_lib.wcodec_gpu_decode_layer.restype = ctypes.c_int

_lib.wcodec_gpu_is_available.argtypes = []
_lib.wcodec_gpu_is_available.restype = ctypes.c_int


class Encoder:
    """Encoder wrapper"""
    
    def __init__(self, tile_size: int = 16):
        self._handle = _lib.wcodec_encoder_create(tile_size, tile_size)
        if not self._handle:
            raise RuntimeError("Failed to create encoder")
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.wcodec_encoder_destroy(self._handle)
    
    def encode_layer(self, data: np.ndarray) -> Tuple[bytes, dict]:
        """
        Encode a layer.
        
        Args:
            data: 2D numpy array (int8)
        
        Returns:
            tuple: (encoded_bytes, stats_dict)
        """
        if data.dtype != np.int8:
            raise ValueError(f"Expected int8 data, got {data.dtype}")
        
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        
        rows, cols = data.shape
        
        # Prepare output pointers
        output_ptr = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_size_t()
        compression_ratio = ctypes.c_double()
        encode_time_ms = ctypes.c_double()
        
        # Call C function
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        result = _lib.wcodec_encode_layer(
            self._handle,
            data_ptr,
            rows,
            cols,
            ctypes.byref(output_ptr),
            ctypes.byref(output_size),
            ctypes.byref(compression_ratio),
            ctypes.byref(encode_time_ms)
        )
        
        if result != 0:
            raise RuntimeError("Encoding failed")
        
        # Copy output to Python bytes
        output_bytes = bytes(ctypes.cast(
            output_ptr,
            ctypes.POINTER(ctypes.c_uint8 * output_size.value)
        ).contents)
        
        # Free C buffer
        _lib.wcodec_free_buffer(output_ptr)
        
        stats = {
            'original_bytes': rows * cols,
            'compressed_bytes': output_size.value,
            'compression_ratio': compression_ratio.value,
            'encode_time_ms': encode_time_ms.value,
            'compression_percent': (1 - output_size.value / (rows * cols)) * 100
        }
        
        return output_bytes, stats


class Decoder:
    """Decoder wrapper"""
    
    def __init__(self, tile_size: int = 16):
        self._handle = _lib.wcodec_decoder_create(tile_size, tile_size)
        if not self._handle:
            raise RuntimeError("Failed to create decoder")
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.wcodec_decoder_destroy(self._handle)
    
    def decode_layer(self, data: bytes, rows: int, cols: int) -> Tuple[np.ndarray, dict]:
        """
        Decode a layer.
        
        Args:
            data: Encoded bytes
            rows: Expected number of rows
            cols: Expected number of columns
        
        Returns:
            tuple: (decoded_array, stats_dict)
        """
        # Prepare output array
        output = np.zeros((rows, cols), dtype=np.int8)
        
        # Prepare input
        input_size = len(data)
        input_ptr = (ctypes.c_uint8 * input_size).from_buffer_copy(data)
        
        # Decode
        decode_time_ms = ctypes.c_double()
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        result = _lib.wcodec_decode_layer(
            self._handle,
            input_ptr,
            input_size,
            rows,
            cols,
            output_ptr,
            ctypes.byref(decode_time_ms)
        )
        
        if result != 0:
            raise RuntimeError("Decoding failed")
        
        stats = {
            'compressed_bytes': input_size,
            'decompressed_bytes': rows * cols,
            'decode_time_ms': decode_time_ms.value
        }
        
        return output, stats


class GPUDecoder:
    """GPU-accelerated decoder wrapper"""
    
    def __init__(self, tile_size: int = 16):
        self._handle = _lib.wcodec_gpu_decoder_create(tile_size, tile_size)
        if not self._handle:
            raise RuntimeError("Failed to create GPU decoder")
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.wcodec_gpu_decoder_destroy(self._handle)
    
    def decode_layer(self, data: bytes, rows: int, cols: int) -> Tuple[np.ndarray, dict]:
        """
        Decode a layer (GPU-accelerated if available, otherwise CPU)
        
        Args:
            data: Encoded bytes
            rows: Expected number of rows
            cols: Expected number of columns
        
        Returns:
            tuple: (decoded_array, stats_dict)
        """
        # Prepare output array
        output = np.zeros((rows, cols), dtype=np.int8)
        
        # Prepare input
        input_size = len(data)
        input_ptr = (ctypes.c_uint8 * input_size).from_buffer_copy(data)
        
        # Decode
        decode_time_ms = ctypes.c_double()
        used_gpu = ctypes.c_int()
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        result = _lib.wcodec_gpu_decode_layer(
            self._handle,
            input_ptr,
            input_size,
            rows,
            cols,
            output_ptr,
            ctypes.byref(decode_time_ms),
            ctypes.byref(used_gpu)
        )
        
        if result != 0:
            raise RuntimeError("GPU decoding failed")
        
        stats = {
            'compressed_bytes': input_size,
            'decompressed_bytes': rows * cols,
            'decode_time_ms': decode_time_ms.value,
            'used_gpu': bool(used_gpu.value),
            'device': 'GPU' if used_gpu.value else 'CPU (fallback)'
        }
        
        return output, stats


def is_gpu_available() -> bool:
    """Check if GPU decode is available"""
    try:
        return bool(_lib.wcodec_gpu_is_available())
    except:
        return False
