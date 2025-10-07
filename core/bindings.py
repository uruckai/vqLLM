"""
Python bindings for core codec
"""

import ctypes
import numpy as np
from pathlib import Path

# Load library
lib_path = Path(__file__).parent / "build" / "libcodec_core.so"
if not lib_path.exists():
    # Try relative to repo root
    lib_path = Path(__file__).parent.parent / "build" / "libcodec_core.so"

lib = ctypes.CDLL(str(lib_path))

# C API definitions
lib.encoder_create.restype = ctypes.c_void_p
lib.encoder_create.argtypes = [ctypes.c_uint16]

lib.encoder_destroy.argtypes = [ctypes.c_void_p]

lib.encoder_encode.restype = ctypes.c_float
lib.encoder_encode.argtypes = [
    ctypes.c_void_p,  # encoder
    ctypes.POINTER(ctypes.c_int8),  # data
    ctypes.c_uint32,  # rows
    ctypes.c_uint32,  # cols
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # output
    ctypes.POINTER(ctypes.c_size_t)  # output_size
]

lib.decoder_create.restype = ctypes.c_void_p
lib.decoder_create.argtypes = []

lib.decoder_destroy.argtypes = [ctypes.c_void_p]

lib.decoder_decode.restype = ctypes.c_float
lib.decoder_decode.argtypes = [
    ctypes.c_void_p,  # decoder
    ctypes.POINTER(ctypes.c_uint8),  # compressed
    ctypes.c_size_t,  # compressed_size
    ctypes.POINTER(ctypes.c_int8)  # output
]

lib.decoder_is_available.restype = ctypes.c_bool
lib.decoder_is_available.argtypes = []

lib.free_buffer.argtypes = [ctypes.POINTER(ctypes.c_uint8)]


class Encoder:
    """CPU Encoder for INT8 weights"""
    
    def __init__(self, tile_size=16):
        self.handle = lib.encoder_create(tile_size)
        if not self.handle:
            raise RuntimeError("Failed to create encoder")
    
    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            lib.encoder_destroy(self.handle)
    
    def encode(self, data):
        """
        Encode INT8 weight matrix
        
        Args:
            data: numpy array (int8, 2D)
        
        Returns:
            compressed: bytes
            ratio: compression ratio
        """
        if not isinstance(data, np.ndarray) or data.dtype != np.int8:
            raise ValueError("Data must be numpy int8 array")
        
        if data.ndim != 2:
            raise ValueError("Data must be 2D")
        
        rows, cols = data.shape
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        output_ptr = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_size_t()
        
        ratio = lib.encoder_encode(
            self.handle, data_ptr, rows, cols,
            ctypes.byref(output_ptr), ctypes.byref(output_size)
        )
        
        if ratio < 0:
            raise RuntimeError("Encoding failed")
        
        # Copy to Python bytes
        compressed = bytes(ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_uint8 * output_size.value)).contents)
        
        # Free C buffer
        lib.free_buffer(output_ptr)
        
        return compressed, ratio


class GPUDecoder:
    """GPU Decoder for compressed weights"""
    
    def __init__(self):
        if not lib.decoder_is_available():
            raise RuntimeError("GPU not available")
        
        self.handle = lib.decoder_create()
        if not self.handle:
            raise RuntimeError("Failed to create decoder")
    
    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            lib.decoder_destroy(self.handle)
    
    def decode(self, compressed, rows, cols):
        """
        Decode compressed data on GPU
        
        Args:
            compressed: bytes (compressed data)
            rows: output rows
            cols: output cols
        
        Returns:
            output: numpy array (int8, 2D)
            time_ms: decode time in milliseconds
        """
        output = np.zeros((rows, cols), dtype=np.int8)
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        compressed_array = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
        compressed_ptr = ctypes.cast(compressed_array, ctypes.POINTER(ctypes.c_uint8))
        
        time_ms = lib.decoder_decode(
            self.handle, compressed_ptr, len(compressed), output_ptr
        )
        
        if time_ms < 0:
            raise RuntimeError("Decoding failed")
        
        return output, time_ms
    
    @staticmethod
    def is_available():
        """Check if GPU is available"""
        return lib.decoder_is_available()


def test_roundtrip():
    """Quick test of encode/decode"""
    print("=== Core Codec Test ===")
    
    # Generate test data
    data = np.random.randint(-128, 127, (256, 256), dtype=np.int8)
    print(f"Original: {data.shape}, {data.nbytes} bytes")
    
    # Encode
    encoder = Encoder(tile_size=16)
    compressed, ratio = encoder.encode(data)
    print(f"Compressed: {len(compressed)} bytes")
    print(f"Ratio: {ratio:.2f}x")
    
    # Decode
    if GPUDecoder.is_available():
        decoder = GPUDecoder()
        decoded, time_ms = decoder.decode(compressed, 256, 256)
        print(f"Decode time: {time_ms:.2f} ms")
        print(f"Bit-exact: {np.array_equal(data, decoded)}")
        
        if not np.array_equal(data, decoded):
            diff = np.sum(data != decoded)
            print(f"  Errors: {diff} / {data.size}")
    else:
        print("GPU not available, skipping decode test")


if __name__ == "__main__":
    test_roundtrip()

