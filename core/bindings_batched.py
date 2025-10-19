"""
Python bindings for batched codec (layer-level compression)
"""

import ctypes
import numpy as np
from pathlib import Path

# Load library
lib_path = Path(__file__).parent / "build" / "libcodec_core.so"
if not lib_path.exists():
    lib_path = Path(__file__).parent.parent / "build" / "libcodec_core.so"

lib = ctypes.CDLL(str(lib_path))

# ===============================
# Batched Encoder API
# ===============================

lib.batched_encoder_create.restype = ctypes.c_void_p
lib.batched_encoder_create.argtypes = [ctypes.c_uint16]

lib.batched_encoder_destroy.argtypes = [ctypes.c_void_p]

lib.batched_encoder_encode_layer.restype = ctypes.c_float
lib.batched_encoder_encode_layer.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t)
]

# ===============================
# Batched Decoder API
# ===============================

lib.batched_decoder_create.restype = ctypes.c_void_p
lib.batched_decoder_create.argtypes = []

lib.batched_decoder_destroy.argtypes = [ctypes.c_void_p]

lib.batched_decoder_decode_layer.restype = ctypes.c_float
lib.batched_decoder_decode_layer.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int8)
]

lib.batched_decoder_is_available.restype = ctypes.c_bool
lib.batched_decoder_is_available.argtypes = []

lib.batched_free_buffer.argtypes = [ctypes.POINTER(ctypes.c_uint8)]


class BatchedEncoder:
    """Layer-level encoder (CPU)"""
    
    def __init__(self, tile_size=256):
        self.handle = lib.batched_encoder_create(tile_size)
        if not self.handle:
            raise RuntimeError("Failed to create batched encoder")
    
    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            lib.batched_encoder_destroy(self.handle)
    
    def encode_layer(self, data):
        """
        Encode entire layer at once
        
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
        
        # Ensure contiguous
        data = np.ascontiguousarray(data)
        
        rows, cols = data.shape
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        output_ptr = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_size_t()
        
        ratio = lib.batched_encoder_encode_layer(
            self.handle, data_ptr, rows, cols,
            ctypes.byref(output_ptr), ctypes.byref(output_size)
        )
        
        if ratio < 0:
            raise RuntimeError("Encoding failed")
        
        # Copy to Python bytes
        compressed = bytes(ctypes.cast(output_ptr, 
                          ctypes.POINTER(ctypes.c_uint8 * output_size.value)).contents)
        
        # Free C buffer
        lib.batched_free_buffer(output_ptr)
        
        return compressed, ratio


class BatchedGPUDecoder:
    """Layer-level GPU decoder"""
    
    def __init__(self):
        if not lib.batched_decoder_is_available():
            raise RuntimeError("GPU not available")
        
        self.handle = lib.batched_decoder_create()
        if not self.handle:
            raise RuntimeError("Failed to create batched decoder")
    
    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            lib.batched_decoder_destroy(self.handle)
    
    def decode_layer(self, compressed, rows, cols):
        """
        Decode entire layer on GPU (all tiles in parallel)
        
        Args:
            compressed: bytes (compressed layer data)
            rows: output rows
            cols: output cols
        
        Returns:
            output: numpy array (int8, 2D)
            time_ms: decode time in milliseconds
        """
        output = np.zeros((rows, cols), dtype=np.int8, order='C')
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        compressed_array = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
        compressed_ptr = ctypes.cast(compressed_array, ctypes.POINTER(ctypes.c_uint8))
        
        time_ms = lib.batched_decoder_decode_layer(
            self.handle, compressed_ptr, len(compressed), output_ptr
        )
        
        if time_ms < 0:
            raise RuntimeError("Decoding failed")
        
        return output, time_ms
    
    @staticmethod
    def is_available():
        """Check if GPU is available"""
        return lib.batched_decoder_is_available()


def test_batched_roundtrip():
    """Quick test of batched encode/decode"""
    print("=== Batched Codec Test ===")
    
    # Generate test data (large layer)
    data = np.random.randint(-128, 127, (2048, 2048), dtype=np.int8)
    print(f"Original: {data.shape}, {data.nbytes} bytes")
    
    # Encode
    encoder = BatchedEncoder(tile_size=256)
    compressed, ratio = encoder.encode_layer(data)
    print(f"Compressed: {len(compressed)} bytes")
    print(f"Ratio: {ratio:.2f}x")
    
    # Decode
    if BatchedGPUDecoder.is_available():
        decoder = BatchedGPUDecoder()
        decoded, time_ms = decoder.decode_layer(compressed, 2048, 2048)
        print(f"Decode time: {time_ms:.2f} ms ({time_ms/64:.2f} ms/tile for 64 tiles)")
        print(f"Bit-exact: {np.array_equal(data, decoded)}")
        
        if not np.array_equal(data, decoded):
            diff = np.sum(data != decoded)
            print(f"  Errors: {diff} / {data.size}")
    else:
        print("GPU not available, skipping decode test")


if __name__ == "__main__":
    test_batched_roundtrip()

