#!/usr/bin/env python3
"""
Debug test for batched codec - prints detailed info
"""
import numpy as np
import ctypes
import struct

# Load library
lib = ctypes.CDLL('./build/libcodec_core.so')

# Encoder
lib.batched_encoder_create.argtypes = [ctypes.c_uint16]
lib.batched_encoder_create.restype = ctypes.c_void_p
lib.batched_encoder_destroy.argtypes = [ctypes.c_void_p]
lib.batched_encoder_encode_layer.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t)
]
lib.batched_encoder_encode_layer.restype = ctypes.c_float

# Decoder
lib.batched_decoder_create.restype = ctypes.c_void_p
lib.batched_decoder_destroy.argtypes = [ctypes.c_void_p]
lib.batched_decoder_decode_layer.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int8)
]
lib.batched_decoder_decode_layer.restype = ctypes.c_float

lib.batched_free_buffer.argtypes = [ctypes.POINTER(ctypes.c_uint8)]

print("=== Batched Codec Debug Test ===")

# Create TINY test data (one tile)
rows, cols = 256, 256
data = np.random.randint(-100, 100, (rows, cols), dtype=np.int8)
data_contiguous = np.ascontiguousarray(data)

print(f"Original: ({rows}, {cols}), {data.nbytes} bytes")
print(f"Sample values: {data[0, :8]}")

# Encode
encoder = lib.batched_encoder_create(256)  # tile_size
compressed_ptr = ctypes.POINTER(ctypes.c_uint8)()
compressed_size = ctypes.c_size_t()

ratio = lib.batched_encoder_encode_layer(
    encoder,
    data_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    rows, cols,
    ctypes.byref(compressed_ptr),
    ctypes.byref(compressed_size)
)

print(f"Compressed: {compressed_size.value} bytes")
print(f"Ratio: {ratio:.2f}x")

# Read compressed data
compressed_bytes = bytes(ctypes.cast(compressed_ptr, ctypes.POINTER(ctypes.c_uint8 * compressed_size.value)).contents)

# Parse header
header_size = 32  # LayerHeader size
rows_read, cols_read, tile_size, num_tiles = struct.unpack_from('IIII', compressed_bytes, 0)
print(f"\nHeader:")
print(f"  rows={rows_read}, cols={cols_read}, tile_size={tile_size}, num_tiles={num_tiles}")

# Find RANS table
rans_table_offset = header_size
rans_table_size = 1024
print(f"\nRANS table at offset {rans_table_offset}, size {rans_table_size}")

# Find tile index
tile_index_offset_val = struct.unpack_from('Q', compressed_bytes, 16)[0]
tile_index_size = 20  # TileIndexEntry size
print(f"\nTile index at offset {tile_index_offset_val}")

# Read first tile index entry
tile_offset, tile_comp_size, tile_row, tile_col = struct.unpack_from('QIII', compressed_bytes, tile_index_offset_val)
tile_data_offset_val = struct.unpack_from('Q', compressed_bytes, 24)[0]
absolute_tile_offset = tile_data_offset_val + tile_offset

print(f"\nTile 0:")
print(f"  offset={tile_offset}, compressed_size={tile_comp_size}")
print(f"  absolute offset={absolute_tile_offset}")

# Read tile data
tile_data = compressed_bytes[absolute_tile_offset:absolute_tile_offset + tile_comp_size]
print(f"  tile data length: {len(tile_data)}")

# Parse tile
uncompressed_size = struct.unpack_from('I', tile_data, 0)[0]
print(f"  uncompressed size from header: {uncompressed_size}")

# Read state from end
state_bytes = tile_data[-4:]
state = struct.unpack('<I', state_bytes)[0]
print(f"  state at end: 0x{state:08x} ({state})")

# Show first few encoded bytes (after size header, before state)
encoded_data = tile_data[4:-4]
print(f"  encoded data length: {len(encoded_data)}")
print(f"  first 8 encoded bytes: {list(encoded_data[:8])}")
print(f"  last 8 encoded bytes (before state): {list(encoded_data[-8:])}")

# Now decode
decoder = lib.batched_decoder_create()
decoded = np.zeros((rows, cols), dtype=np.int8)

decode_time = lib.batched_decoder_decode_layer(
    decoder,
    compressed_ptr,
    compressed_size,
    decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
)

print(f"\nDecode time: {decode_time:.2f} ms")
print(f"Decoded sample values: {decoded[0, :8]}")

# Check
errors = np.sum(data != decoded)
print(f"Bit-exact: {errors == 0}")
if errors > 0:
    print(f"  Errors: {errors} / {data.size}")
    # Show first mismatch
    mismatch_idx = np.where(data != decoded)
    if len(mismatch_idx[0]) > 0:
        r, c = mismatch_idx[0][0], mismatch_idx[1][0]
        print(f"  First mismatch at ({r}, {c}): expected {data[r,c]}, got {decoded[r,c]}")

# Cleanup
lib.batched_free_buffer(compressed_ptr)
lib.batched_encoder_destroy(encoder)
lib.batched_decoder_destroy(decoder)

