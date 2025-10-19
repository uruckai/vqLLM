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

# Parse header (according to format_batched.h)
# struct LayerHeader {
#     uint32_t magic;              // offset 0
#     uint16_t version;            // offset 4
#     uint16_t tile_size;          // offset 6
#     uint32_t rows;               // offset 8
#     uint32_t cols;               // offset 12
#     uint16_t num_tiles_row;      // offset 16
#     uint16_t num_tiles_col;      // offset 18
#     uint32_t num_tiles;          // offset 20
#     uint32_t compressed_size;    // offset 24
#     uint32_t tile_index_offset;  // offset 28
#     uint32_t tile_data_offset;   // offset 32
#     uint8_t predictor_mode;      // offset 36
#     uint8_t padding[3];          // offset 37-39
# };
header_fmt = 'IHHIIHHI IIIBxxx'  # Total 40 bytes
(magic, version, tile_size, rows_read, cols_read, num_tiles_row, num_tiles_col, num_tiles,
 compressed_size, tile_index_offset_val, tile_data_offset_val, predictor_mode) = struct.unpack_from(header_fmt, compressed_bytes, 0)

print(f"\nHeader:")
print(f"  magic=0x{magic:08x}, version={version}")
print(f"  rows={rows_read}, cols={cols_read}, tile_size={tile_size}")
print(f"  num_tiles={num_tiles} ({num_tiles_row}x{num_tiles_col})")
print(f"  compressed_size={compressed_size}")
print(f"  tile_index_offset={tile_index_offset_val}")
print(f"  tile_data_offset={tile_data_offset_val}")

# Find RANS table
rans_table_offset = 40  # After header
rans_table_size = 1024
print(f"\nRANS table at offset {rans_table_offset}, size {rans_table_size}")

# Tile index info
tile_index_size = 12  # TileIndexEntry: uint32_t + uint32_t + uint16_t + uint16_t = 12 bytes
print(f"\nTile index at offset {tile_index_offset_val}, size {tile_index_size} bytes/entry")

# Read first tile index entry
# struct TileIndexEntry { uint32_t offset; uint32_t compressed_size; uint16_t row; uint16_t col; };
tile_offset, tile_comp_size, tile_row, tile_col = struct.unpack_from('IIHH', compressed_bytes, tile_index_offset_val)
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

