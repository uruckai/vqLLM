# Week 5: Container Format & Full Integration

## Goal
Complete the `.wcodec` container format and wire up all components for end-to-end functionality.

Target: Encode real checkpoint → decode on GPU → load into PyTorch

---

## Deliverables

### 1. Container Format Specification
- Binary file format for `.wcodec` files
- Header with magic bytes, version, metadata
- Layer index with offsets and sizes
- Per-tile metadata (predictor modes, transform types, frequencies)
- Compressed bitstream layout
- CRC32 checksums for validation

### 2. Container Implementation
- **Writer**: Serialize encoded layers to `.wcodec`
- **Reader**: Parse `.wcodec` and extract metadata
- **Validator**: Verify checksums and format correctness

### 3. GPU Pipeline Integration
- Parse container format
- Extract per-tile metadata
- Transfer to GPU
- Launch CUDA kernels with real data
- Validate bit-exact reconstruction

### 4. Complete PyTorch Integration
- `encode_checkpoint()`: safetensors → .wcodec
- `decode_checkpoint()`: .wcodec → safetensors or state_dict
- Direct GPU loading with zero-copy where possible

### 5. End-to-End Testing
- Encode synthetic checkpoint
- Decode on GPU
- Verify bit-exact vs original
- Measure compression ratio and decode speed

---

## Container Format Layout

```
┌─────────────────────────────────────────┐
│ Magic Bytes: "WCDC" (4 bytes)           │
├─────────────────────────────────────────┤
│ Version: uint16 (2 bytes)               │
├─────────────────────────────────────────┤
│ Header Size: uint32 (4 bytes)           │
├─────────────────────────────────────────┤
│ Metadata (JSON): variable               │
│ - Model info                            │
│ - Quantization type                     │
│ - Tile size                             │
│ - Compression stats                     │
├─────────────────────────────────────────┤
│ Layer Index (N layers):                 │
│   - Name (null-terminated string)       │
│   - Shape (rows, cols)                  │
│   - Offset (uint64)                     │
│   - Compressed size (uint64)            │
│   - Uncompressed size (uint64)          │
│   - CRC32 (uint32)                      │
├─────────────────────────────────────────┤
│ Layer 0 Data:                           │
│   ┌─────────────────────────────────┐   │
│   │ Tile Metadata Array:            │   │
│   │   - Predictor mode (uint8)      │   │
│   │   - Transform type (uint8)      │   │
│   │   - Frequency table (256×uint32)│   │
│   │   - Compressed offset (uint32)  │   │
│   │   - Compressed size (uint32)    │   │
│   └─────────────────────────────────┘   │
│   ┌─────────────────────────────────┐   │
│   │ Compressed Bitstream:           │   │
│   │   - Tile 0 rANS data            │   │
│   │   - Tile 1 rANS data            │   │
│   │   - ...                         │   │
│   └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│ Layer 1 Data: ...                       │
├─────────────────────────────────────────┤
│ Footer:                                 │
│   - File CRC32 (uint32)                 │
│   - Magic Bytes: "CDCW" (4 bytes)       │
└─────────────────────────────────────────┘
```

---

## Implementation Plan

### Day 1: Container Format
- Define binary layout
- Implement serialization/deserialization
- Add validation and checksums

### Day 2: Writer Integration
- Modify encoder to output container format
- Test writing synthetic checkpoints

### Day 3: Reader Integration
- Implement container parser
- Extract metadata for GPU kernels

### Day 4: GPU Wiring
- Connect reader → GPU decoder
- Pass tile metadata to CUDA kernels
- Validate correctness

### Day 5: PyTorch Integration
- Complete `encode_checkpoint()`
- Complete `decode_checkpoint()`
- Test with safetensors files

### Day 6-7: Testing & Optimization
- End-to-end tests
- Performance validation
- Bug fixes

---

## Success Criteria

✅ Can encode safetensors → .wcodec  
✅ Can decode .wcodec → safetensors  
✅ Bit-exact reconstruction verified  
✅ GPU decode working (if CUDA available)  
✅ CPU fallback working  
✅ Compression ratio measured  
✅ Decode speed measured  

---

## Files to Create

```
cpp/src/
  container_writer.cpp         # Write .wcodec files
  container_reader.cpp         # Read .wcodec files

cpp/include/wcodec/
  container_writer.h           # Writer API
  container_reader.h           # Reader API

python/wcodec/
  encoder_api.py              # High-level encode API
  decoder_api.py              # High-level decode API

tests/
  test_container_format.py    # Container format tests
  test_end_to_end.py          # Full pipeline test

scripts/
  encode_checkpoint.py        # CLI tool to encode
  decode_checkpoint.py        # CLI tool to decode
```

