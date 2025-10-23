# nvCOMP 3.0.6 vs 5.0 API Differences

## Key Differences

### Compression

**nvCOMP 5.0:**
```cpp
nvcompBatchedZstdCompressGetTempSizeAsync(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedZstdCompressOpts_t opts,
    size_t* temp_bytes,
    size_t max_total_uncompressed_bytes);
```

**nvCOMP 3.0.6:**
```cpp
nvcompBatchedZstdCompressGetTempSize(
    size_t batch_size,
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedZstdOpts_t opts,  // Note: different type name!
    size_t* temp_bytes);
```

### Decompression

**nvCOMP 5.0:**
```cpp
nvcompBatchedZstdDecompressGetTempSizeAsync(
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    nvcompBatchedZstdDecompressOpts_t opts,
    size_t* temp_bytes,
    size_t max_total_uncompressed_bytes,
    cudaStream_t stream);
```

**nvCOMP 3.0.6:**
```cpp
nvcompBatchedZstdDecompressGetTempSize(
    size_t num_chunks,
    size_t max_uncompressed_chunk_size,
    size_t* temp_bytes);

// OR with total size:
nvcompBatchedZstdDecompressGetTempSizeEx(
    size_t num_chunks,
    size_t max_uncompressed_chunk_size,
    size_t* temp_bytes,
    size_t max_uncompressed_total_size);
```

## Summary
- v3.0.6 is **simpler** - no Async suffix, fewer parameters
- v3.0.6 uses `nvcompBatchedZstdOpts_t` (no "Compress" in the name)
- v3.0.6 temp size queries don't need opts struct
- v3.0.6 has recommended chunk size of **64 KB** (not 65536)
- v3.0.6 max chunk size is **16 MB** (1 << 24)

