# nvCOMP 5.0 API Breaking Changes

## Problem
nvCOMP 5.0 has completely different API signatures than 4.x/3.x:

### Old API (v4.x):
```cpp
nvcompBatchedZstdDecompressGetTempSize(
    size_t max_chunk_size,
    size_t batch_size,
    size_t* temp_bytes
)
```

### New API (v5.0):
```cpp
nvcompBatchedZstdDecompressGetTempSizeSync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t max_uncompressed_chunk_size,
    size_t batch_size,
    size_t* temp_bytes,
    size_t max_temp_bytes,
    nvcompBatchedZstdDecompressOpts_t format_opts,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream
)
```

## Solutions

### Option 1: Downgrade to nvCOMP 3.0
Use older stable API that matches our code.

### Option 2: Use Manager API
nvCOMP 5.0 provides high-level Manager classes that hide these details.

### Option 3: Rewrite for v5.0
Adapt our code to the new batched API (complex).

### Option 4: CPU Zstd fallback
Use CPU Zstd decompression (still very fast, ~200 MB/s).

## Recommendation
Use **Option 1** (downgrade to nvCOMP 3.0) or **Option 2** (Manager API) for simplest path forward.

