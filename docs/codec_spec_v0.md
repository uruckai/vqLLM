# Weight Codec v0 Specification

**Version:** 0.1.0  
**Date:** 2025-10-04  
**Status:** Draft

## Overview

Weight Codec (WCodec) is a storage codec for quantized neural network weights inspired by video codec techniques (AV1/VP9). It combines predictive coding, transform coding, bitplane representation, and context-adaptive entropy coding (rANS) to achieve 30–60% smaller checkpoint files than flat INT8/INT4 storage, with lossless reconstruction and negligible decode overhead.

**Key properties:**
- Lossless relative to quantized tensors (INT8/INT4)
- Decode on load; zero per-token runtime overhead
- Parallel decode-friendly (tile-based, independent streams)
- Hardware-agnostic container; GPU decode path for NVIDIA

---

## Design Principles

1. **Tiling:** Partition weight matrices into tiles for independent processing and parallel decode
2. **Predictive coding:** Exploit spatial correlation via intra-prediction (like video codecs)
3. **Transform coding:** Decorrelate residuals with integer DCT/ADST
4. **Bitplane representation:** MSB→LSB order enables progressive refinement (future)
5. **Context-adaptive entropy coding:** rANS with local statistics for maximum compression
6. **Load-time decode:** Heavy lifting happens once per layer load, not per inference step

---

## Architecture

```
┌─────────────────┐
│ Quantized       │  INT8/INT4 tensors (e.g., from PTQ, GPTQ, AWQ)
│ Weights         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tile Partition  │  16×16 (default), 8×32, 32×8 based on shape
└────────┬────────┘
         │
    ┌────┴────┐
    │ Per Tile│
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ Intra Predict   │  Modes: left, top, avg, planar
│                 │  → Residual
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transform       │  8×8 sub-blocks: {None, DCT-II, ADST}
│ (optional)      │  Integer approximations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bitplane Pack   │  MSB→LSB; zig-zag scan; band grouping
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Context Model   │  ~32-64 contexts: layer, band, position, neighbors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ rANS Encode     │  Independent streams per tile/tile-group
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Container       │  .wcodec file with header, metadata, streams
└─────────────────┘
```

**Decode path (inverse):**
Container → rANS decode → bitplane unpack → inverse transform → add prediction → reconstructed tile

---

## Tiling Strategy

### Tile Size Selection

| Layer Type | Default Tile | Alt Tile | Rationale |
|------------|--------------|----------|-----------|
| MLP (square-ish) | 16×16 | 8×32, 32×8 | Balance compression & parallelism |
| Attention proj | 16×16 | 8×32 | Often tall/narrow |
| Embeddings | 32×8 | 16×16 | Wide, shallow |

**Rules:**
- If `min(rows, cols) < 16`: use smaller tile or pad
- Tiles are row-major over output channels
- Edge tiles: pad with replication or zero (flag in metadata)

### Tile Metadata

Each tile stores:
- Predictor mode ID (2–3 bits)
- Transform map: per 8×8 sub-block, 2 bits {None=0, DCT=1, ADST=2}
- rANS stream offset and length

---

## Predictive Coding

Exploit spatial correlation in weight matrices by predicting each tile from reconstructed neighbors.

### Predictor Modes

| Mode | Description | Formula |
|------|-------------|---------|
| **LEFT** | Copy left column | `pred[i,j] = recon[i, -1]` |
| **TOP** | Copy top row | `pred[i,j] = recon[-1, j]` |
| **AVG** | Average of left & top | `pred[i,j] = (recon[i,-1] + recon[-1,j]) >> 1` |
| **PLANAR** | Linear extrapolation | `pred[i,j] = a·left + b·top + c` (fit on edges) |

**Mode selection:** Per tile, compute residual for each mode; estimate bits via simple entropy model; choose minimum.

**Edge handling:** For tiles at matrix boundaries without left/top neighbors, use DC mode (mean of available neighbors) or skip prediction.

### Residual Computation

```
residual[i,j] = tile[i,j] - pred[i,j]
```

Residuals are typically smaller and more compressible than raw values.

---

## Transform Coding

Further decorrelate residuals using integer transforms applied to 8×8 sub-blocks within each tile.

### Transform Set

| Transform | Description | Use Case |
|-----------|-------------|----------|
| **None** | Identity (no transform) | Already sparse residuals |
| **DCT-II** | Integer 8×8 DCT (AV1/JPEG style) | General decorrelation |
| **ADST** | Asymmetric DST | Directional residuals |

**Integer transforms:** Use fixed-point integer approximations (e.g., 16-bit intermediate) for bit-exact decode without float ops.

### Transform Selection (RD Probe)

For each 8×8 sub-block:
1. Compute transform candidates: {None, DCT, ADST}
2. For each, estimate bitplane encoding cost (distortion D = 0 for lossless)
3. Choose transform with minimum estimated bits
4. Store 2-bit transform ID per sub-block

**Cost estimation:**
```python
bits = sum(estimate_bits(ctx, bitplane) for bitplane in pack(coeffs))
```

### Coefficient Scanning

- **Zig-zag order:** Low-frequency (top-left) → high-frequency (bottom-right)
- **Band grouping:** Partition zig-zag positions into 3 bands: low (DC + nearby), mid, high
- Bands inform context model (low-freq coeffs use different contexts)

---

## Bitplane Coding

Represent coefficients in binary bitplanes (MSB → LSB) to enable:
- Progressive refinement (future P3)
- Efficient entropy coding (lower planes often sparser)

### Bitplane Structure

For a signed integer coefficient `c`:
```
sign = (c < 0) ? 1 : 0
mag = abs(c)
bitplanes = [MSB, ..., LSB] of mag
```

Encode sign separately; encode magnitude bitplanes MSB-first.

### Significance Propagation (Optional)

Track which coefficients become non-zero at each bitplane:
- First non-zero plane = "significance plane"
- Subsequent planes refine magnitude
- Can use zerotree-like coding (SPIHT-style) for deeper compression (future enhancement)

---

## Context-Adaptive Entropy Coding

Use rANS (range Asymmetric Numeral Systems) with context modeling for maximum compression efficiency.

### Context Model

Maintain ~32–64 contexts per layer, conditioned on:

| Factor | Values | Bits |
|--------|--------|------|
| Layer type | {Embedding, Attn, MLP, Output} | 2 |
| Transform type | {None, DCT, ADST} | 2 |
| Band | {Low, Mid, High} | 2 |
| Position class | {DC, Low-freq, Mid-freq, High-freq} | 2 |
| Neighbor non-zero | {0–3 neighbors non-zero} | 2 |

**Context index:**
```
ctx_id = (layer_type << 8) | (transform << 6) | (band << 4) | (pos_class << 2) | neighbor_nz
```

Cap at 64 contexts/layer; share probability tables across similar layers.

### rANS Encoding

- **Interleaved rANS:** Maintain multiple ANS states for parallel decode
- **Independent streams:** Each tile (or small tile group) has independent rANS stream with byte-aligned boundaries
- **Probability model:** Adaptive or static frequency tables per context; update after encoding tile group

**Benefits:**
- Parallel decode: GPU threads decode tiles independently
- Random access: Can decode single layer or tile without full checkpoint parse
- 20–40% better compression than flat Huffman

---

## Container Format (.wcodec)

Binary container with:
- Global header
- Per-layer records
- Per-tile records
- rANS streams (byte-aligned)

### File Structure

```
┌─────────────────────────────────────┐
│ Magic: "WCODEC\0\0" (8 bytes)       │
├─────────────────────────────────────┤
│ Version: major.minor (uint16 × 2)  │
├─────────────────────────────────────┤
│ Model hash (SHA256, 32 bytes)      │
├─────────────────────────────────────┤
│ Base quant: INT8=1, INT4=2 (uint8) │
├─────────────────────────────────────┤
│ Default tile size (uint8 × 2)      │
├─────────────────────────────────────┤
│ Num layers (uint32)                 │
├─────────────────────────────────────┤
│ Layer index table offset (uint64)  │
└─────────────────────────────────────┘
         (128-byte aligned)

┌─────────────────────────────────────┐
│ Layer 0 Record                       │
│  - Layer name (null-term string)    │
│  - Shape: [rows, cols] (uint32 × 2) │
│  - Tile grid: [n_row, n_col]       │
│  - Context table offset/size        │
│  - Tile record offset               │
│  - Checksum (CRC32)                 │
└─────────────────────────────────────┘
         ...
┌─────────────────────────────────────┐
│ Layer N Record                       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Tile Records (per layer)            │
│  - Predictor ID (uint8)             │
│  - Transform map (bitpacked)        │
│  - rANS stream offset (uint64)     │
│  - rANS stream length (uint32)     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ rANS Streams (byte-aligned)         │
│  - Compressed bitplane data         │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Footer: magic + checksum            │
└─────────────────────────────────────┘
```

**Alignment:** 128-byte boundaries for GPU-friendly access.

**Checksums:** Per-layer CRC32 for corruption detection.

---

## Decode Path

### CPU Decode (Fallback)

```python
def decode_layer(layer_record):
    tiles = []
    for tile_rec in layer_record.tiles:
        # 1. rANS decode
        bitplanes = rans_decode(tile_rec.stream, contexts)
        
        # 2. Unpack coefficients
        coeffs = bitplane_unpack(bitplanes)
        
        # 3. Inverse transform (8×8 sub-blocks)
        residual = inverse_transform(coeffs, tile_rec.transform_map)
        
        # 4. Add prediction
        pred = compute_prediction(tile_rec.predictor_id, reconstructed_neighbors)
        tile = residual + pred
        
        tiles.append(tile)
    
    # Reassemble layer
    return assemble_tiles(tiles, layer_record.shape)
```

### GPU Decode (CUDA)

**Strategy:** Parallelize across tiles; each thread block decodes one tile.

```cuda
__global__ void decode_tiles_kernel(TileRecord* tiles, uint8_t* streams, 
                                     int* output, int num_tiles) {
    int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;
    
    TileRecord tile = tiles[tile_id];
    
    // 1. Thread-parallel rANS decode into shared memory
    __shared__ int16_t coeffs[16*16];
    rans_decode_tile(streams + tile.stream_offset, tile.stream_length, coeffs);
    
    // 2. Inverse transform (cooperatively across threads)
    __shared__ int16_t residual[16*16];
    inverse_dct_8x8_parallel(coeffs, residual, tile.transform_map);
    
    __syncthreads();
    
    // 3. Add prediction (boundary tiles read from global; others from shared)
    int16_t pred = compute_prediction_gpu(tile.predictor_id, tile_id, output);
    int idx = threadIdx.x;
    if (idx < 256) {
        output[tile.output_offset + idx] = residual[idx] + pred;
    }
}
```

**Performance target:** Decode entire 8B model in <500ms on RTX 5090.

---

## Integration Points

### PyTorch

```python
import wcodec

# Encode
wcodec.encode_checkpoint(
    input_path="llama3_8b_int8.safetensors",
    output_path="llama3_8b.wcodec",
    tile_size=16,
    predictor_modes=["left", "top", "avg", "planar"],
    transforms=["none", "dct", "adst"]
)

# Decode (automatic in model load)
model = torch.load("llama3_8b.wcodec", map_location="cuda")
# Transparent decode-on-load via custom deserializer
```

### TensorRT

Hook into `IPluginV2` or `INetworkDefinition` loader to decode weights on engine build.

---

## Performance Targets (M1 KPIs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **File size reduction** | ≥30–60% vs flat INT8/INT4 | Total checkpoint size |
| **Decode latency** | ≤ model warm-up time | P50/P95 per layer |
| **Accuracy delta** | ≤0.1 pp | MMLU, GSM8K, HellaSwag |
| **Bit-exactness** | 100% | Round-trip checksum match |
| **Parallelism** | >80% GPU util during decode | nsys profile |

---

## Future Extensions (P2, P3)

- **Rate-distortion optimization:** Per-tile bit allocation based on sensitivity (Fisher/grad)
- **Progressive coding:** Stream MSB planes first; refine with LSB planes on demand
- **Inter-checkpoint deltas:** Compress fine-tunes as residuals vs base model
- **Low-rank residuals:** Structured enhancements for faster apply

---

## References

- AV1 Bitstream & Decoding Process Specification
- JPEG2000 (wavelet + bitplane coding)
- rANS: Duda (2013), "Asymmetric Numeral Systems"
- SPIHT: Said & Pearlman (1996), zerotree coding
- Neural network quantization: GPTQ, AWQ, SmoothQuant

---

## Changelog

- **v0.1.0 (2025-10-04):** Initial draft specification

---

**Contact:** [Your research team]  
**License:** TBD (likely Apache 2.0 or MIT for tooling)

