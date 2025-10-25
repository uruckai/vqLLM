# Speed Optimization Guide - Easy Wins

## Current Bottlenecks

Looking at the current rANS implementation, here are the main bottlenecks:

1. **Creating/destroying encoder per tile** (biggest bottleneck)
2. **Tile-by-tile processing** (sequential, not parallel)
3. **CPU-based encoding** (no GPU acceleration)
4. **Small tile size** (256x256 = lots of overhead)
5. **Memory copies** (NumPy ↔ ctypes ↔ C++)

## Easy Wins (Ordered by Impact)

### 🥇 #1: Reuse Encoder/Decoder (10-50x speedup)

**Current code:**
```python
for i in range(num_tiles):
    encoder = lib.encoder_create(tile_size)  # NEW encoder per tile! ❌
    # ... encode ...
    lib.encoder_destroy(encoder)             # Destroy immediately
```

**Problem:**
- Creating encoder allocates memory, builds tables
- Destroying frees everything
- Do this 170+ times per layer = MASSIVE overhead!

**Fix:**
```python
encoder = lib.encoder_create(tile_size)  # Create ONCE ✅
for i in range(num_tiles):
    # ... encode (reuse same encoder) ...
lib.encoder_destroy(encoder)              # Destroy ONCE at end
```

**Impact:** 10-50x speedup on compression
**Effort:** 5 minutes (move 2 lines)
**Risk:** None (this is how it should work anyway)

---

### 🥈 #2: Increase Tile Size (2-5x speedup)

**Current:** tile_size = 256 (256×256 = 65KB per tile)
**Problem:** Small tiles = more overhead per tile

**Fix:**
```python
tile_size = 512  # or 1024
```

**Trade-offs:**

| Tile Size | Tiles per Layer | Overhead | Memory |
|-----------|-----------------|----------|--------|
| 256×256 | ~170 | High | 256KB |
| 512×512 | ~43 | Medium | 1MB |
| 1024×1024 | ~11 | Low | 4MB |

**Recommendation:** Start with 512, try 1024 if no OOM

**Impact:** 2-5x speedup
**Effort:** 1 line change
**Risk:** Possible OOM with 1024 (test on 512 first)

---

### 🥉 #3: Pre-allocate Buffers (1.5-2x speedup)

**Current:** Every tile allocates new numpy arrays

**Fix:**
```python
# Pre-allocate reusable buffers
tile_buffer = np.zeros((tile_size, tile_size), dtype=np.int8)

for i in range(num_tiles):
    # Reuse tile_buffer instead of creating new arrays
    tile_buffer[:] = 0  # Clear
    tile_buffer.flat[:len(tile_data)] = tile_data  # Fill
    # ... encode tile_buffer ...
```

**Impact:** 1.5-2x speedup
**Effort:** 10 minutes
**Risk:** None

---

### 🏅 #4: Batch Tile Processing (2-3x speedup)

**Current:** Process one tile at a time

**Fix:** Process multiple tiles before freeing

```python
encoder = lib.encoder_create(tile_size)
compressed_batch = []

for i in range(num_tiles):
    # Encode tile
    output_ptr = ctypes.POINTER(ctypes.c_uint8)()
    output_size = ctypes.c_size_t()
    
    lib.encoder_encode(encoder, data_ptr, ...)
    
    # Copy compressed data (don't free yet)
    compressed = bytes(ctypes.cast(output_ptr, ...).contents)
    compressed_batch.append((output_ptr, compressed))
    
    # Free in batches of 10
    if len(compressed_batch) >= 10:
        for ptr, _ in compressed_batch:
            lib.free_buffer(ptr)
        compressed_batch.clear()

# Free remaining
for ptr, _ in compressed_batch:
    lib.free_buffer(ptr)

lib.encoder_destroy(encoder)
```

**Impact:** 2-3x speedup (reduces malloc/free overhead)
**Effort:** 20 minutes
**Risk:** Slightly more memory usage

---

### 🎖️ #5: Use Contiguous Memory (1.2-1.5x speedup)

**Current:** reshape() might create non-contiguous arrays

**Fix:**
```python
# Ensure contiguous memory before passing to C
tile_2d = tile_data.reshape(tile_size, tile_size)
if not tile_2d.flags['C_CONTIGUOUS']:
    tile_2d = np.ascontiguousarray(tile_2d)
```

**Impact:** 1.2-1.5x speedup
**Effort:** 5 minutes
**Risk:** None

---

### 🏆 #6: Parallel Tile Encoding (5-10x speedup)

**Current:** Sequential tile processing

**Fix:** Use Python multiprocessing or threading

```python
from multiprocessing import Pool

def encode_tile(tile_data):
    encoder = lib.encoder_create(tile_size)
    # ... encode ...
    lib.encoder_destroy(encoder)
    return compressed

# Parallel encoding
with Pool(processes=4) as pool:
    compressed_tiles = pool.map(encode_tile, tiles)
```

**Impact:** 5-10x speedup on multi-core CPU
**Effort:** 1 hour
**Risk:** Need thread-safe encoder (might need changes to C++ code)

---

## Recommended Implementation Order

### Phase 1: Quick Wins (30 minutes, 20-100x speedup)
1. ✅ **Reuse encoder/decoder** (#1)
2. ✅ **Increase tile size to 512** (#2)
3. ✅ **Pre-allocate buffers** (#3)

**Expected result:** 20-100x faster compression/decompression

### Phase 2: Medium Effort (1 hour, additional 2-5x)
4. ✅ **Batch tile processing** (#4)
5. ✅ **Contiguous memory** (#5)

**Expected result:** 40-500x faster total

### Phase 3: Advanced (2-4 hours, additional 5-10x)
6. ⏳ **Parallel tile encoding** (#6)
7. ⏳ **GPU decompression** (use existing CUDA kernel)

**Expected result:** 200-5000x faster total

---

## Optimization #1 Implementation (HIGHEST PRIORITY)

Here's the exact fix for the biggest bottleneck:

**File:** `test_rans_fp16_multilayer.py`

**Current (SLOW):**
```python
def compress_fp16_layer(weight_np, tile_size=256):
    # ...
    for i in range(num_tiles):
        start = i * tile_elements
        end = min(start + tile_elements, len(weight_int8_view))
        tile_data = weight_int8_view[start:end]
        
        # SLOW: Create encoder per tile ❌
        encoder = lib.encoder_create(tile_size)
        
        # ... encoding ...
        
        # SLOW: Destroy encoder per tile ❌
        lib.encoder_destroy(encoder)
```

**Fixed (FAST):**
```python
def compress_fp16_layer(weight_np, tile_size=512):  # Also increase tile_size
    # ...
    
    # Create encoder ONCE ✅
    encoder = lib.encoder_create(tile_size)
    
    for i in range(num_tiles):
        start = i * tile_elements
        end = min(start + tile_elements, len(weight_int8_view))
        tile_data = weight_int8_view[start:end]
        
        # ... encoding (reuse encoder) ...
        
        lib.free_buffer(output_ptr)  # Still free each buffer
    
    # Destroy encoder ONCE ✅
    lib.encoder_destroy(encoder)
```

**Same fix for decompression:**
```python
def decompress_fp16_layer(compressed_tiles, rows, cols, tile_size=512):
    # ...
    
    # Create decoder ONCE ✅
    decoder = lib.decoder_create()
    
    for compressed in compressed_tiles:
        # ... decoding (reuse decoder) ...
    
    # Destroy decoder ONCE ✅
    lib.decoder_destroy(decoder)
```

---

## Expected Results

### Current Performance (rough estimates):
```
Compress 1 layer (11MB):   ~2000ms
Decompress 1 layer:        ~1500ms
Total per layer:           ~3500ms

5 layers total:            ~17,500ms (17.5 seconds)
```

### After Phase 1 Optimizations:
```
Compress 1 layer:          ~100ms   (20x faster)
Decompress 1 layer:        ~75ms    (20x faster)
Total per layer:           ~175ms   (20x faster)

5 layers total:            ~875ms   (0.9 seconds)
```

### After Phase 2 Optimizations:
```
Compress 1 layer:          ~50ms    (40x faster)
Decompress 1 layer:        ~30ms    (50x faster)
Total per layer:           ~80ms    (44x faster)

5 layers total:            ~400ms   (0.4 seconds)
```

### After Phase 3 (Parallel + GPU):
```
Compress 1 layer:          ~10ms    (200x faster)
Decompress 1 layer:        ~5ms     (300x faster, GPU)
Total per layer:           ~15ms    (233x faster)

5 layers total:            ~75ms    (0.075 seconds)
```

---

## Priority Recommendation

**Start with Phase 1 - it's literally a 5-minute fix for 20-100x speedup!**

1. Move `encoder_create()` outside the loop
2. Change tile_size from 256 → 512
3. Pre-allocate tile buffer

These three changes will make compression go from ~17 seconds to ~1 second for 5 layers.

Want me to implement these optimizations now?

