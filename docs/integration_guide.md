# Weight Codec Integration Guide

**Version:** 0.1.0  
**Date:** 2025-10-04

## Overview

This guide shows how to integrate Weight Codec (WCodec) into your LLM inference pipeline for transparent decode-on-load with zero runtime overhead.

---

## Quick Start

### Installation

```bash
# Clone repo
git clone https://github.com/yourorg/weight-codec.git
cd weight-codec

# Build C++ library and CUDA decoder
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
make -j8

# Install Python bindings
cd ../python
pip install -e .
```

### Encode a Checkpoint

```bash
# Convert quantized safetensors to .wcodec
wcodec-encode \
  --input llama3_8b_int8.safetensors \
  --output llama3_8b.wcodec \
  --tile-size 16 \
  --predictors left,top,avg,planar \
  --transforms none,dct,adst \
  --threads 16
```

**Expected output:**
```
Encoding llama3_8b_int8.safetensors...
  Original size: 8.03 GB
  Encoded size:  4.52 GB
  Reduction:     43.7%
  Time:          24.3s
  Checksum:      a3f2...d891
Saved to llama3_8b.wcodec
```

### Decode and Load

```python
import torch
import wcodec

# Option 1: Explicit decode to safetensors
wcodec.decode_checkpoint(
    input_path="llama3_8b.wcodec",
    output_path="llama3_8b_decoded.safetensors"
)
model = torch.load("llama3_8b_decoded.safetensors")

# Option 2: Decode-on-load (transparent)
model = wcodec.load_model("llama3_8b.wcodec", device="cuda")
# Internally decodes to GPU buffers; no intermediate file
```

---

## Detailed Integration

### PyTorch

#### Custom Deserializer

Register a custom deserializer to intercept `.wcodec` files:

```python
import torch
from torch.serialization import register_package

@register_package("wcodec")
def load_wcodec(file_path, map_location=None):
    import wcodec
    
    # Decode directly to target device
    state_dict = wcodec.decode_to_state_dict(
        file_path, 
        device=map_location or "cuda"
    )
    
    # Build model and load state
    model = YourModelClass()
    model.load_state_dict(state_dict)
    return model

# Now torch.load works transparently
model = torch.load("model.wcodec", map_location="cuda")
```

#### Hugging Face Transformers

Patch `PreTrainedModel.from_pretrained`:

```python
from transformers import AutoModelForCausalLM
import wcodec

# Monkey-patch (or submit PR to transformers)
_original_from_pretrained = AutoModelForCausalLM.from_pretrained

def from_pretrained_wcodec(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path.endswith(".wcodec"):
        # Decode to temp dir
        temp_path = wcodec.decode_to_temp(pretrained_model_name_or_path)
        return _original_from_pretrained(cls, temp_path, *args, **kwargs)
    else:
        return _original_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained_wcodec)

# Usage
model = AutoModelForCausalLM.from_pretrained("llama3_8b.wcodec")
```

---

### TensorRT

#### Plugin Approach

Implement a custom `IPluginV2DynamicExt` that decodes weights during engine build:

```cpp
class WCodecDecoderPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    WCodecDecoderPlugin(const std::string& wcodec_path) 
        : wcodec_path_(wcodec_path) {}
    
    int32_t initialize() noexcept override {
        // Decode .wcodec to GPU buffers
        decoder_ = wcodec::CudaDecoder::create(wcodec_path_);
        return 0;
    }
    
    void configurePlugin(/* ... */) override {
        // Allocate GPU buffers for decoded weights
        decoder_->decode_all(gpu_buffers_);
    }
    
    // ... rest of plugin interface
private:
    std::string wcodec_path_;
    std::unique_ptr<wcodec::CudaDecoder> decoder_;
    std::vector<void*> gpu_buffers_;
};
```

#### Builder Integration

```python
import tensorrt as trt
import wcodec

# Build engine with WCodec weights
builder = trt.Builder(logger)
network = builder.create_network(flags)

# Decode weights on-the-fly
for layer_name in model_layers:
    weights = wcodec.decode_layer(
        "model.wcodec", 
        layer_name, 
        device="cuda"
    )
    
    layer = network.add_fully_connected(
        input=prev_layer,
        num_outputs=weights.shape[0],
        kernel=trt.Weights(weights.data_ptr(), weights.numel())
    )

engine = builder.build_engine(network, config)
```

---

### vLLM

Integrate into `ModelLoader`:

```python
# vllm/model_executor/model_loader.py

from wcodec import decode_checkpoint

def load_weights(model, model_name_or_path):
    if model_name_or_path.endswith(".wcodec"):
        # Decode to GPU buffers directly
        state_dict = decode_checkpoint(
            model_name_or_path,
            device="cuda",
            stream=torch.cuda.current_stream()
        )
        model.load_state_dict(state_dict)
    else:
        # Standard safetensors path
        load_safetensors(model, model_name_or_path)
```

---

## Performance Optimization

### Parallel Decode

Decode multiple layers concurrently:

```python
import wcodec
import concurrent.futures

def decode_layer_async(layer_name):
    return wcodec.decode_layer("model.wcodec", layer_name, device="cuda")

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    layer_names = ["layer.0", "layer.1", "layer.2", "layer.3"]
    futures = [executor.submit(decode_layer_async, name) for name in layer_names]
    
    for future in concurrent.futures.as_completed(futures):
        decoded_layer = future.result()
        # Use immediately
```

### Persistent Cache

Cache decoded tensors on fast NVMe to avoid repeated decode:

```python
import wcodec
import hashlib
import os

def load_with_cache(wcodec_path, cache_dir="/tmp/wcodec_cache"):
    # Compute hash
    with open(wcodec_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    
    cache_path = os.path.join(cache_dir, f"{file_hash}.safetensors")
    
    if os.path.exists(cache_path):
        # Load from cache
        return torch.load(cache_path)
    else:
        # Decode and cache
        state_dict = wcodec.decode_to_state_dict(wcodec_path)
        torch.save(state_dict, cache_path)
        return state_dict
```

### GPU Decode Tuning

Tune kernel launch parameters:

```python
wcodec.set_decode_params(
    tiles_per_block=4,      # Process 4 tiles per thread block
    streams=4,               # Parallel CUDA streams
    use_tensor_cores=True   # Use TC for inverse transforms (if beneficial)
)
```

---

## Fallback and Compatibility

### Auto-Fallback

If CUDA decoder unavailable, automatically fall back to CPU:

```python
import wcodec

# Tries GPU first, falls back to CPU if needed
state_dict = wcodec.decode_checkpoint(
    "model.wcodec",
    device="cuda",
    fallback="cpu"
)
```

### Mixed Loading

Serve models with both `.wcodec` and `.safetensors` clients:

```python
def load_model_universal(path):
    if path.endswith(".wcodec"):
        return wcodec.load_model(path, device="cuda")
    else:
        return torch.load(path, map_location="cuda")
```

---

## Validation and Testing

### Round-Trip Test

Verify bit-exact reconstruction:

```bash
# Encode
wcodec-encode --input model_int8.safetensors --output model.wcodec

# Decode
wcodec-decode --input model.wcodec --output model_decoded.safetensors

# Verify
wcodec-verify --original model_int8.safetensors --decoded model_decoded.safetensors
```

**Expected output:**
```
Comparing tensors...
  All 312 layers match bit-exactly ✓
  Checksum: OK
  Max abs diff: 0
```

### Accuracy Test

Run standard eval suite:

```bash
python scripts/eval_accuracy.py \
  --model-path model.wcodec \
  --tasks mmlu,gsm8k,hellaswag \
  --baseline model_int8.safetensors
```

**Target:** Δacc ≤ 0.1 pp across all tasks.

---

## Troubleshooting

### Decode Slower than Expected

- **Check:** GPU decode enabled? Run `wcodec.is_cuda_available()`
- **Check:** Decoding single-threaded? Use `--threads N` for CPU or multiple CUDA streams
- **Fix:** Increase `tiles_per_block` if GPU occupancy low

### Accuracy Regression

- **Check:** Did you encode a quantized checkpoint (INT8/INT4), not FP16?
- **Check:** Verify bit-exact round-trip with `wcodec-verify`
- **Fix:** If mismatch found, file a bug report with layer name and checksum

### Out of Memory During Decode

- **Cause:** Decoding entire model at once
- **Fix:** Decode layer-by-layer and move to CPU after use:
  ```python
  for layer_name in model.layers:
      layer_weights = wcodec.decode_layer("model.wcodec", layer_name, device="cuda")
      model.load_layer(layer_name, layer_weights)
      del layer_weights  # Free GPU memory
  ```

---

## API Reference

### Python API

```python
wcodec.encode_checkpoint(
    input_path: str,
    output_path: str,
    tile_size: int = 16,
    predictor_modes: List[str] = ["left", "top", "avg", "planar"],
    transforms: List[str] = ["none", "dct", "adst"],
    threads: int = os.cpu_count()
) -> Dict[str, Any]

wcodec.decode_checkpoint(
    input_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    fallback: str = "cpu"
) -> Union[Dict[str, torch.Tensor], None]

wcodec.decode_layer(
    wcodec_path: str,
    layer_name: str,
    device: str = "cuda"
) -> torch.Tensor

wcodec.load_model(
    wcodec_path: str,
    model_class: Type,
    device: str = "cuda"
) -> nn.Module

wcodec.is_cuda_available() -> bool

wcodec.set_decode_params(
    tiles_per_block: int = 4,
    streams: int = 4,
    use_tensor_cores: bool = False
) -> None
```

### C++ API

```cpp
namespace wcodec {

class Encoder {
public:
    explicit Encoder(const Config& config);
    void encode_checkpoint(const std::string& input_path,
                          const std::string& output_path);
};

class Decoder {
public:
    explicit Decoder(const std::string& wcodec_path);
    void decode_all(std::vector<Tensor>& output);
    Tensor decode_layer(const std::string& layer_name);
};

class CudaDecoder : public Decoder {
public:
    static std::unique_ptr<CudaDecoder> create(const std::string& wcodec_path);
    void decode_all_gpu(void** gpu_buffers);
};

} // namespace wcodec
```

---

## Best Practices

1. **Always verify:** Run `wcodec-verify` after encoding to ensure bit-exact reconstruction
2. **Cache decoded weights:** Use NVMe cache for repeated loads (e.g., multi-GPU servers)
3. **Profile decode time:** Ensure ≤ model warm-up time; if not, file performance report
4. **Version control:** Store both `.wcodec` and original `.safetensors` during development
5. **Monitor accuracy:** Run eval suite before/after encoding; set CI gate at Δacc ≤ 0.1 pp

---

## Examples

See `examples/` directory:
- `examples/pytorch_simple.py` — Basic PyTorch integration
- `examples/transformers_llama.py` — Hugging Face Transformers + Llama
- `examples/tensorrt_engine.py` — TensorRT engine builder
- `examples/vllm_integration.py` — vLLM server integration

---

## Support

- **Issues:** https://github.com/yourorg/weight-codec/issues
- **Discussions:** https://github.com/yourorg/weight-codec/discussions
- **Email:** wcodec-dev@yourorg.com

---

**Last updated:** 2025-10-04

